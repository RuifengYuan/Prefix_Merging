import time,copy
import argparse
import math
import torch
import torch.nn as nn
from rouge import Rouge
from transformers import T5Tokenizer,BartTokenizer
from transformers import AdamW
from torch.optim import *
import numpy as np
from model.model_T5 import generator,generator_prefix
from data_loader import data_loader
class Train(object):

    def __init__(self, config):
        self.config = config  
        
        seed = self.config.seed
        torch.manual_seed(seed)            
        torch.cuda.manual_seed(seed)       
        torch.cuda.manual_seed_all(seed)       
        
        if 'bart' in self.config.pretrained_model:
            self.tokenizer = BartTokenizer.from_pretrained(self.config.pretrained_model)
            
        self.log = open('log.txt','w')
        
        self.dataloader=data_loader('train', self.config, self.tokenizer, load_xsum=self.config.load_xsum, load_debate=self.config.load_debate, load_squad=self.config.load_squad, load_kptimes=self.config.load_kptimes, mix=self.config.mix, low_data=self.config.low_data)


        if self.config.mid_start == 0:
            if self.config.promote==1:        
                self.generator = generator_prefix(self.config)
            else:
                self.generator = generator(self.config)
        else:

            if self.config.promote==0:        
                x=torch.load('save_model/'+self.config.load_model,map_location='cpu')
                self.generator = x['generator']
            else:
                self.generator = generator_prefix(self.config)
                
                
                x=torch.load('save_model/'+self.config.load_model,map_location='cpu')
                load_model = x['generator']  
                load_dict = load_model.state_dict()
                load_config = x['config'] 
                load_prefix_emb_size=load_config.prefix_length
                
                forbid_layer=[]            
                model_dict = self.generator.state_dict()
                load_dict = {k:v for k,v in load_dict.items() if (k in model_dict and k not in forbid_layer)}
                model_dict.update(load_dict)
                self.generator.load_state_dict(model_dict)
    

        self.generator.cuda()
    
    
        if self.config.promote==1:
            param_optimizer = list(self.generator.prefix_layer.named_parameters())
            #param_optimizer = list(self.generator.prefix_layer.named_parameters())
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer], 'weight_decay': 0}]
            for param in self.generator.model.parameters():
                param.requires_grad = False
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr)
            
        else:
            param_optimizer = list(self.generator.named_parameters())
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer], 'weight_decay': 0}]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr)        

        if self.config.use_lr_decay == 1:
            scheduler=lr_scheduler.StepLR(self.optimizer,step_size=1,gamma = self.config.lr_decay)

        
    def save_model(self, running_avg_loss,loss_list,rouge1,rouge2,loss_text=''):

        state = {
            'iter': self.dataloader.count/self.config.true_batch_size*self.config.batch_size,
            'ecop': self.dataloader.epoch,
            'generator':self.generator,
            'current_loss': running_avg_loss,
            'loss_list': loss_list,
            'rouge1':rouge1,
            'rouge2':rouge2,
            'config':self.config
        }
        
        model_save_path = self.config.save_path+str(self.dataloader.count/self.config.true_batch_size*self.config.batch_size)+'_iter_'+str(self.dataloader.epoch) +'_epoch__rouge_'+str(rouge1)+'_'+str(rouge2)+'__loss_'+str(running_avg_loss)+loss_text
        torch.save(state, model_save_path)
        
    def train_one_batch(self):

        try:
            article_id,article_id_mask,summary_id,summary_id_mask,summary,label1,label2 = \
            self.dataloader.load_data()

        except:
            print('fail to load the data')
            return 0,0
        
        input_id=article_id
        input_id_mask=article_id_mask
        
        decode_id=torch.cat([torch.full((summary_id.size()[0],1), self.config.bos_token_id, dtype=torch.long).cuda(),summary_id[:,1:-1]],1)
        decode_id_mask=summary_id_mask[:,:-1]
        gold_id=summary_id[:,1:]
        gold_id_mask=summary_id_mask[:,1:]
        
        
        output = self.generator(input_id,input_id_mask,decode_id,decode_id_mask,label1,label2)
        output = output[0]
 

        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        loss = ce_loss_fct(output.view(-1, output.shape[-1]), gold_id.reshape(1,-1).squeeze(0))

        
        #print(loss,loss2)

        loss.backward()
        return loss.item(),1,label1,label2
    
    def train_iter(self):
        loss_list=[]
        loss_list_xsum=[0]
        loss_list_debate=[0]
        loss_list_squad=[0]
        loss_list_kptimes=[0]
        count=0
        self.generator.train()
        for i in range(self.config.max_epoch*self.config.train_set_len):
            count=count+1
            time_start=time.time()
            
            success=0
            for j in range(int(self.config.true_batch_size/self.config.batch_size)):     
                loss,tag,label1,label2 = self.train_one_batch()
                if tag == 1:
                    loss_list.append(loss)
                    success=success+1
                    if label1==0 and label2==0:
                        loss_list_debate.append(loss)
                    if label1==0 and label2==1:
                        loss_list_xsum.append(loss)
                    if label1==1 and label2==0:
                        loss_list_squad.append(loss)
                    if label1==1 and label2==1:
                        loss_list_kptimes.append(loss)
                if tag == 0:
                    print('one mini batch fail')                            
                    continue
            if success == int(self.config.true_batch_size/self.config.batch_size):
                
                self.optimizer.step()                         
                self.optimizer.zero_grad()

                if self.config.use_lr_decay == 1:
                    if count%self.config.lr_decay_step == 0:
                        self.scheduler.step()         
              

            else:
                print('jump one batch')     
                
            time_end=time.time()                
                
            if count % self.config.checkfreq == 0:       
                
                recent_loss=loss_list[max(0,len(loss_list)-1000*int(self.config.true_batch_size/self.config.batch_size)):]
                recent_loss_debate=loss_list_debate[max(0,len(loss_list_debate)-100*int(self.config.true_batch_size/self.config.batch_size)):]
                recent_loss_xsum=loss_list_xsum[max(0,len(loss_list_xsum)-100*int(self.config.true_batch_size/self.config.batch_size)):]
                recent_loss_squad=loss_list_squad[max(0,len(loss_list_squad)-100*int(self.config.true_batch_size/self.config.batch_size)):]                    
                recent_loss_kptimes=loss_list_kptimes[max(0,len(loss_list_kptimes)-100*int(self.config.true_batch_size/self.config.batch_size)):]
                    
                avg_loss=sum(recent_loss)/len(recent_loss)
                print(str(count)+' iter '+str(self.dataloader.epoch) +' epoch avg_loss:'+str(avg_loss)[:5]+' debate_loss:'+str(np.mean(recent_loss_debate))[:5]+\
                      ' xsum_loss:'+str(np.mean(recent_loss_xsum))[:5]+' squad_loss:'+str(np.mean(recent_loss_squad))[:5]+' kptimes_loss:'+str(np.mean(recent_loss_kptimes))[:5]+' time:'+str(time_end-time_start))        
                
            if count % self.config.savefreq == 0 and count > self.config.savefreq-100 and count > self.config.startfreq:     
                recent_loss=loss_list[max(0,len(loss_list)-1000*int(self.config.true_batch_size/self.config.batch_size)):]
                avg_loss=sum(recent_loss)/len(recent_loss)
                 
                print('start val')
                rouge1,rouge2=self.do_val(1000)  
                print(rouge1,rouge2)
                
                loss_text=''
                if self.config.load_xsum == 1:
                    loss_text+=' xsum_loss:'+str(np.mean(recent_loss_xsum))[:5]
                if self.config.load_debate == 1:
                    loss_text+=' debate_loss:'+str(np.mean(recent_loss_debate))[:5]
                if self.config.load_squad == 1:
                    loss_text+=' squad_loss:'+str(np.mean(recent_loss_squad))[:5]
                if self.config.load_kptimes == 1:
                    loss_text+=' kptimes_loss:'+str(np.mean(recent_loss_kptimes))[:5]
                
                self.save_model(avg_loss,loss_list,rouge1,rouge2,loss_text) 
                self.generator.train()
                           
    def do_val(self, val_num):

        self.raw_rouge=Rouge()
        self.generator.eval()
        
        val_config=copy.deepcopy(self.config)
        val_config.true_batch_size=8
        val_config.buffer_size=8
        val_config.batch_size=4
        if (self.config.load_xsum == 1 and self.config.load_debate ==1) or (self.config.load_squad == 1 and self.config.load_debate ==1):
            data_loader_val = data_loader('val', val_config, self.tokenizer, load_xsum=0, load_debate=1, load_squad=0, load_kptimes=0, mix=1)
        elif self.config.load_squad == 1 and self.config.load_xsum ==1 and self.config.load_debate !=1:
            data_loader_val = data_loader('val', val_config, self.tokenizer, load_xsum=1, load_debate=0, load_squad=0, load_kptimes=0, mix=1)
        else:
            data_loader_val = data_loader('val', val_config, self.tokenizer, load_xsum=self.config.load_xsum, load_debate=self.config.load_debate, load_squad=self.config.load_squad, load_kptimes=self.config.load_kptimes, mix=self.config.mix)
      
        r1=[]
        r2=[]
        rl=[]
        for i in range(int(val_num/val_config.batch_size)):       
            try:
                article_id_b,article_id_mask_b,summary_i_b,summary_id_mask_b,summary_b,label1,label2 = \
                data_loader_val.load_data()
            except:
                print('load data fail during the evaluation')
                continue
            
            if (self.config.load_xsum == 1 and self.config.load_debate ==1) or (self.config.load_squad == 1 and self.config.load_debate ==1):
                if label1==0 and label2==0:            
                    divide=1
                    start=0
                    for mini in range(int(val_config.batch_size/divide)):
                        try:
                            article_id=article_id_b[start:start+divide]
                            article_id_mask=article_id_mask_b[start:start+divide]
                            gold=summary_b[start]

                            input_id=article_id
                            input_id_mask=article_id_mask
                            
                            start=start+divide
                            
                            output = self.generator.inference(input_id,input_id_mask,label1,label2,use_beam=2)
                            pred = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                             
                            scores = self.raw_rouge.get_scores(pred, gold)
                            r1.append(scores[0]['rouge-1']['f'])
                            r2.append(scores[0]['rouge-2']['f'])    
        
                        except:
                            print('one sample batch fail') 
            else:
                divide=1
                start=0
                for mini in range(int(val_config.batch_size/divide)):
                    try:
                        article_id=article_id_b[start:start+divide]
                        article_id_mask=article_id_mask_b[start:start+divide]
                        gold=summary_b[start]

                        input_id=article_id
                        input_id_mask=article_id_mask
                        
                        start=start+divide
                        
                        output = self.generator.inference(input_id,input_id_mask,label1,label2,use_beam=2)
                        pred = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                        scores = self.raw_rouge.get_scores(pred, gold)
                        r1.append(scores[0]['rouge-1']['f'])
                        r2.append(scores[0]['rouge-2']['f'])    
                        rl.append(scores[0]['rouge-l']['f'])     
                    except:
                        print('one sample batch fail') 
                    
            if data_loader_val.epoch == 10:
                break
                    
        if len(r1) != 0 and len(r2) != 0:
            print(np.mean(r1),np.mean(r2),np.mean(rl))
            return np.mean(r1),np.mean(r2)
        else:
            return 0,0               



class Test(object):
    
    def __init__(self, config):
        x=torch.load('save_model/'+config.test_model,map_location='cpu')
        self.generator = x['generator']      
        self.config = x['config']
        
        self.config.true_batch_size=1
        self.config.buffer_size=1
        self.config.batch_size=1
        

        self.config.seed=10   
        self.config.only_target=0
        self.config.use_pretrained_seed=0
        self.config.use_same_seed=0
        self.generator.cuda()
        
        if 'bart' in self.config.pretrained_model:
            self.tokenizer = BartTokenizer.from_pretrained(self.config.pretrained_model)
        
        self.raw_rouge=Rouge()
        
        self.can_path = 'result/'+config.test_model+'_cand.txt'

        self.gold_path ='result/'+config.test_model+'_gold.txt'
        
        self.source_path ='result/'+config.test_model+'_source.txt'


        
    
    def test(self,test_num=50):
        self.raw_rouge=Rouge()
        self.generator.eval()
        if (self.config.load_xsum == 1 and self.config.load_debate ==1) or (self.config.load_squad == 1 and self.config.load_debate ==1):
            data_loader_val = data_loader('test', self.config, self.tokenizer, load_xsum=0, load_debate=1, load_squad=0, load_kptimes=0, mix=1)
        elif self.config.load_squad == 1 and self.config.load_xsum ==1 and self.config.load_debate !=1:
            data_loader_val = data_loader('test', self.config, self.tokenizer, load_xsum=1, load_debate=0, load_squad=0, load_kptimes=0, mix=1)
        else:
            data_loader_val = data_loader('test', self.config, self.tokenizer, load_xsum=self.config.load_xsum, load_debate=self.config.load_debate, load_squad=self.config.load_squad, load_kptimes=self.config.load_kptimes, mix=self.config.mix)
      
        r1=[]
        r2=[]
        
        pred_list=[]
        gold_list=[]
        source_list=[]
        with open(self.can_path, 'w', encoding='utf-8') as save_pred:
            with open(self.gold_path, 'w', encoding='utf-8') as save_gold:
                with open(self.source_path, 'w', encoding='utf-8') as save_source:
        
                    for i in range(int(test_num/self.config.batch_size)): 
                        
                        if i%500 == 0:
                            print(i)
                        
                        try:
                            article_id_b,article_id_mask_b,summary_i_b,summary_id_mask_b,summary_b,label1,label2 = \
                            data_loader_val.load_data()
                        except:
                            print('load data fail during the evaluation')
                            break
                        
                        if (self.config.load_xsum == 1 and self.config.load_debate ==1) or (self.config.load_squad == 1 and self.config.load_debate ==1):
                            if label1==0 and label2==0:                         
                        
                                divide=1
                                start=0
                                for mini in range(int(self.config.batch_size/divide)):
                                    
                                    try:
                                        article_id=article_id_b[start:start+divide]
                                        article_id_mask=article_id_mask_b[start:start+divide]
                                        gold=summary_b[start]
                                        gold=summary_b[start]
                                        
                                        input_id=article_id
                                        input_id_mask=article_id_mask
                            
                                        start=start+divide
            
                                        output = self.generator.inference(input_id,input_id_mask,label1,label2,use_beam=0)
                                        pred = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                        article=self.tokenizer.decode(article_id[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                        scores = self.raw_rouge.get_scores(pred, gold)
                                        r1.append(scores[0]['rouge-1']['f'])
                                        r2.append(scores[0]['rouge-2']['f'])    
                    
                                        pred_list.append(pred)
                                        gold_list.append(gold)  
                                        source_list.append(article)
                                        
                                    except Exception as e:
                                        print('one test batch fail')                  
                                        print('Reason for batch fail:', e)  
 
                        else:        
                            divide=1
                            start=0
                            for mini in range(int(self.config.batch_size/divide)):
                                
                                try:
                                    article_id=article_id_b[start:start+divide]
                                    article_id_mask=article_id_mask_b[start:start+divide]
                                    gold=summary_b[start]
                                    gold=summary_b[start]
                                    
                                    input_id=article_id
                                    input_id_mask=article_id_mask
                        
                                    start=start+divide
        
                                    output = self.generator.inference(input_id,input_id_mask,label1,label2,use_beam=0)
                                    pred = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                    article=self.tokenizer.decode(article_id[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                    scores = self.raw_rouge.get_scores(pred, gold)
                                    r1.append(scores[0]['rouge-1']['f'])
                                    r2.append(scores[0]['rouge-2']['f'])    
                                    
                                    pred_list.append(pred)
                                    gold_list.append(gold)  
                                    source_list.append(article)
                                    
                                except Exception as e:
                                    print('one test batch fail')                  
                                    print('Reason for batch fail:', e)   
    
                                
                        if data_loader_val.epoch == 2:
                            break
                        
                    
                    for sent in gold_list:
                        save_gold.write(sent.strip() + '\n')
                    for sent in pred_list:
                        save_pred.write(sent.strip() + '\n')
                    for sent in source_list:
                        save_source.write(sent.strip() + '\n')

        print(np.mean(r1),np.mean(r2))



def argLoader():

    parser = argparse.ArgumentParser()
    
    
    #device
    
    parser.add_argument('--device', type=int, default=1)    
    
    # Do What
    
    parser.add_argument('--do_train', action='store_true', help="Whether to run training")

    parser.add_argument('--do_test', action='store_true', help="Whether to run test")
    
    parser.add_argument('--promote', type=int, default=1)
    
    parser.add_argument('--low_data', type=int, default=0)

    parser.add_argument('--low_data_start', type=int, default=0)
    
    parser.add_argument('--low_data_num', type=int, default=50)
    
    parser.add_argument('--seed', type=int, default=10)

    parser.add_argument('--use_prefix', type=str, default='')

    parser.add_argument('--prefix_length', type=int, default=70)
    
    parser.add_argument('--unq_prefix_length', type=int, default=0)
    
    parser.add_argument('--share_prefix_length', type=int, default=40)
    
    parser.add_argument('--target_prefix_length', type=int, default=30)
    
    parser.add_argument('--only_target', type=int, default=0)
    
    parser.add_argument('--use_pretrained_seed', type=int, default=0)    
    
    parser.add_argument('--use_same_seed', type=int, default=0) 
    
    
    
    
    parser.add_argument('--mix', type=int, default=1)
    
    parser.add_argument('--load_xsum', type=int, default=0)
    
    parser.add_argument('--load_debate', type=int, default=1)
    
    parser.add_argument('--load_squad', type=int, default=0)
    
    parser.add_argument('--load_kptimes', type=int, default=0)

    parser.add_argument('--pretrained_model', type=str, default='facebook/bart-large')    
    
    parser.add_argument('--bos_token_id', type=int, default=0) 
    
    parser.add_argument('--pad_token_id', type=int, default=1) 
    
    parser.add_argument('--eos_token_id', type=int, default=2) 
    #Preprocess Setting
    parser.add_argument('--max_summary', type=int, default=40)

    parser.add_argument('--max_article', type=int, default=200)    
    
    #Model Setting
    parser.add_argument('--hidden_dim', type=int, default=1024)

    parser.add_argument('--emb_dim', type=int, default=1024)
    
    parser.add_argument('--vocab_size', type=int, default=50264)      

    parser.add_argument('--lr', type=float, default=5e-5)     
    
    parser.add_argument('--eps', type=float, default=1e-10)
    
    parser.add_argument('--prefix_dropout', type=float, default=0)    
        
    parser.add_argument('--batch_size', type=int, default=1)  

    parser.add_argument('--true_batch_size', type=int, default=12)  

    parser.add_argument('--buffer_size', type=int, default=48)      
    
    parser.add_argument('--scale_embedding', type=int, default=0)  
    
    #lr setting

    parser.add_argument('--use_lr_decay', type=int, default=0)  
    
    parser.add_argument('--lr_decay_step', type=int, default=10000)  
    
    parser.add_argument('--lr_decay', type=float, default=1)  

    # Testing setting
    parser.add_argument('--beam_size', type=int, default=2)
    
    parser.add_argument('--max_dec_steps', type=int, default=40)
    
    parser.add_argument('--min_dec_steps', type=int, default=10)
    
    parser.add_argument('--test_model', type=str, default='')   
    
    parser.add_argument('--load_model', type=str, default='')  
    
    parser.add_argument('--save_path', type=str, default='')  
    
    parser.add_argument('--mid_start', type=int, default=1)
   
    # Checkpoint Setting
    parser.add_argument('--max_epoch', type=int, default=120)
    
    parser.add_argument('--train_set_len', type=int, default=4)
    
    parser.add_argument('--savefreq', type=int, default=16)

    parser.add_argument('--checkfreq', type=int, default=1)    

    parser.add_argument('--startfreq', type=int, default=1)       
    
    args = parser.parse_args()
    
    return args





def main():

    args = argLoader()

    torch.cuda.set_device(args.device)

    print('CUDA', torch.cuda.current_device())
    

    if args.do_train:
        
        x=Train(args)
    
        x.trainIters()
        
    if args.do_test:
        
        x = Test(args)
        x.test()

main()
        