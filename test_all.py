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
import os

class Test(object):
    
    def __init__(self, config):
        x=torch.load('save_model/'+config.test_model,map_location='cpu')
        self.generator = x['generator']      
        self.config = x['config']
        
        self.config.true_batch_size=1
        self.config.buffer_size=1
        self.config.batch_size=1
        

        self.config.seed=10   
        self.config.only_target=1
        self.config.use_pretrained_seed=0
        self.config.use_same_seed=0
        self.generator.cuda()
        
        if 'bart' in self.config.pretrained_model:
            self.tokenizer = BartTokenizer.from_pretrained(self.config.pretrained_model)
        
        self.raw_rouge=Rouge()
        
        self.can_path = 'result/'+config.test_description+config.test_model+'/'+config.test_model+'_cand.txt'

        self.gold_path ='result/'+config.test_description+config.test_model+'/'+config.test_model+'_gold.txt'
        
        self.source_path ='result/'+config.test_description+config.test_model+'/'+config.test_model+'_source.txt'

        if not os.path.exists('result/'+config.test_description+config.test_model):
            os.mkdir('result/'+config.test_description+config.test_model)

    
    def test(self,test_num=21126):
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
        rl=[]        
        
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
            
                                        output = self.generator.inference(input_id,input_id_mask,label1,label2,use_beam=2)
                                        pred = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                        article=self.tokenizer.decode(article_id[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                        scores = self.raw_rouge.get_scores(pred, gold)
                                        r1.append(scores[0]['rouge-1']['f'])
                                        r2.append(scores[0]['rouge-2']['f'])    
                                        rl.append(scores[0]['rouge-l']['f'])

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
        
                                    output = self.generator.inference(input_id,input_id_mask,label1,label2,use_beam=2)
                                    pred = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                    article=self.tokenizer.decode(article_id[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                                    scores = self.raw_rouge.get_scores(pred, gold)
                                    r1.append(scores[0]['rouge-1']['f'])
                                    r2.append(scores[0]['rouge-2']['f'])    
                                    rl.append(scores[0]['rouge-l']['f'])

                
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

        print(np.mean(r1),np.mean(r2),np.mean(rl))



def argLoader():

    parser = argparse.ArgumentParser()
    
    
    #device
    
    parser.add_argument('--device', type=int, default=0)    
    
    # Do What
    
    parser.add_argument('--do_train', action='store_true', help="Whether to run training")

    parser.add_argument('--do_test', action='store_true', help="Whether to run test")
    
    parser.add_argument('--promote', type=int, default=0)
    
    parser.add_argument('--low_data', type=int, default=1)

    parser.add_argument('--low_data_start', type=int, default=0)
    
    parser.add_argument('--low_data_num', type=int, default=288)
    
    parser.add_argument('--seed', type=int, default=10)

    parser.add_argument('--use_prefix', type=str, default='')

    parser.add_argument('--prefix_length', type=int, default=60)
    
    parser.add_argument('--unq_prefix_length', type=int, default=0)
    
    parser.add_argument('--share_prefix_length', type=int, default=30)
    
    parser.add_argument('--target_prefix_length', type=int, default=30)
    
    parser.add_argument('--only_target', type=int, default=1)
    
    parser.add_argument('--use_pretrained_seed', type=int, default=1)    
    
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
    parser.add_argument('--max_summary', type=int, default=100)

    parser.add_argument('--max_article', type=int, default=250)    
    
    #Model Setting
    parser.add_argument('--hidden_dim', type=int, default=1024)

    parser.add_argument('--emb_dim', type=int, default=1024)
    
    parser.add_argument('--vocab_size', type=int, default=50264)      

    parser.add_argument('--lr', type=float, default=5e-5)     
    
    parser.add_argument('--eps', type=float, default=1e-10)
    
    parser.add_argument('--prefix_dropout', type=float, default=0)    
        
    parser.add_argument('--batch_size', type=int, default=4)  

    parser.add_argument('--true_batch_size', type=int, default=48)  

    parser.add_argument('--buffer_size', type=int, default=144)      
    
    parser.add_argument('--scale_embedding', type=int, default=0)  
    
    #lr setting

    parser.add_argument('--use_lr_decay', type=int, default=0)  
    
    parser.add_argument('--lr_decay_step', type=int, default=10000)  
    
    parser.add_argument('--lr_decay', type=float, default=1)  

    # Testing setting
    parser.add_argument('--beam_size', type=int, default=2)
    
    parser.add_argument('--max_dec_steps', type=int, default=75)
    
    parser.add_argument('--min_dec_steps', type=int, default=30)
    
    parser.add_argument('--test_model', type=str, default='')   
    
    parser.add_argument('--test_description', type=str, default='')   
    
    parser.add_argument('--load_model', type=str, default='')  
    
    parser.add_argument('--save_path', type=str, default='')  
    
    parser.add_argument('--mid_start', type=int, default=0)    

   
    # Checkpoint Setting
    parser.add_argument('--max_epoch', type=int, default=60)
    
    parser.add_argument('--train_set_len', type=int, default=6)
    
    parser.add_argument('--savefreq', type=int, default=24)

    parser.add_argument('--checkfreq', type=int, default=1)    

    parser.add_argument('--startfreq', type=int, default=48)        
    
    args = parser.parse_args()
    
    return args





def main():

    args = argLoader()

    torch.cuda.set_device(args.device)


    test_model_list=[]
    
    for model in test_model_list:
        print('test model', model)
        print('CUDA', torch.cuda.current_device())
        
        
        args.test_model=model
        x = Test(args)
        x.test()
        
        print('finish testing model')
        x=0
        torch.cuda.empty_cache()  
        

main()
        