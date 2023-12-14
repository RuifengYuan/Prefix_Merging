import glob
import random
import struct
import json
import re
import torch
#from tensorflow.core.example import example_pb2
import csv
import argparse
from transformers import BartTokenizer
class data_loader():

    def __init__(self, part, config, tokenizer, load_xsum=1, load_debate=1, load_squad=1, load_kptimes=1, mix=1, low_data=0):

        super(data_loader,self).__init__()  
      
        self.part=part
        self.tokenizer=tokenizer
        self.config=config
        
        random.seed(self.config.seed)
        
        self.count=0
        self.epoch=0

        self.max_epoch=config.max_epoch
        self.buffer_size=config.buffer_size
        self.batch_size=config.batch_size
        self.true_batch_size=config.true_batch_size
        self.max_article=config.max_article
        self.max_summary=config.max_summary

        '''
        self.max_epoch=1
        self.buffer_size=128
        self.batch_size=4
        self.max_article=512
        self.max_summary=100
        '''
        self.mix=mix
        self.load_xsum=load_xsum
        self.load_debate=load_debate
        self.load_squad=load_squad
        self.load_kptimes=load_kptimes
        self.low_data=low_data
        '''
        if load_debate == 1:
            if self.part == 'train':
                self.debate_content_path = 'data/debate/train_content'
                self.debate_query_path = 'data/debate/train_query'
                self.debate_summary_path = 'data/debate/train_summary'
            if self.part == 'val':
                self.debate_content_path = 'data/debate/test_content'
                self.debate_query_path = 'data/debate/test_query'
                self.debate_summary_path = 'data/debate/test_summary'        
            if self.part == 'test':
                self.debate_content_path = 'data/debate/test_content'
                self.debate_query_path = 'data/debate/test_query'
                self.debate_summary_path = 'data/debate/test_summary'  
        '''

        if load_debate == 1:
            
            if self.part == 'train':
                self.pub_file_path = 'data/pubmed/pubmedqa_train.txt'
            if self.part == 'val':
                self.pub_file_path = 'data/pubmed/pubmedqa_val.txt'        
            if self.part == 'test':
                self.pub_file_path = 'data/pubmed/pubmedqa_test.txt'




        if load_xsum == 1:
            with open("data/xsum/index.json",'r') as index_file:
                split_index = json.load(index_file)
            
            if self.part == 'train':
                self.xsum_file_list = split_index['train']
            if self.part == 'val':
                self.xsum_file_list = split_index['validation']     
            if self.part == 'test':
                self.xsum_file_list = split_index['test']
                
        if load_squad == 1:

            if self.part == 'train':
                self.squad_file_path = 'data/squad/train-v2.0.json'
            if self.part == 'val':
                self.squad_file_path = 'data/squad/dev-v2.0.json'   
            if self.part == 'test':
                self.squad_file_path = 'data/squad/dev-v2.0.json'
        
        
        if load_kptimes == 1:

            if self.part == 'train':
                self.kptimes_file_path = 'data/duc/duc2006.txt'
            if self.part == 'val':
                self.kptimes_file_path = 'data/duc/duc2007.txt'   
            if self.part == 'test':
                self.kptimes_file_path = 'data/duc/duc2007.txt'

        if load_debate == 1:
            self.data_generator_debate=self.next_data_debate()
        if load_xsum == 1:
            self.data_generator_xsum=self.next_data_xsum()
        if load_squad == 1:
            self.data_generator_squad=self.next_data_squad()
        if load_kptimes == 1:
            self.data_generator_kptimes=self.next_data_kptimes()
        
        self.batch_generator=self.next_batch()
        

    def next_data_debate(self):
        buffer=[]
        for epoch in range(self.max_epoch):
            self.epoch=self.epoch+1           
            with open(self.pub_file_path, 'r', encoding='utf-8') as f:
                all_data = f.readlines()

            if self.low_data == 1:
                all_data=all_data[self.config.low_data_start:self.config.low_data_start+self.config.low_data_num]

            if self.part == 'train':
                random.shuffle(all_data)
            else:
                pass
            
            for idx in range(len(all_data)):
                
                one=all_data[idx].split('\t')
                
                article_text_list = eval(one[1].lower())
                article_text=''
                for i in article_text_list:
                    article_text=article_text+i+' '
                article_text=article_text.strip().lower()
                
                abstract_text  = one[2].lower().strip('\n')
                
                query = one[0].lower()
                
                answer = 0                    


                src= self.end_replace(article_text)
                ref= self.end_replace(abstract_text)
                src=self.clean(src)
                ref=self.clean(ref)
                src=src.replace('[','(')
                src=src.replace(']',')')
                ref=ref.replace('[','(')
                ref=ref.replace(']',')')
                article=src.split('.')
                ref=ref.replace('<s>','')
                summary=ref.split('</s>')
                
                summary_f=[]
                for i in summary:
                    if len(i) > 10:
                        summary_f.append(i.strip())
                article_f=[]
                for i in article:
                    if len(i) > 25:
                        article_f.append(i.strip())
                if article_f==[] or summary_f==[]:
                    pass
                else:
                    buffer.append((article_f,summary_f,query))
                if len(buffer) == self.buffer_size:
                    yield buffer
                    buffer=[]
        print ("data_generator completed reading all datafiles for all epoches. No more data.")                       
        return 0        

    
    '''
    def next_data_debate(self):
        buffer=[]
        for epoch in range(self.max_epoch):
            self.epoch=self.epoch+1
            
            with open(self.debate_content_path, 'r', encoding='utf-8') as f:
                all_data_content = f.readlines()
            with open(self.debate_query_path, 'r', encoding='utf-8') as f:
                all_data_query = f.readlines()
            with open(self.debate_summary_path, 'r', encoding='utf-8') as f:
                all_data_summary = f.readlines()
            
            idx_list=list(range(len(all_data_content)))

            if self.low_data == 1:
                idx_list=idx_list[self.config.low_data_start:self.config.low_data_start+self.config.low_data_num]

            
            if self.part == 'train':
                random.shuffle(idx_list)
            else:
                pass
            
            for idx in idx_list:
                
                article_text = all_data_content[idx].lower().strip('\n').strip(' <eos>').strip('<s> ')
                
                abstract_text = all_data_summary[idx].lower().strip('\n').strip(' <eos>').strip('<s> ')
                
                query = all_data_query[idx].lower().strip('\n').strip(' <eos>').strip('<s> ')
                
                if ' .' not in abstract_text:
                    abstract_text+=' .'
                
              
                src= self.end_replace(article_text)
                ref= self.end_replace(abstract_text)
                src=self.clean(src)
                ref=self.clean(ref)
                article=src.split('. ')
                ref=ref.replace('<s>','')
                summary=ref.split('</s>')
                
                summary_f=[]
                for i in summary:
                    if len(i) > 10:
                        summary_f.append(i.strip())
                article_f=[]
                for i in article:
                    if len(i) > 10:
                        article_f.append(i.strip())
                if article_f==[] or summary_f==[]:
                    pass
                else:
                    buffer.append((article_f,summary_f,query))
                if len(buffer) == self.buffer_size:
                    yield buffer
                    buffer=[]
        print ("data_generator completed reading all datafiles for all epoches. No more data.")                       
        return 0
    '''
    
    def next_data_xsum(self):
        buffer=[]
        for epoch in range(self.max_epoch):
            #self.epoch=self.epoch+1
            filelist = self.xsum_file_list
            if self.part == 'train':
                random.shuffle(filelist)
            else:
                pass
            for f in filelist:
                one_data_path='data/xsum/data/'+f+'.data'
                
                try:
                    with open(one_data_path, encoding='utf-8') as f:
                        one_data=f.readlines()
                except:
                    continue
                
                
                split_pos=one_data.index("\[XSUM\]RESTBODY\[XSUM\]\n")    
              
                summary_list=one_data[:split_pos]
                article_list=one_data[split_pos:]              
              
                split_pos2=summary_list.index("\[XSUM\]FIRST-SENTENCE\[XSUM\]\n")     
  
                summary_list=summary_list[split_pos2:]

                article_text=''
                for i in article_list:
                    if i != '\n' and i != '\[XSUM\]RESTBODY\[XSUM\]\n' and i !='\[XSUM\]FIRST-SENTENCE\[XSUM\]\n':
                        article_text=article_text+i.strip('\n')+' '
                article_text=article_text.strip().lower()

                abstract_text=''
                for i in summary_list:
                    if i != '\n' and i != '\[XSUM\]RESTBODY\[XSUM\]\n' and i !='\[XSUM\]FIRST-SENTENCE\[XSUM\]\n':
                        abstract_text=abstract_text+i.strip('\n')+' '
                abstract_text=abstract_text.strip().lower()   

                src= self.end_replace(article_text)
                ref= self.end_replace(abstract_text)
                src=self.clean(src)
                ref=self.clean(ref)
                src=src.replace('[','(')
                src=src.replace(']',')')
                ref=ref.replace('[','(')
                ref=ref.replace(']',')')
                article=src.split('. ')
                ref=ref.replace('<s>','')
                summary=[ref]
                summary_f=[]
                for i in summary:
                    if len(i) > 10:
                        summary_f.append(i.strip())
                article_f=[]
                for i in article:
                    if len(i) > 10:
                        article_f.append(i.strip())
                        
                if self.part == 'train':
                    if len(article_f)<=3 or summary_f==[]:
                        pass
                    else:
                        buffer.append((article_f,summary_f))
                else:
                    if len(article_f)==[] or summary_f==[]:
                        pass
                    else:
                        buffer.append((article_f,summary_f))     
                        
                if len(buffer) == self.buffer_size:
                    yield buffer
                    buffer=[]
                    
        print ("data_generator completed reading all datafiles for all epoches. No more data.")                       
        return 0
                
    
    def next_data_squad(self):
        buffer=[]
        for epoch in range(self.max_epoch):
            #self.epoch=self.epoch+1
            with open(self.squad_file_path, 'r', encoding='utf-8') as f:
                all_data=json.load(f)
            
            data_list=all_data['data']
            if self.part == 'train':
                random.shuffle(data_list)
            else:
                pass            
            
            for one_data in data_list:
                for one_paragraph in one_data['paragraphs']:
                    
                    context=one_paragraph['context']
                    qa_list=one_paragraph['qas']
                    
                    for one_qa in qa_list:
                        
                        if one_qa['is_impossible'] == True:
                            continue
                        
                        article_text=context.lower()
        
                        abstract_text=one_qa['answers'][0]['text'].lower()
                        
                        query=one_qa['question'].lower()

                        src=self.clean(article_text)

                        summary_f=[abstract_text]

                        article_f=src.split('. ')
    
                                
                        if self.part == 'train':
                            if len(article_f)<=2 or summary_f==[]:
                                pass
                            else:
                                buffer.append((article_f,summary_f, query))
                        else:
                            if len(article_f)==[] or summary_f==[]:
                                pass
                            else:
                                buffer.append((article_f, summary_f, query))     
                                
                        if len(buffer) == self.buffer_size:
                            yield buffer
                            buffer=[]
                    
        print ("data_generator completed reading all datafiles for all epoches. No more data.")                       
        return 0



    def next_data_kptimes(self):
        buffer=[]
        for epoch in range(self.max_epoch):
            #self.epoch=self.epoch+1
            with open(self.kptimes_file_path, 'r', encoding='utf-8') as f:
                all_data=f.readlines()

            if self.part == 'train':
                random.shuffle(all_data)
            else:
                pass    

            for one_data in all_data:
                dict_data=eval(one_data)
                query=dict_data['query'].lower()
                summary=dict_data['summary'].lower()
                source=dict_data['ext_sent']
                source=[s.lower() for s in source]
                article_f=source
                
                summary_f=summary.split('. ')

                        
                if self.part == 'train':
                    if len(article_f)<=2 or summary_f==[]:
                        pass
                    else:
                        buffer.append((article_f,summary_f, query))
                else:
                    if len(article_f)==[] or summary_f==[]:
                        pass
                    else:
                        buffer.append((article_f, summary_f, query))     
                        
                if len(buffer) == self.buffer_size:
                    yield buffer
                    buffer=[]
                    
        print ("data_generator completed reading all datafiles for all epoches. No more data.")                       
        return 0

                    
 

    def next_batch(self):
        while(True):
            
            if self.load_debate == 1:
                data_debate = self.data_generator_debate.__next__()
                process_data_debate=[]
                for i in data_debate:
                    example=[]
                    article=i[0]
                    summary=i[1]
                    query=i[2]                    
                
                    example.append(self.tokenizer.encode('summarization and answer the question '+query+' '+ ' . '.join(article))[0:self.max_article])
                    example.append(self.tokenizer.encode(' . '.join(summary))[0:self.max_summary])   
                    example.append(' . '.join(summary))  
                    process_data_debate.append(example)

                process_data_debate.sort(key=self.get_sort)
                
            if self.load_xsum == 1:
                data_xsum = self.data_generator_xsum.__next__()  
                process_data_xsum=[]
                for i in data_xsum:
                    example=[]
                    article=i[0]
                    summary=i[1]

                    example.append(self.tokenizer.encode('summarization '+' . '.join(article))[0:self.max_article])
                    example.append(self.tokenizer.encode(' . '.join(summary))[0:self.max_summary])   
                    example.append(' . '.join(summary))  
                    process_data_xsum.append(example)

                process_data_xsum.sort(key=self.get_sort)
                
            if self.load_squad == 1:
                data_squad = self.data_generator_squad.__next__() 
                process_data_squad=[]
                for i in data_squad:
                    example=[]
                    article=i[0]
                    summary=i[1]
                    query=i[2]


                    example.append(self.tokenizer.encode('answer the question '+query.replace('?',' ?')+' '+' . '.join(article))[0:self.max_article])
                    example.append(self.tokenizer.encode(self.answer_query_merge(' '.join(summary), query.replace('?',''))+' .')[0:self.max_summary])   
                    example.append(' . '.join(summary))  

                    process_data_squad.append(example)
                process_data_squad.sort(key=self.get_sort)
                
            if self.load_kptimes == 1:
                data_kptimes = self.data_generator_kptimes.__next__()  
                process_data_kptimes=[]
                for i in data_kptimes:
                    example=[]
                    article=i[0]
                    summary=i[1]
                    query=i[2]                    
                    example.append(self.tokenizer.encode('summarization and answer the question '+query+'?'+' '+ ' '.join(article))[0:self.max_article])
                    example.append(self.tokenizer.encode(' . '.join(summary))[0:self.max_summary])   
                    example.append(' . '.join(summary))  
                    process_data_kptimes.append(example)
                    
                process_data_kptimes.sort(key=self.get_sort)

            
            
            if self.mix == 1:
                iter_times = int(self.buffer_size/self.batch_size) 
                for i in range(iter_times):
                    task_list=[]
                    if self.load_debate == 1:
                        task_list.append('debate')
                    if self.load_xsum == 1:
                        task_list.append('xsum')
                    if self.load_squad == 1:
                        task_list.append('squad')
                    if self.load_kptimes == 1:
                        task_list.append('kptimes')
    
                    random.shuffle(task_list)
                    
                    for task in task_list:
                        if task == 'debate':
                            process_data=process_data_debate
                            label1=0
                            label2=0
                        if task == 'xsum':
                            process_data=process_data_xsum
                            label1=0
                            label2=1
                        if task == 'squad':
                            process_data=process_data_squad
                            label1=1
                            label2=0
                        if task == 'kptimes':
                            process_data=process_data_kptimes
                            label1=1
                            label2=1                        
                            
                        process_data_chunked=process_data[i*self.batch_size:(i+1)*self.batch_size]

                        
                        article_id=[]
                        summary_id=[]
                        summary=[]
                        for one_example in process_data_chunked:
                            article_id.append(one_example[0])
                            summary_id.append(one_example[1])            
                            summary.append(one_example[2]) 
                            
                        article_id,article_id_mask=self.pad_with_mask(article_id, pad_id=self.config.pad_token_id)
                        article_id=torch.tensor(article_id)
                        article_id_mask=torch.tensor(article_id_mask)
                        
                        summary_id,summary_id_mask=self.pad_with_mask(summary_id, pad_id=self.config.pad_token_id)
                        summary_id=torch.tensor(summary_id)                    
                        summary_id_mask=torch.tensor(summary_id_mask)
                        
                        article_id=article_id.cuda()
                        article_id_mask=article_id_mask.cuda()
                        summary_id=summary_id.cuda()
                        summary_id_mask=summary_id_mask.cuda()
                            
                        yield [article_id,article_id_mask,summary_id,summary_id_mask,summary,label1,label2]
                        
            if self.mix == 0:
                iter_times = int(self.buffer_size/self.true_batch_size)  #128/32=4
                for i in range(iter_times):
                    task_list=[]
                    if self.load_debate == 1:
                        task_list.append('debate')
                    if self.load_xsum == 1:
                        task_list.append('xsum')
                    if self.load_squad == 1:
                        task_list.append('squad')
                    if self.load_kptimes == 1:
                        task_list.append('kptimes')
    
                    #random.shuffle(task_list)
                    
                    for task in task_list:
                        if task == 'debate':
                            process_data=process_data_debate
                            label1=0
                            label2=0
                        if task == 'xsum':
                            process_data=process_data_xsum
                            label1=0
                            label2=1
                        if task == 'squad':
                            process_data=process_data_squad
                            label1=1
                            label2=0
                        if task == 'kptimes':
                            process_data=process_data_kptimes
                            label1=1
                            label2=1     
                            
                        for j in range(int(self.true_batch_size/self.batch_size)):  #32/4=8
                            
                            process_data_chunked=process_data[i*self.true_batch_size+j*self.batch_size:i*self.true_batch_size+(j+1)*self.batch_size]
                            #print(label1,label2,i*self.true_batch_size+j*self.batch_size,i*self.true_batch_size+(j+1)*self.batch_size)                            
                            article_id=[]
                            summary_id=[]
                            summary=[]
                            for one_example in process_data_chunked:
                                article_id.append(one_example[0])
                                summary_id.append(one_example[1])            
                                summary.append(one_example[2]) 
                                
                            article_id,article_id_mask=self.pad_with_mask(article_id, pad_id=self.config.pad_token_id)
                            article_id=torch.tensor(article_id)
                            article_id_mask=torch.tensor(article_id_mask)
                            
                            summary_id,summary_id_mask=self.pad_with_mask(summary_id, pad_id=self.config.pad_token_id)
                            summary_id=torch.tensor(summary_id)                    
                            summary_id_mask=torch.tensor(summary_id_mask)
                            
                            article_id=article_id.cuda()
                            article_id_mask=article_id_mask.cuda()
                            summary_id=summary_id.cuda()
                            summary_id_mask=summary_id_mask.cuda()
                                
                            yield [article_id,article_id_mask,summary_id,summary_id_mask,summary,label1,label2]
                        
    
    def load_data(self):
        self.count=self.count+1
        return self.batch_generator.__next__()
                    
            
    def get_sort(self, x):
        return len(x[0])


    def pad_with_mask(self, data, pad_id=0, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
            
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        
        pad_mask = [[1] * len(d) + [0] * (width - len(d)) for d in data]
        return rtn_data,pad_mask 


    def clean(self,x):
        REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
             "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"', "\n": ''}
        return re.sub(r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''", lambda m: REMAP.get(m.group()), x)
    
    
    def end_replace(self,s):
        forbidden=['!', '?']
        for i in forbidden:
            s=s.replace(i,'.')
        return s                
        
    def answer_query_merge(self,answer,query):
        query_word=['which', 'what', 'when', 'where','who','whom','whose','why','how']
        query_pharse=['what time', 'what color', 'how many', 'how long','how much','how old','how far', 'when did', 'when will', 'when do', 
                      'which did', 'which will', 'which do', 'what did', 'what will', 'what do','where did', 'where will', 'where do',
                      'who did', 'who will', 'who do','why did', 'why will', 'why do','how did', 'how will', 'how do',]
        
        replace_target=query_pharse+query_word
        
        for i in replace_target:
            if i in query:
                return query.replace(i,answer)
        
        return query+' is '+answer
        
        
