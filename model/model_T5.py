import os
import sys
import math
import torch
import torch.nn as nn
from transformers_modify.modeling_bart import BartForConditionalGeneration
from transformers import T5Tokenizer,BartTokenizer
from soft_emb import *
import torch.nn.functional as F
import copy
import transformers


class generator_prefix(nn.Module):

    def __init__(self, config):

        super(generator_prefix,self).__init__()  
        
        self.config=config
        
        seed = self.config.seed
        torch.manual_seed(seed)           
        torch.cuda.manual_seed(seed)       
        torch.cuda.manual_seed_all(seed)           

        self.model = BartForConditionalGeneration.from_pretrained(self.config.pretrained_model, set_preseqlen=self.config.prefix_length)   
        self.tokenizer = BartTokenizer.from_pretrained(self.config.pretrained_model)
        self.emb_layer = self.model.model.encoder.embed_tokens
        
            
            
        if config.use_prefix == 'activation_prefix_embedding_original_simple_all':
            self.prefix_layer = activation_prefix_embedding_original_simple_all(self.emb_layer,n_tokens=self.config.prefix_length, uni_tokens=self.config.unq_prefix_length, share_tokens=self.config.share_prefix_length, target_tokens=self.config.target_prefix_length, decoder_layers=12,decoder_attention_heads=16,prefix_dropout=self.config.prefix_dropout, only_target=self.config.only_target)
        if config.use_prefix == 'activation_prefix_embedding_original_simple_all_divide':
            self.prefix_layer = activation_prefix_embedding_original_simple_all_divide(self.emb_layer,n_tokens=self.config.prefix_length, uni_tokens=self.config.unq_prefix_length, share_tokens=self.config.share_prefix_length, target_tokens=self.config.target_prefix_length, decoder_layers=12,decoder_attention_heads=16,prefix_dropout=self.config.prefix_dropout, only_target=self.config.only_target)
        if config.use_prefix == 'activation_prefix_embedding_original_simple_lowdata_only':
            self.prefix_layer = activation_prefix_embedding_original_simple_lowdata_only(self.emb_layer,n_tokens=self.config.prefix_length, uni_tokens=self.config.unq_prefix_length, share_tokens=self.config.share_prefix_length, target_tokens=self.config.target_prefix_length, decoder_layers=12,decoder_attention_heads=16,prefix_dropout=self.config.prefix_dropout, only_target=self.config.only_target)
        if config.use_prefix == 'activation_prefix_embedding_original_simple_all_full_data':
            self.prefix_layer = activation_prefix_embedding_original_simple_all_full_data(self.emb_layer,n_tokens=self.config.prefix_length, uni_tokens=self.config.unq_prefix_length, share_tokens=self.config.share_prefix_length, target_tokens=self.config.target_prefix_length, decoder_layers=12,decoder_attention_heads=16,prefix_dropout=self.config.prefix_dropout)
        if config.use_prefix == 'activation_prefix_embedding_original_simple_all_divide_full_data':
           self.prefix_layer = activation_prefix_embedding_original_simple_all_divide_full_data(self.emb_layer,n_tokens=self.config.prefix_length, uni_tokens=self.config.unq_prefix_length, share_tokens=self.config.share_prefix_length, target_tokens=self.config.target_prefix_length, decoder_layers=12,decoder_attention_heads=16,prefix_dropout=self.config.prefix_dropout, only_target=self.config.only_target)
        if config.use_prefix == 'activation_prefix_embedding_original_simple_all_full_data_fisher':
            self.prefix_layer = activation_prefix_embedding_original_simple_all_full_data_fisher(self.emb_layer,n_tokens=self.config.prefix_length, uni_tokens=self.config.unq_prefix_length, share_tokens=self.config.share_prefix_length, target_tokens=self.config.target_prefix_length, decoder_layers=12,decoder_attention_heads=16,prefix_dropout=self.config.prefix_dropout)
        if config.use_prefix == 'activation_prefix_embedding_original_simple_all_fisher':
            self.prefix_layer = activation_prefix_embedding_original_simple_all_fisher(self.emb_layer,n_tokens=self.config.prefix_length, uni_tokens=self.config.unq_prefix_length, share_tokens=self.config.share_prefix_length, target_tokens=self.config.target_prefix_length, decoder_layers=12,decoder_attention_heads=16,prefix_dropout=self.config.prefix_dropout, only_target=self.config.only_target)




    def forward(self, input_ids, input_mask, summary_ids, summary_mask, label1, label2):
        
        #input_id: (batch_size, sequence_length)
        
        past_key_values = self.prefix_layer(input_ids, label1, label2)         
        outputs = self.model(input_ids=input_ids,attention_mask=input_mask,past_key_values=past_key_values, decoder_input_ids = summary_ids, use_cache=False, use_prefix=True)
        
        return outputs 

    
    def inference(self, input_ids, input_mask, label1, label2, use_beam = 0, return_attention = 0):
        
        #input_id: (batch_size, sequence_length)
        if use_beam == 0:
            past_key_values = self.prefix_layer(input_ids, label1, label2)         
        else:
            past_key_values = self.prefix_layer(input_ids, label1, label2, sample_size = 2)          
        #print(past_key_values[0]['encoder_decoder']['prev_key_padding_mask'])
        if use_beam == 0:
            #outputs = self.generate_without_beam_search(inputs_embeds, input_mask, self.config.max_dec_steps)
            
            outputs,attention = self.model.generate(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=input_mask,
            use_cache=True,
            length_penalty=1.0,
            use_prefix=True,
            decoder_start_token_id=self.config.bos_token_id,
            num_beams=1,
            min_length=self.config.min_dec_steps,
            max_length=self.config.max_dec_steps,
            return_attention=1
            )
            outputs=outputs[0]
            
        else:
            outputs = self.model.generate(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=input_mask,
            use_cache=True,
            length_penalty=1.0,
            use_prefix=True,
            decoder_start_token_id=self.config.bos_token_id,
            num_beams=2,
            min_length=self.config.min_dec_steps,
            max_length=self.config.max_dec_steps,
            )
            outputs=outputs[0]            
        if return_attention==0:
            return outputs
        else:
            return outputs,attention    
 



class generator(nn.Module):

    def __init__(self, config):

        super(generator,self).__init__()  
        self.config=config      
        if 't5' in self.config.pretrained_model:
            self.model = transformers.T5ForConditionalGeneration.from_pretrained(self.config.pretrained_model)
        if 'bart' in self.config.pretrained_model:
            self.model = transformers.BartForConditionalGeneration.from_pretrained(self.config.pretrained_model)   


        

    def forward(self, input_ids, input_mask, summary_ids, summary_mask, label1, label2):
        
        outputs = self.model(input_ids=input_ids,attention_mask=input_mask, decoder_input_ids = summary_ids)
        
        return outputs 

    
    def inference(self, input_ids, input_mask, label1, label2, use_beam = 1):

        
        if use_beam == 0:
            outputs = self.model.generate(
            input_ids,
            attention_mask=input_mask,
            use_cache=True,
            length_penalty=1.0,
            decoder_start_token_id=self.config.bos_token_id,
            num_beams=1,
            min_length=self.config.min_dec_steps,
            max_length=self.config.max_dec_steps,
            )
            outputs=outputs[0]  
        else:
            outputs = self.model.generate(
            input_ids,
            attention_mask=input_mask,
            use_cache=True,
            length_penalty=1.0,
            decoder_start_token_id=self.config.bos_token_id,
            num_beams=2,
            min_length=self.config.min_dec_steps,
            max_length=self.config.max_dec_steps,
            )
            outputs=outputs[0]            
        return outputs
    
    

