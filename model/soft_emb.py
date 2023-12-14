import torch
import torch.nn as nn




class activation_prefix_embedding_original_simple_all(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 20, 
                uni_tokens: int = 20,
                share_tokens: int = 20,
                target_tokens: int = 0,
                decoder_layers: int = 12,
                decoder_attention_heads: int = 16,                
                random_range: float = 0.5,
                initialize_from_vocab: bool = True,
                prefix_dropout: float = 0.0,
                only_target: int = 0):

        super(activation_prefix_embedding_original_simple_all, self).__init__()
        
        self.dropout = nn.Dropout(prefix_dropout)
        self.only_target = only_target

        self.match_n_layer = decoder_layers
        self.match_n_head =  decoder_attention_heads
        self.voc_size,self.dim_size=wte.weight.size()
        self.match_n_embd = self.dim_size // self.match_n_head
        
        self.wte = wte
        self.n_tokens = n_tokens         
        self.uni_tokens = uni_tokens 
        self.share_tokens = share_tokens 
        self.target_tokens = target_tokens
        assert self.n_tokens == self.uni_tokens+self.share_tokens+self.target_tokens, 'the total prefix length is not equal to share + unique length'
        
        self.wte_prefix_self = nn.Embedding(self.share_tokens+3*self.uni_tokens+self.target_tokens, self.dim_size)
        self.wte_prefix_cros = nn.Embedding(self.share_tokens+3*self.uni_tokens+self.target_tokens, self.dim_size)
        self.wte_prefix_encd = nn.Embedding(self.share_tokens+3*self.uni_tokens+self.target_tokens, self.dim_size)
        
        self.seed_tokens_debate = torch.cat([torch.arange(0,self.share_tokens).long(), torch.arange(self.share_tokens,self.share_tokens+self.target_tokens).long()])      
        self.seed_tokens_squad  = torch.cat([torch.arange(0,self.share_tokens).long(), torch.arange(self.share_tokens,self.share_tokens+self.target_tokens).long()])   
        self.seed_tokens_xsum   = torch.cat([torch.arange(0,self.share_tokens).long(), torch.arange(self.share_tokens,self.share_tokens+self.target_tokens).long()])  

        
        self.control_trans_self = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))
        self.control_trans_cros = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))
        self.control_trans_encd = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))                                                  
                                                    

            
    def forward(self, tokens, prefix_1, prefix_2, sample_size=1):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        #self.only_target=0
        
        input_embedding = self.wte(tokens)
        bsz=input_embedding.size(0)
        
        old_bsz = bsz
        bsz = bsz * sample_size
        
        if prefix_1==0 and prefix_2==0: 
            seed_tokens=self.seed_tokens_debate
        if prefix_1==0 and prefix_2==1: 
            seed_tokens=self.seed_tokens_xsum      
        if prefix_1==1 and prefix_2==0: 
            seed_tokens=self.seed_tokens_squad  
        if prefix_1==1 and prefix_2==1: 
            seed_tokens=self.seed_tokens_squad              
            
        seed_tokens_expanded = seed_tokens.unsqueeze(0).expand(bsz, -1).cuda()
        input_tokens_enc = seed_tokens.unsqueeze(0).expand(old_bsz, -1).cuda()     


        temp_control = self.wte_prefix_self(seed_tokens_expanded)
        past_key_values = self.control_trans_self(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        

        temp_control2 = self.wte_prefix_cros(seed_tokens_expanded)
        past_key_values2 = self.control_trans_cros(temp_control2)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values2.shape
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)



        temp_control_enc = self.wte_prefix_encd(input_tokens_enc)
        past_key_values_enc = self.control_trans_encd(temp_control_enc)  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)            
        
        assert self.n_tokens == seqlen, 'the total prefix length is not equal to mask length'        
        
        if prefix_1==0 and prefix_2==0 and self.only_target==0: 
            
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.zeros(bsz, seqlen).cuda().bool() #bsz, preseqlen
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.zeros(bsz, seqlen).cuda().bool() #bsz, preseqlen
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).cuda().bool() #bsz, preseqlen
                                        }
                result.append(temp_dict)
                

        if prefix_1==0 and prefix_2==0 and self.only_target==1: 
            
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.cat([torch.full([bsz, self.share_tokens+self.uni_tokens],1).bool().cuda(),torch.full([bsz, self.target_tokens],0).bool().cuda()], 1)
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.cat([torch.full([bsz, self.share_tokens+self.uni_tokens],1).bool().cuda(),torch.full([bsz, self.target_tokens],0).bool().cuda()], 1)
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.cat([torch.full([bsz_enc, self.share_tokens+self.uni_tokens],1).bool().cuda(),torch.full([bsz_enc, self.target_tokens],0).bool().cuda()], 1)
                                        }
                result.append(temp_dict)
   
                
                
        if (prefix_1==1 and prefix_2==0) or (prefix_1==0 and prefix_2==1) or (prefix_1==1 and prefix_2==1): 
        
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens+self.uni_tokens).cuda().bool(),torch.full([bsz, self.target_tokens],1).bool().cuda()], 1)
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens+self.uni_tokens).cuda().bool(),torch.full([bsz, self.target_tokens],1).bool().cuda()], 1)
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.cat([torch.zeros(bsz_enc, self.share_tokens+self.uni_tokens).cuda().bool(),torch.full([bsz_enc, self.target_tokens],1).bool().cuda()], 1)
                                        }
                result.append(temp_dict)
    
                             
        return result 
    


class activation_prefix_embedding_original_simple_all_fisher(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 20, 
                uni_tokens: int = 20,
                share_tokens: int = 20,
                target_tokens: int = 0,
                decoder_layers: int = 12,
                decoder_attention_heads: int = 16,                
                random_range: float = 0.5,
                initialize_from_vocab: bool = True,
                prefix_dropout: float = 0.0,
                only_target: int = 0):

        super(activation_prefix_embedding_original_simple_all_fisher, self).__init__()
        
        self.dropout = nn.Dropout(prefix_dropout)
        self.only_target = only_target

        self.match_n_layer = decoder_layers
        self.match_n_head =  decoder_attention_heads
        self.voc_size,self.dim_size=wte.weight.size()
        self.match_n_embd = self.dim_size // self.match_n_head
        
        self.wte = wte
        self.n_tokens = n_tokens         
        self.uni_tokens = uni_tokens 
        self.share_tokens = share_tokens 
        self.target_tokens = target_tokens
        assert self.n_tokens == self.uni_tokens+self.share_tokens+self.target_tokens, 'the total prefix length is not equal to share + unique length'
        
        self.wte_prefix_self = nn.Embedding(self.share_tokens+3*self.uni_tokens+self.target_tokens, self.dim_size)
        self.wte_prefix_cros = nn.Embedding(self.share_tokens+3*self.uni_tokens+self.target_tokens, self.dim_size)
        self.wte_prefix_encd = nn.Embedding(self.share_tokens+3*self.uni_tokens+self.target_tokens, self.dim_size)
        
        self.seed_tokens_debate = torch.cat([torch.arange(0,self.share_tokens).long(), torch.arange(self.share_tokens,self.share_tokens+self.target_tokens).long()])      
        self.seed_tokens_squad  = torch.cat([torch.arange(0,self.share_tokens).long(), torch.arange(self.share_tokens,self.share_tokens+self.target_tokens).long()])   
        self.seed_tokens_xsum   = torch.cat([torch.arange(0,self.share_tokens).long(), torch.arange(self.share_tokens,self.share_tokens+self.target_tokens).long()])  

        
        self.control_trans_self = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))
        self.control_trans_cros = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))
        self.control_trans_encd = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))    
                                              

        self.mask_self=[1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        self.mask_cros=[1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.mask_encd=[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0]        
            
    def forward(self, tokens, prefix_1, prefix_2, sample_size=1):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        #self.only_target=0
        
        input_embedding = self.wte(tokens)
        bsz=input_embedding.size(0)
        
        old_bsz = bsz
        bsz = bsz * sample_size
        
        if prefix_1==0 and prefix_2==0: 
            seed_tokens=self.seed_tokens_debate
        if prefix_1==0 and prefix_2==1: 
            seed_tokens=self.seed_tokens_xsum      
        if prefix_1==1 and prefix_2==0: 
            seed_tokens=self.seed_tokens_squad  
        if prefix_1==1 and prefix_2==1: 
            seed_tokens=self.seed_tokens_squad              
            
        seed_tokens_expanded = seed_tokens.unsqueeze(0).expand(bsz, -1).cuda()
        input_tokens_enc = seed_tokens.unsqueeze(0).expand(old_bsz, -1).cuda()     


        temp_control = self.wte_prefix_self(seed_tokens_expanded)
        past_key_values = self.control_trans_self(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        

        temp_control2 = self.wte_prefix_cros(seed_tokens_expanded)
        past_key_values2 = self.control_trans_cros(temp_control2)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values2.shape
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)



        temp_control_enc = self.wte_prefix_encd(input_tokens_enc)
        past_key_values_enc = self.control_trans_encd(temp_control_enc)  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)            
        
        assert self.n_tokens == seqlen, 'the total prefix length is not equal to mask length'        
        
        if prefix_1==0 and prefix_2==0 and self.only_target==0: 
            
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.zeros(bsz, seqlen).cuda().bool() #bsz, preseqlen
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.zeros(bsz, seqlen).cuda().bool() #bsz, preseqlen
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).cuda().bool() #bsz, preseqlen
                                        }
                result.append(temp_dict)
                

        if prefix_1==0 and prefix_2==0 and self.only_target==1: 
            
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.cat([torch.full([bsz, self.share_tokens+self.uni_tokens],1).bool().cuda(),torch.tensor(self.mask_self).expand(bsz, 40).bool().cuda()], 1)
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.cat([torch.full([bsz, self.share_tokens+self.uni_tokens],1).bool().cuda(),torch.tensor(self.mask_cros).expand(bsz, 40).bool().cuda()], 1)
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.cat([torch.full([bsz_enc, self.share_tokens+self.uni_tokens],1).bool().cuda(),torch.tensor(self.mask_encd).expand(bsz_enc, 40).bool().cuda()], 1)
                                        }
                result.append(temp_dict)
   
                
                
        if (prefix_1==1 and prefix_2==0) or (prefix_1==0 and prefix_2==1) or (prefix_1==1 and prefix_2==1): 
        
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens+self.uni_tokens).cuda().bool(),torch.full([bsz, self.target_tokens],1).bool().cuda()], 1)
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens+self.uni_tokens).cuda().bool(),torch.full([bsz, self.target_tokens],1).bool().cuda()], 1)
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.cat([torch.zeros(bsz_enc, self.share_tokens+self.uni_tokens).cuda().bool(),torch.full([bsz_enc, self.target_tokens],1).bool().cuda()], 1)
                                        }
                result.append(temp_dict)
    
                             
        return result 



class activation_prefix_embedding_original_simple_lowdata_only(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 20, 
                uni_tokens: int = 20,
                share_tokens: int = 20,
                target_tokens: int = 0,
                decoder_layers: int = 12,
                decoder_attention_heads: int = 16,                
                random_range: float = 0.5,
                initialize_from_vocab: bool = True,
                prefix_dropout: float = 0.0,
                only_target: int = 0):

        super(activation_prefix_embedding_original_simple_lowdata_only, self).__init__()
        
        self.dropout = nn.Dropout(prefix_dropout)
        self.only_target = only_target

        self.match_n_layer = decoder_layers
        self.match_n_head =  decoder_attention_heads
        self.voc_size,self.dim_size=wte.weight.size()
        self.match_n_embd = self.dim_size // self.match_n_head
        
        self.wte = wte
        self.n_tokens = n_tokens         
        self.uni_tokens = uni_tokens 
        self.share_tokens = share_tokens 
        self.target_tokens = target_tokens
        assert self.n_tokens == self.uni_tokens+self.share_tokens+self.target_tokens, 'the total prefix length is not equal to share + unique length'
        
        self.wte_prefix_self_pre = nn.Embedding(self.share_tokens, self.dim_size)
        self.wte_prefix_cros_pre = nn.Embedding(self.share_tokens, self.dim_size)
        self.wte_prefix_encd_pre = nn.Embedding(self.share_tokens, self.dim_size)
        
        self.wte_prefix_self = nn.Embedding(self.target_tokens, self.dim_size)
        self.wte_prefix_cros = nn.Embedding(self.target_tokens, self.dim_size)
        self.wte_prefix_encd = nn.Embedding(self.target_tokens, self.dim_size)
        
        self.seed_tokens_debate_pre = torch.arange(0,self.share_tokens).long()      
        self.seed_tokens_debate = torch.arange(0,self.target_tokens).long()     

        
        self.control_trans_self = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))
        self.control_trans_cros = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))
        self.control_trans_encd = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))                                                  
                                                    

            
    def forward(self, tokens, prefix_1, prefix_2, sample_size=1):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens)
        bsz=input_embedding.size(0)
        
        old_bsz = bsz
        bsz = bsz * sample_size
        
        if prefix_1==0 and prefix_2==0: 
            seed_tokens_pre=self.seed_tokens_debate_pre
            seed_tokens=self.seed_tokens_debate

        seed_tokens_expanded_pre = seed_tokens_pre.unsqueeze(0).expand(bsz, -1).cuda()
        seed_tokens_expanded = seed_tokens.unsqueeze(0).expand(bsz, -1).cuda()
        
        input_tokens_enc_pre = seed_tokens_pre.unsqueeze(0).expand(old_bsz, -1).cuda()   
        input_tokens_enc = seed_tokens.unsqueeze(0).expand(old_bsz, -1).cuda()     


        temp_control = torch.cat([self.wte_prefix_self_pre(seed_tokens_expanded_pre), self.wte_prefix_self(seed_tokens_expanded)],1)
        past_key_values = self.control_trans_self(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        

        temp_control2 = torch.cat([self.wte_prefix_cros_pre(seed_tokens_expanded_pre), self.wte_prefix_cros(seed_tokens_expanded)],1)
        past_key_values2 = self.control_trans_cros(temp_control2)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values2.shape
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)



        temp_control_enc = torch.cat([self.wte_prefix_encd_pre(input_tokens_enc_pre), self.wte_prefix_encd(input_tokens_enc)],1)
        past_key_values_enc = self.control_trans_encd(temp_control_enc)  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)            
        
        assert self.n_tokens == seqlen, 'the total prefix length is not equal to mask length'        
        
        if prefix_1==0 and prefix_2==0: 
            
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.zeros(bsz, seqlen).cuda().bool() #bsz, preseqlen
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.zeros(bsz, seqlen).cuda().bool() #bsz, preseqlen
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).cuda().bool() #bsz, preseqlen
                                        }
                result.append(temp_dict)
                             
        return result     
    
    
    
    
class activation_prefix_embedding_original_simple_all_divide(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 20, 
                uni_tokens: int = 20,
                share_tokens: int = 0,
                target_tokens: int = 0,
                decoder_layers: int = 12,
                decoder_attention_heads: int = 16,                
                random_range: float = 0.5,
                initialize_from_vocab: bool = True,
                prefix_dropout: float = 0.0,
                only_target: int = 0):

        super(activation_prefix_embedding_original_simple_all_divide, self).__init__()
        
        self.dropout = nn.Dropout(prefix_dropout)
        
        self.match_n_layer = decoder_layers
        self.match_n_head =  decoder_attention_heads
        self.voc_size,self.dim_size=wte.weight.size()
        self.match_n_embd = self.dim_size // self.match_n_head
        self.only_target = only_target
        
        self.wte = wte
        self.n_tokens = n_tokens         
        self.uni_tokens = uni_tokens 
        self.share_tokens = share_tokens 
        self.target_tokens = target_tokens
        assert self.n_tokens == self.share_tokens+self.uni_tokens*2+self.target_tokens, 'the total prefix length is not equal to share + unique length'
        
        self.wte_prefix_self = nn.Embedding(self.share_tokens+2*self.uni_tokens+self.target_tokens, self.dim_size)
        self.wte_prefix_cros = nn.Embedding(self.share_tokens+2*self.uni_tokens+self.target_tokens, self.dim_size)
        self.wte_prefix_encd = nn.Embedding(self.share_tokens+2*self.uni_tokens+self.target_tokens, self.dim_size)
        
        self.seed_tokens_debate = torch.cat([torch.arange(self.share_tokens+2*self.uni_tokens+self.target_tokens).long()])      
        self.seed_tokens_squad  = torch.cat([torch.arange(self.share_tokens+2*self.uni_tokens+self.target_tokens).long()])   
        self.seed_tokens_xsum   = torch.cat([torch.arange(self.share_tokens+2*self.uni_tokens+self.target_tokens).long()])  
        
        
        self.control_trans_self = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))
        self.control_trans_cros = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))
        self.control_trans_encd = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))                                                  
                                                    

            
    def forward(self, tokens, prefix_1, prefix_2, sample_size=1):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        #self.only_target=0
        
        input_embedding = self.wte(tokens)
        bsz=input_embedding.size(0)

        old_bsz = bsz
        bsz = bsz * sample_size
        
        if prefix_1==0 and prefix_2==0: 
            seed_tokens=self.seed_tokens_debate
        if prefix_1==0 and prefix_2==1: 
            seed_tokens=self.seed_tokens_xsum      
        if prefix_1==1 and prefix_2==0: 
            seed_tokens=self.seed_tokens_squad  
            
            
        seed_tokens_expanded = seed_tokens.unsqueeze(0).expand(bsz, -1).cuda()
        input_tokens_enc = seed_tokens.unsqueeze(0).expand(old_bsz, -1).cuda()           


        temp_control = self.wte_prefix_self(seed_tokens_expanded)
        past_key_values = self.control_trans_self(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        

        temp_control2 = self.wte_prefix_cros(seed_tokens_expanded)
        past_key_values2 = self.control_trans_cros(temp_control2)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values2.shape
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)



        temp_control_enc = self.wte_prefix_encd(input_tokens_enc)
        past_key_values_enc = self.control_trans_encd(temp_control_enc)  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)            
        
        assert self.n_tokens == seqlen, 'the total prefix length is not equal to mask length'        



        if prefix_1==0 and prefix_2==0 and self.only_target==0: 
            
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.zeros(bsz, seqlen).cuda().bool() #bsz, preseqlen
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.zeros(bsz, seqlen).cuda().bool() #bsz, preseqlen
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).cuda().bool() #bsz, preseqlen
                                        }
                result.append(temp_dict)
                

        if prefix_1==0 and prefix_2==0 and self.only_target==1: 
            
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.cat([torch.full([bsz, self.share_tokens+2*self.uni_tokens],1).bool().cuda(),torch.full([bsz, self.target_tokens],0).bool().cuda()], 1)
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.cat([torch.full([bsz, self.share_tokens+2*self.uni_tokens],1).bool().cuda(),torch.full([bsz, self.target_tokens],0).bool().cuda()], 1)
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.cat([torch.full([bsz_enc, self.share_tokens+2*self.uni_tokens],1).bool().cuda(),torch.full([bsz_enc, self.target_tokens],0).bool().cuda()], 1)
                                        }
                result.append(temp_dict)
   
                
                
        if prefix_1==1 and prefix_2==0: 
        
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens).cuda().bool(), torch.zeros(bsz, self.uni_tokens).cuda().bool(),torch.full([bsz, self.uni_tokens+self.target_tokens],1).bool().cuda()], 1)
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens).cuda().bool(),torch.zeros(bsz, self.uni_tokens).cuda().bool(),torch.full([bsz, self.uni_tokens+self.target_tokens],1).bool().cuda()], 1)
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.cat([torch.zeros(bsz_enc, self.share_tokens).cuda().bool(),torch.zeros(bsz_enc, self.uni_tokens).cuda().bool(),torch.full([bsz_enc, self.uni_tokens+self.target_tokens],1).bool().cuda()], 1)
                                        }
                result.append(temp_dict)
             
                
        if prefix_1==0 and prefix_2==1: 
        
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens).cuda().bool(), torch.full([bsz, self.uni_tokens],1).bool().cuda(), torch.zeros(bsz, self.uni_tokens).cuda().bool(),torch.full([bsz, self.target_tokens],1).bool().cuda()], 1)
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens).cuda().bool(), torch.full([bsz, self.uni_tokens],1).bool().cuda(), torch.zeros(bsz, self.uni_tokens).cuda().bool(),torch.full([bsz, self.target_tokens],1).bool().cuda()], 1)
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.cat([torch.zeros(bsz_enc, self.share_tokens).cuda().bool(), torch.full([bsz_enc, self.uni_tokens],1).bool().cuda(),torch.zeros(bsz_enc, self.uni_tokens).cuda().bool(),torch.full([bsz_enc, self.target_tokens],1).bool().cuda()], 1)
                                        }
                result.append(temp_dict)
                
        return result 
    
    
    
class activation_prefix_embedding_original_simple_all_divide_full_data(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 20, 
                uni_tokens: int = 20,
                share_tokens: int = 0,
                target_tokens: int = 0,
                decoder_layers: int = 12,
                decoder_attention_heads: int = 16,                
                random_range: float = 0.5,
                initialize_from_vocab: bool = True,
                prefix_dropout: float = 0.0,
                only_target: int = 0):

        super(activation_prefix_embedding_original_simple_all_divide_full_data, self).__init__()
        
        self.dropout = nn.Dropout(prefix_dropout)
        
        self.match_n_layer = decoder_layers
        self.match_n_head =  decoder_attention_heads
        self.voc_size,self.dim_size=wte.weight.size()
        self.match_n_embd = self.dim_size // self.match_n_head
        self.only_target = only_target
        
        self.wte = wte
        self.n_tokens = n_tokens         
        self.uni_tokens = uni_tokens 
        self.share_tokens = share_tokens 
        self.target_tokens = target_tokens
        #assert self.n_tokens == self.share_tokens+self.uni_tokens*2+self.target_tokens, 'the total prefix length is not equal to share + unique length'
        
        self.wte_prefix_self = nn.Embedding(self.share_tokens+2*self.uni_tokens+self.target_tokens, self.dim_size)
        self.wte_prefix_cros = nn.Embedding(self.share_tokens+2*self.uni_tokens+self.target_tokens, self.dim_size)
        self.wte_prefix_encd = nn.Embedding(self.share_tokens+2*self.uni_tokens+self.target_tokens, self.dim_size)
        
        self.seed_tokens_debate = torch.arange(2*self.uni_tokens+self.share_tokens).long()     
        self.seed_tokens_squad  = torch.cat([torch.arange(0, self.uni_tokens).long(), torch.arange(self.uni_tokens*2,self.uni_tokens*2+self.share_tokens).long()])  
        self.seed_tokens_xsum   = torch.cat([torch.arange(self.uni_tokens,self.uni_tokens*2).long(), torch.arange(self.uni_tokens*2,self.uni_tokens*2+self.share_tokens).long()])  
        
        
        self.control_trans_self = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))
        self.control_trans_cros = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))
        self.control_trans_encd = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))                                                  
                                                    

            
    def forward(self, tokens, prefix_1, prefix_2, sample_size=1):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens)
        bsz=input_embedding.size(0)
        
        old_bsz = bsz
        bsz = bsz * sample_size
        
        if prefix_1==0 and prefix_2==0: 
            seed_tokens=self.seed_tokens_debate
        if prefix_1==0 and prefix_2==1: 
            seed_tokens=self.seed_tokens_xsum      
        if prefix_1==1 and prefix_2==0: 
            seed_tokens=self.seed_tokens_squad  
            
            
        seed_tokens_expanded = seed_tokens.unsqueeze(0).expand(bsz, -1).cuda()
        input_tokens_enc = seed_tokens.unsqueeze(0).expand(old_bsz, -1).cuda()           


        temp_control = self.wte_prefix_self(seed_tokens_expanded)
        past_key_values = self.control_trans_self(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        

        temp_control2 = self.wte_prefix_cros(seed_tokens_expanded)
        past_key_values2 = self.control_trans_cros(temp_control2)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values2.shape
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)



        temp_control_enc = self.wte_prefix_encd(input_tokens_enc)
        past_key_values_enc = self.control_trans_encd(temp_control_enc)  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)                   



        if prefix_1==0 and prefix_2==0 and self.only_target==0: 
            
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.zeros(bsz, seqlen).cuda().bool() #bsz, preseqlen
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.zeros(bsz, seqlen).cuda().bool() #bsz, preseqlen
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).cuda().bool() #bsz, preseqlen
                                        }
                result.append(temp_dict)
                

        if prefix_1==0 and prefix_2==0 and self.only_target==1: 
            
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.cat([torch.full([bsz, self.share_tokens+2*self.uni_tokens],1).bool().cuda(),torch.full([bsz, self.target_tokens],0).bool().cuda()], 1)
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.cat([torch.full([bsz, self.share_tokens+2*self.uni_tokens],1).bool().cuda(),torch.full([bsz, self.target_tokens],0).bool().cuda()], 1)
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.cat([torch.full([bsz_enc, self.share_tokens+2*self.uni_tokens],1).bool().cuda(),torch.full([bsz_enc, self.target_tokens],0).bool().cuda()], 1)
                                        }
                result.append(temp_dict)
   
                
                
        if (prefix_1==1 and prefix_2==0) or (prefix_1==0 and prefix_2==1): 
        
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.zeros(bsz, seqlen).cuda().bool() #bsz, preseqlen
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.zeros(bsz, seqlen).cuda().bool() #bsz, preseqlen
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen).cuda().bool() #bsz, preseqlen
                                        }
                result.append(temp_dict)
                
        return result 

class activation_prefix_embedding_original_simple_all_full_data(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 20, 
                uni_tokens: int = 20,
                share_tokens: int = 0,
                target_tokens: int = 0,
                decoder_layers: int = 12,
                decoder_attention_heads: int = 16,                
                random_range: float = 0.5,
                initialize_from_vocab: bool = True,
                prefix_dropout: float = 0.0):

        super(activation_prefix_embedding_original_simple_all_full_data, self).__init__()
        
        self.dropout = nn.Dropout(prefix_dropout)
        
        self.match_n_layer = decoder_layers
        self.match_n_head =  decoder_attention_heads
        self.voc_size,self.dim_size=wte.weight.size()
        self.match_n_embd = self.dim_size // self.match_n_head
        
        self.wte = wte
        self.n_tokens = n_tokens         
        self.uni_tokens = uni_tokens 
        self.share_tokens = share_tokens 
        self.target_tokens = target_tokens
        assert self.n_tokens == self.share_tokens+self.uni_tokens*2+self.target_tokens, 'the total prefix length is not equal to share + unique length'
        
        self.wte_prefix_self = nn.Embedding(self.share_tokens+2*self.uni_tokens+self.target_tokens, self.dim_size)
        self.wte_prefix_cros = nn.Embedding(self.share_tokens+2*self.uni_tokens+self.target_tokens, self.dim_size)
        self.wte_prefix_encd = nn.Embedding(self.share_tokens+2*self.uni_tokens+self.target_tokens, self.dim_size)
        
        self.seed_tokens_debate = torch.cat([torch.arange(self.share_tokens+2*self.uni_tokens+self.target_tokens).long()])      
        self.seed_tokens_squad  = torch.cat([torch.arange(self.share_tokens+2*self.uni_tokens+self.target_tokens).long()])   
        self.seed_tokens_xsum   = torch.cat([torch.arange(self.share_tokens+2*self.uni_tokens+self.target_tokens).long()])  
        
        
        self.control_trans_self = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))
        self.control_trans_cros = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))
        self.control_trans_encd = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))                                                  
                                                    

            
    def forward(self, tokens, prefix_1, prefix_2):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens)
        bsz=input_embedding.size(0)
        
        if prefix_1==0 and prefix_2==0: 
            seed_tokens=self.seed_tokens_debate
        if prefix_1==0 and prefix_2==1: 
            seed_tokens=self.seed_tokens_xsum      
        if prefix_1==1 and prefix_2==0: 
            seed_tokens=self.seed_tokens_squad  
            
            
        seed_tokens_expanded = seed_tokens.unsqueeze(0).expand(bsz, -1).cuda()
        


        temp_control = self.wte_prefix_self(seed_tokens_expanded)
        past_key_values = self.control_trans_self(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        

        temp_control2 = self.wte_prefix_cros(seed_tokens_expanded)
        past_key_values2 = self.control_trans_cros(temp_control2)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values2.shape
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)



        temp_control_enc = self.wte_prefix_encd(seed_tokens_expanded)
        past_key_values_enc = self.control_trans_encd(temp_control_enc)  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)            
        
        assert self.n_tokens == seqlen, 'the total prefix length is not equal to mask length'        
                

        if prefix_1==0 and prefix_2==0: 
            
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens).cuda().bool(), torch.full([bsz, 2*self.uni_tokens],1).bool().cuda(),torch.full([bsz, self.target_tokens],0).bool().cuda()], 1)
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens).cuda().bool(), torch.full([bsz, 2*self.uni_tokens],1).bool().cuda(),torch.full([bsz, self.target_tokens],0).bool().cuda()], 1)
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens).cuda().bool(), torch.full([bsz, 2*self.uni_tokens],1).bool().cuda(),torch.full([bsz, self.target_tokens],0).bool().cuda()], 1)
                                        }
                result.append(temp_dict)
   
                
                
        if prefix_1==1 and prefix_2==0: 
        
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens).cuda().bool(), torch.zeros(bsz, self.uni_tokens).cuda().bool(),torch.full([bsz, self.uni_tokens+self.target_tokens],1).bool().cuda()], 1)
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens).cuda().bool(),torch.zeros(bsz, self.uni_tokens).cuda().bool(),torch.full([bsz, self.uni_tokens+self.target_tokens],1).bool().cuda()], 1)
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens).cuda().bool(),torch.zeros(bsz, self.uni_tokens).cuda().bool(),torch.full([bsz, self.uni_tokens+self.target_tokens],1).bool().cuda()], 1)
                                        }
                result.append(temp_dict)
             
                
        if prefix_1==0 and prefix_2==1: 
        
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens).cuda().bool(), torch.full([bsz, self.uni_tokens],1).bool().cuda(), torch.zeros(bsz, self.uni_tokens).cuda().bool(),torch.full([bsz, self.target_tokens],1).bool().cuda()], 1)
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens).cuda().bool(), torch.full([bsz, self.uni_tokens],1).bool().cuda(), torch.zeros(bsz, self.uni_tokens).cuda().bool(),torch.full([bsz, self.target_tokens],1).bool().cuda()], 1)
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens).cuda().bool(), torch.full([bsz, self.uni_tokens],1).bool().cuda(),torch.zeros(bsz, self.uni_tokens).cuda().bool(),torch.full([bsz, self.target_tokens],1).bool().cuda()], 1)
                                        }
                result.append(temp_dict)
                
        return result     
    

class activation_prefix_embedding_original_simple_all_full_data_fisher(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 20, 
                uni_tokens: int = 20,
                share_tokens: int = 0,
                target_tokens: int = 0,
                decoder_layers: int = 12,
                decoder_attention_heads: int = 16,                
                random_range: float = 0.5,
                initialize_from_vocab: bool = True,
                prefix_dropout: float = 0.0):

        super(activation_prefix_embedding_original_simple_all_full_data_fisher, self).__init__()
        
        self.dropout = nn.Dropout(prefix_dropout)
        
        self.match_n_layer = decoder_layers
        self.match_n_head =  decoder_attention_heads
        self.voc_size,self.dim_size=wte.weight.size()
        self.match_n_embd = self.dim_size // self.match_n_head
        
        self.wte = wte
        self.n_tokens = n_tokens         
        self.uni_tokens = uni_tokens 
        self.share_tokens = share_tokens 
        self.target_tokens = target_tokens
        assert self.n_tokens == self.share_tokens+self.uni_tokens*2+self.target_tokens, 'the total prefix length is not equal to share + unique length'
        
        self.wte_prefix_self = nn.Embedding(self.share_tokens+2*self.uni_tokens+self.target_tokens, self.dim_size)
        self.wte_prefix_cros = nn.Embedding(self.share_tokens+2*self.uni_tokens+self.target_tokens, self.dim_size)
        self.wte_prefix_encd = nn.Embedding(self.share_tokens+2*self.uni_tokens+self.target_tokens, self.dim_size)
        
        self.seed_tokens_debate = torch.cat([torch.arange(self.share_tokens+2*self.uni_tokens+self.target_tokens).long()])      
        self.seed_tokens_squad  = torch.cat([torch.arange(self.share_tokens+2*self.uni_tokens+self.target_tokens).long()])   
        self.seed_tokens_xsum   = torch.cat([torch.arange(self.share_tokens+2*self.uni_tokens+self.target_tokens).long()])  
        

        self.squad_mask_self=[1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        self.squad_mask_cros=[1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
        self.squad_mask_encd=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0]
        
        self.xsum_mask_self=[1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]
        self.xsum_mask_cros=[1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        self.xsum_mask_encd=[0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1]        
        
        
        
        
        
        self.control_trans_self = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))
        self.control_trans_cros = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))
        self.control_trans_encd = nn.Sequential(
                    nn.Linear(self.dim_size, 800),
                    nn.Tanh(),
                    nn.Linear(800, self.dim_size*2*12))                                                  
                                                    

            
    def forward(self, tokens, prefix_1, prefix_2):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens)
        bsz=input_embedding.size(0)
        
        if prefix_1==0 and prefix_2==0: 
            seed_tokens=self.seed_tokens_debate
        if prefix_1==0 and prefix_2==1: 
            seed_tokens=self.seed_tokens_xsum      
        if prefix_1==1 and prefix_2==0: 
            seed_tokens=self.seed_tokens_squad  
            
            
        seed_tokens_expanded = seed_tokens.unsqueeze(0).expand(bsz, -1).cuda()
        


        temp_control = self.wte_prefix_self(seed_tokens_expanded)
        past_key_values = self.control_trans_self(temp_control) #bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        

        temp_control2 = self.wte_prefix_cros(seed_tokens_expanded)
        past_key_values2 = self.control_trans_cros(temp_control2)  # bsz, seqlen, layer*emb
        bsz, seqlen, _ = past_key_values2.shape
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)



        temp_control_enc = self.wte_prefix_encd(seed_tokens_expanded)
        past_key_values_enc = self.control_trans_encd(temp_control_enc)  # bsz, seqlen, layer*emb
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                 self.match_n_embd)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)            
        
        assert self.n_tokens == seqlen, 'the total prefix length is not equal to mask length'        
                

        if prefix_1==0 and prefix_2==0: 
            
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens).cuda().bool(), torch.full([bsz, 2*self.uni_tokens],1).bool().cuda(),torch.full([bsz, self.target_tokens],0).bool().cuda()], 1)
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens).cuda().bool(), torch.full([bsz, 2*self.uni_tokens],1).bool().cuda(),torch.full([bsz, self.target_tokens],0).bool().cuda()], 1)
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.cat([torch.zeros(bsz, self.share_tokens).cuda().bool(), torch.full([bsz, 2*self.uni_tokens],1).bool().cuda(),torch.full([bsz, self.target_tokens],0).bool().cuda()], 1)
                                        }
                result.append(temp_dict)
   
                
                
        if prefix_1==1 and prefix_2==0: 
        
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask": torch.cat([torch.tensor(self.squad_mask_self).expand(bsz,len(self.xsum_mask_cros)).bool().cuda(), torch.full([bsz, self.target_tokens],1).bool().cuda()], 1)
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.cat([torch.tensor(self.squad_mask_cros).expand(bsz,len(self.xsum_mask_cros)).bool().cuda(), torch.full([bsz, self.target_tokens],1).bool().cuda()], 1)
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.cat([torch.tensor(self.squad_mask_encd).bool().expand(bsz,len(self.xsum_mask_cros)).cuda(), torch.full([bsz, self.target_tokens],1).bool().cuda()], 1)
                                        }
                result.append(temp_dict)
             
                
        if prefix_1==0 and prefix_2==1: 
        
            result = []
            for i, key_val in enumerate(past_key_values):
                temp_dict = {'self': {"prev_key": key_val[0].contiguous(),
                                      "prev_value": key_val[1].contiguous(),
                                      "prev_key_padding_mask":torch.cat([torch.tensor(self.xsum_mask_self).expand(bsz,len(self.xsum_mask_cros)).bool().cuda(), torch.full([bsz, self.target_tokens],1).bool().cuda()], 1)
                                     },
                            }
    
                key_val2 = past_key_values2[i]
                temp_dict['encoder_decoder'] = {"prev_key": key_val2[0].contiguous(),
                                                "prev_value": key_val2[1].contiguous(),
                                                "prev_key_padding_mask": torch.cat([torch.tensor(self.xsum_mask_cros).expand(bsz,len(self.xsum_mask_cros)).bool().cuda(), torch.full([bsz, self.target_tokens],1).bool().cuda()], 1)
                                                }
    
                key_val_enc = past_key_values_enc[i]
                temp_dict['encoder'] = {"prev_key": key_val_enc[0].contiguous(),
                                        "prev_value": key_val_enc[1].contiguous(),
                                        "prev_key_padding_mask": torch.cat([torch.tensor(self.xsum_mask_encd).expand(bsz, len(self.xsum_mask_cros)).bool().cuda(), torch.full([bsz, self.target_tokens], 1).bool().cuda()], 1)
                                        }
                result.append(temp_dict)
                
        return result    


    
