import torch
import torch.nn as nn 
from models.GPT.config import Config

class CausalMultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.w_query=nn.Linear(config.d_emb,config.d_emb,bias=False)
        self.w_key=nn.Linear(config.d_emb,config.d_emb,bias=False)
        self.w_value=nn.Linear(config.d_emb,config.d_emb,bias=False)
        self.final_proj=nn.Linear(config.d_emb,config.d_emb,bias=False)
        self.register_buffer('mask',torch.triu(torch.ones(config.n_blocks,config.n_blocks),diagonal=1))

    def forward(self,x):
        batch,tokens,_=x.shape
        query=self.w_query(x)
        key=self.w_key(x)
        value=self.w_value(x)

        query=query.view(batch,tokens,self.config.n_heads,self.config.head_d_emb)
        key=key.view(batch,tokens,self.config.n_heads,self.config.head_d_emb)
        value=value.view(batch,tokens,self.config.n_heads,self.config.head_d_emb)
        
        if self.config.is_debug:
            print(f"Shape of Query:{query.shape}, Shape of Key:{key.shape}, Shape of Value:{value.shape}")
            

        query=query.transpose(1,2)
        key=key.transpose(1,2)
        value=value.transpose(1,2)

        if self.config.is_debug:
            print(f"Shape of Query After Transposing so as to make it compatible to compute attention per head for each token :{query.shape}, Shape of Key:{key.shape}, Shape of Value:{value.shape}")

        scores=query@key.transpose(2,3)
        masking_vector=self.mask.bool()[:tokens,:tokens]
        masked_scores=scores.masked_fill(masking_vector,-torch.inf)

        if self.config.is_debug:
            print(f"Shape of masked scores: {masked_scores.shape}")

        attention_weights=torch.softmax(masked_scores/(key.shape[-1])**0.5,dim=-1)
        if self.config.is_debug:
            print(f"Shape of attention weights: {attention_weights.shape} and Masked Vector is: {attention_weights}")

        context_vector=attention_weights@value 

        context_vector=context_vector.transpose(1,2)

        context_vector=context_vector.contiguous().view(batch,tokens,self.config.d_emb)

        context_vector=self.final_proj(context_vector)

        if self.config.is_debug:
            print(f"Shape of Context rich Vector: {context_vector.shape} and Context Vector is {context_vector}")

        return context_vector


if __name__=="__main__":
    cm=CausalMultiHeadAttention(Config())
    x= torch.randn(1,2,768)
    cm(x)

        

       

        
