import torch.nn as nn
from models.GPT.config import Config 
from models.GPT.FFN import FeedForwardNN
from models.GPT.attention import CausalMultiHeadAttention
from models.GPT.layer_norm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attn=CausalMultiHeadAttention(config)
        self.layernorm1=LayerNorm(config)
        self.layernorm2=LayerNorm(config)
        self.ffn=FeedForwardNN(config)

    def forward(self,x):
        shortcut1=x
        x=self.layernorm1(x)
        x=self.attn(x)
        x=x+shortcut1 

        shortcut2=x
        x=self.layernorm2(x)
        x=self.ffn(x)
        x=x+shortcut2

        return x 
    

if __name__=="__main__":
    obj=TransformerBlock(Config())
    total_params=0
    for p in obj.parameters():
        total_params+=p.numel()

    print(f"Total Trainable Params are: {total_params}")


