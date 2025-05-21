import torch 
import torch.nn as nn 
from models.GPT.config import Config

class LayerNorm(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.eps=1e-5
        self.scale=nn.Parameter(torch.ones(config.d_emb))
        self.shift=nn.Parameter(torch.zeros(config.d_emb))

    def forward(self,x):
        mean_value=x.mean(dim=-1,keepdim=True)
        std_value=x.std(dim=-1,keepdim=True)
        normalised_x=x-mean_value/(std_value+self.eps)
        return self.scale*normalised_x+self.shift
    
if __name__=="__main__":
    obj=LayerNorm(Config())
    total_params=0
    for p in obj.parameters():
        total_params+=p.numel()
    print(f"TOtal Trainable Params are: {total_params}")