import torch.nn as nn
from models.GPT.activation_fn import GELU
from models.GPT.config import Config

class FeedForwardNN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(in_features=config.d_emb,out_features=4*config.d_emb),
            GELU(),
            nn.Linear(in_features=4*config.d_emb,out_features=config.d_emb)
        )

    def forward(self,x):
        return self.layers(x)
    

if __name__=="__main__":
    obj=FeedForwardNN(Config())
    total_params=0
    for p in obj.parameters():
        total_params+=p.numel()
    print(f"TOtal Trainable Params are: {total_params}")


