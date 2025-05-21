import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        )) 
    

if __name__=="__main__":
    obj=GELU()
    total_params=0
    for p in obj.parameters():
        total_params+=p.numel()
    print(f"TOtal Trainable Params are: {total_params}")