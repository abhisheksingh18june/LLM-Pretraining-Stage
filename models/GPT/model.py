import torch
import torch.nn as nn 
from models.GPT.config import Config
from models.GPT.block import TransformerBlock
from models.GPT.layer_norm import LayerNorm

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.token_embedding=nn.Embedding(num_embeddings=config.vocab_size,embedding_dim=config.d_emb)
        self.positional_embedding=nn.Embedding(num_embeddings=config.n_blocks,embedding_dim=config.d_emb)
        self.transformer_blocks=nn.Sequential(
         *[TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_layernorm=LayerNorm(config)
        self.final_projection=nn.Linear(in_features=config.d_emb,out_features=config.vocab_size,bias=False)

    def forward(self,x):
        _,tokens=x.shape
        token_embeddings=self.token_embedding(x)
        pos_embeddings=self.positional_embedding(torch.arange(0,tokens))
        x=token_embeddings+pos_embeddings 

        x=self.transformer_blocks(x)
        x=self.final_layernorm(x)
        x=self.final_projection(x)

        return x 
    

if __name__=="__main__":
    obj=GPT(Config())
    total_params=0
    for p in obj.parameters():
        total_params+=p.numel()

    print(f"Total Trainable Params are: {total_params}")


