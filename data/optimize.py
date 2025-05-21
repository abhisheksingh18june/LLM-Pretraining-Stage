import torch 

class createOptimizer:
    def create_optimiser(self,model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
        return optimizer 
    
'''
  12     def forward(self,x):
     13         _,tokens=x.shape
---> 14         token_embeddings=self.token_embedding(x)
     15         pos_embeddings=self.positional_embedding(torch.arange(0,tokens,device=x.device))
     16         x=token_embeddings+pos_embeddings

     RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument index in method wrapper_CUDA__index_select)
add Codeadd Markdown
'''