import torch
import tiktoken
from torch.utils.data import Dataset

class DatasetVersion1(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        super().__init__()
        self.input_ids,self.target_ids=[],[]

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        print(f"Total Number of Tokens are: {len(token_ids)}")

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index],self.target_ids[index]
    


if __name__=="__main__":
    txt="Hi I am Abhishek Singh"
    tokenizer=tiktoken.encoding_for_model("gpt2")
    obj=DatasetVersion1(txt,tokenizer,max_length=2,stride=2)
    print(len(obj))
    print(obj[:3])
    
