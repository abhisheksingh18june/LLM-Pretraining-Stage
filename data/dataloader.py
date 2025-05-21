import tiktoken
from torch.utils.data import DataLoader
from data.create_dataset import DatasetVersion1


class createDataloader:
    def __init__(self,batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
        self.batch_size=batch_size
        self.max_length=max_length
        self.stride=stride 
        self.shuffle=shuffle 
        self.drop_last=drop_last
        self.num_workers=num_workers 

    def loader(self,txt):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = DatasetVersion1(txt, tokenizer, self.max_length, self.stride)

        dataloader=DataLoader(
        dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last, num_workers=self.num_workers)

        return dataloader

        
if __name__=="__main__":
    with open("the-verdict.txt" ,"r") as f:
        text_data=f.read()


    train_ratio = 0.80
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    obj=createDataloader(batch_size=2,max_length=256,stride=256,shuffle=True,drop_last=True,num_workers=0)

    train_loader=obj.loader(train_data)
    val_loader=obj.loader(val_data)
    
    print("Train loader:")
    print(f"Number of Batches in Training Set: {len(train_loader)} and Number of Batches in Validation Set: {len(val_loader)} ")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    