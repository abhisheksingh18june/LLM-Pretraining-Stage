import tiktoken,torch
from models.GPT.model import GPT
from models.GPT.config import Config 
from data.utils import responseGenerator
from data.losses import LossComputer
from data.dataloader import createDataloader
from data.optimize import createOptimizer


class Trainer:
    def __init__(self):
        self.loss_computer=LossComputer()
        self.response_obj=responseGenerator()

    def train_model_simple(self,model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, start_context, tokenizer):
        
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1


        for epoch in range(num_epochs):
            model.train()  
            
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad() 
                loss =  self.loss_computer.calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward() 
                optimizer.step() 
                tokens_seen += input_batch.numel()
                global_step += 1

                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate_model(
                        model, train_loader, val_loader, device)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:03d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
            
            print(f"Epoch {epoch+1} Ended , Now Going to generate sample text")
            self.generate_and_print_sample(
                model, tokenizer, device, start_context
            )

        return train_losses, val_losses, track_tokens_seen
    
    def evaluate_model(self,model, train_loader, val_loader, device):
        model.eval()
        with torch.no_grad():
            train_loss = self.loss_computer.calc_loss_loader(train_loader, model, device)
            val_loss = self.loss_computer.calc_loss_loader(val_loader, model, device)
        model.train()
        return train_loss, val_loss


    def generate_and_print_sample(self, model, tokenizer, device, start_context):
        model.eval()
        context_size = Config().n_blocks
        encoded = self.response_obj.text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            token_ids = self.response_obj.generate_text_simple(
                model=model, idx=encoded,
                max_new_tokens=15, context_size=context_size
            )
        decoded_text = self.response_obj.token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  
        model.train()



if __name__=="__main__":
    with open("data/the-verdict.txt","r") as f:
        text_data=f.read()

    device="cuda" if torch.cuda.is_available() else "cpu"
    model=GPT(Config())
    tokenizer=tiktoken.get_encoding("gpt2")
    optimizer=createOptimizer().create_optimiser(model)
    torch.manual_seed(123) 


    train_ratio = 0.80
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    obj=createDataloader(batch_size=2,max_length=256,stride=256,shuffle=True,drop_last=True,num_workers=0)

    train_loader=obj.loader(train_data)
    val_loader=obj.loader(val_data)

    model.to(device)

    trainer=Trainer().train_model_simple(
                                         model=model,
                                         train_loader=train_loader,
                                         val_loader=val_loader,
                                         optimizer=optimizer,
                                         device=device,
                                         num_epochs=2,
                                         eval_freq=5,
                                         start_context="Hi, I am Abhishek",
                                         tokenizer=tokenizer
                                         )






