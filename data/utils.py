import os , torch 
import urllib.request


class FileDownloadFromURL:
    def __init__(self,file_path,url):
        self.file_path=file_path 
        self.url=url 

    def load(self):
        if not os.path.exists(self.file_path):
            with urllib.request.urlopen(self.url) as response:
                text_data = response.read().decode('utf-8')
            with open(self.file_path, "w", encoding="utf-8") as file:
                file.write(text_data)
        else:
            print("File Already Exists")
            with open(self.file_path, "r", encoding="utf-8") as file:
                text_data = file.read()


class responseGenerator:
    def __init__(self,is_debug=False):
        self.is_debug=is_debug 

    def generate_text_simple(self,model, idx, max_new_tokens, context_size):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            
            if self.is_debug:
                print(f"Current Token Sequence getting used for the prediction is: {idx}")
            
            with torch.no_grad():
                logits = model(idx_cond)

            logits = logits[:, -1, :]

            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  

            idx = torch.cat((idx, idx_next), dim=1)  

        return idx
    
    def text_to_token_ids(self,text, tokenizer):
        encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        if self.is_debug:
            print(f"Encoded tensor is: {encoded_tensor}")
        return encoded_tensor

    def token_ids_to_text(self,token_ids, tokenizer):
        flat = token_ids.squeeze(0) 
        if self.is_debug:
            print(f"Flattened Tokens are: {flat}")
        return tokenizer.decode(flat.tolist())


if __name__=="__main__":
    pass
    # file_path = "the-verdict.txt"
    # url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    # obj=FileDownloadFromURL(file_path,url)
    # obj.load()

#     model=GPT(Config())
#     tokenizer = tiktoken.get_encoding("gpt2")
#     obj=responseGenerator(is_debug=True)
#     start_context="Hi I am Abhishek"

#     token_ids = obj.generate_text_simple(
#         model=model,
#         idx=obj.text_to_token_ids(start_context, tokenizer),
#         max_new_tokens=10,
#         context_size=Config().n_blocks
#     )

# print("Output text:\n", obj.token_ids_to_text(token_ids, tokenizer))
