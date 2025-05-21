from dataclasses import dataclass

@dataclass
class Config:
    n_blocks=1024
    d_emb=768
    n_layers=12
    vocab_size=50257
    n_heads=12
    is_causal=True
    is_debug=False
    head_d_emb=d_emb//n_heads


if __name__=="__main__":
    config=Config()
    print(config.head_d_emb)
    print(config.is_debug)