# @Author: Khush Patel, Zhi lab

from config import *

# If using bert architecture, please change as needed. 
m_config = transformers.BertConfig(
    vocab_size=90000,
    max_position_embeddings=256,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=1000,
    hidden_size=132,
    intermediate_size=64
)

# If using bert architecture, please change as needed. 
model = transformers.BertForMaskedLM(m_config)

# print(model)
