import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from PIL import Image
from utils import make_sprite

# ============ CONFIG ============
CONCEPT_FILE = "data/concept_file.pth"
MODEL_NAME = "bert-base-uncased"
LAYER_NAME = "encoder.layer.11.output"
IMAGE_DIR = "data/images"
TOP_K_CONCEPTS = 3

# ============ LOAD MODEL ============
st.title("Interactive Concept-Embedding Dashboard")
model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()

# ============ LOAD CONCEPT VECTORS ============
concepts = torch.load(CONCEPT_FILE)

# ============ HOOK LAYER ============
activation_store = {}
def hook_fn(module, input, output):
    activation_store['embeddings'] = output.detach()
for name, module in model.named_modules():
    if name == LAYER_NAME:
        module.register_forward_hook(hook_fn)

# ============ USER INPUT ============
prompt = st.text_input("Enter a prompt:", "The cat is sitting on the mat.")

if prompt:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = activation_store['embeddings'][0]  # [seq_len, hidden_dim]
    
    # ============ COSINE SIMILARITY ============
    emb_norm = F.normalize(embeddings, dim=-1)
    concept_norm = F.normalize(concepts, dim=-1)
    cos_sim = torch.matmul(emb_norm, concept_norm.T)  # [seq_len, num_concepts]

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    top_concepts_idx = cos_sim.topk(TOP_K_CONCEPTS, dim=-1).indices.tolist()

    st.subheader("Tokens and Top Activated Concepts")
    for i, token in enumerate(tokens):
        st.markdown(f"**Token:** `{token}`")
        top_idx = top_concepts_idx[i]
        cols = st.columns(TOP_K_CONCEPTS)
        for j, c_idx in enumerate(top_idx):
            img_path = os.path.join(IMAGE_DIR, f"concept_{c_idx}.png")
            if os.path.exists(img_path):
                cols[j].image(img_path, caption=f"Concept {c_idx}", use_column_width=True)
            else:
                cols[j].write(f"Concept {c_idx}")
