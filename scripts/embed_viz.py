#!/usr/bin/env python
"""
Build a corpus of LlamaGuard-7B embeddings + safety scores.

Usage:
    python build_llamaguard_landscape.py --wiki 5000 --toxic 5000 \
        --model meta-llama/LlamaGuard-7b --out llamaguard_landscape.npz
"""

import argparse
import pickle
import random
import time

import numpy as np
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm


# --------------------------------------------------------------------- helpers
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wiki",   type=int, default=5000, help="# Wikipedia items")
    p.add_argument("--toxic",  type=int, default=5000, help="# toxicity items")
    p.add_argument("--model",  type=str,
                   default="meta-llama/LlamaGuard-7b", help="HF model id")
    p.add_argument("--batch",  type=int, default=8, help="forward batch size")
    p.add_argument("--out",    type=str, default="llamaguard_landscape.npz")
    return p.parse_args()

def load_llamaguard(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    safe_token = tok.encode("safe", add_special_tokens=False)

    mod = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    mod.eval()
    return tok, mod, safe_token

# embedding = mean-pool last layer (drop BOS)
@torch.no_grad()
def embed_batch(tok, mod, texts):
    # tokenize
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(mod.device)

    out = mod.model(**enc)                      # BaseModelOutputWithPast
    H = out.last_hidden_state                   # [B, T, D]
    mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]

    # drop BOS
    H = H[:, 1:, :]
    mask = mask[:, 1:, :]

    pooled = (H * mask).sum(1) / mask.sum(1).clamp_min(1)
    return F.normalize(pooled, p=2, dim=-1)

# unsafe probability via first-token softmax
@torch.no_grad()
def unsafe_prob(tok, mod, texts, safe_token):
    chats = [[{"role": "user", "content": text}] for text in texts]
    batch_input_ids = tok.apply_chat_template(
        chats, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=512
    ).to(mod.device)
    
    # Single forward pass
    logits = mod(batch_input_ids).logits
    last_logits = logits[:, -1, :]  # [B, vocab_size]
    probs = F.softmax(last_logits, dim=-1)
    return probs[:, safe_token[0]].cpu().numpy()


# @torch.no_grad()
# def unsafe_prob(tok, mod, texts, safe_token):
#     all_p_safe = []

#     # iterate over each text
#     for text in texts:
#         # convert to chat
#         chat = [{"role": "user", "content": text}]

#         # get input ids
#         input_ids = tok.apply_chat_template(
#             chat, return_tensors="pt").to(mod.device)

#         # [B, T, |V|]  [batch, seq length/#tokens in input, vocab size]
#         # (vocab in Llama is ~32k)
#         logits = mod(input_ids).logits     
    
#         # token after the prompt
#         first_logits = logits[0, -1]                
    
#         # probabilities over vocab
#         probs = F.softmax(first_logits, dim=-1)

#         # extract this prob from the full softmax list
#         p_safe = probs[safe_token[0]].item()

#         all_p_safe.append(p_safe)

#     all_p_safe = np.array(all_p_safe)
#     return all_p_safe


# @torch.no_grad()
# def unsafe_prob(tok, mod, texts):
#     # Build one conversation per text
#     convos = [[{"role":"user",  "content": t}] for t in texts]

#     enc = tok.apply_chat_template(
#         convos,
#         tokenize=True,
#         add_generation_prompt=True,   # predict first assistant token
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=512,
#     )

#     # ---- Normalize enc to input_ids / attention_mask on correct device ----
#     if isinstance(enc, torch.Tensor):
#         input_ids = enc.to(mod.device)
#         attention_mask = torch.ones_like(input_ids, device=mod.device)
#     else:
#         # BatchEncoding or dict-like
#         input_ids = enc["input_ids"].to(mod.device)
#         if "attention_mask" in enc and enc["attention_mask"] is not None:
#             attention_mask = enc["attention_mask"].to(mod.device)
#         else:
#             attention_mask = torch.ones_like(input_ids, device=mod.device)

#     # ---- Fast path: single-token SAFE/UNSAFE? ----
#     ids_unsafe = tok.encode(" UNSAFE", add_special_tokens=False)
#     ids_safe   = tok.encode(" SAFE",   add_special_tokens=False)

#     if len(ids_unsafe) == 1 and len(ids_safe) == 1:
#         out = mod(input_ids=input_ids, attention_mask=attention_mask)
#         first = out.logits[:, -1, :]                   # [B, |V|]
#         probs = F.softmax(first, dim=-1)
#         p_u = probs[:, ids_unsafe[0]]
#         p_s = probs[:, ids_safe[0]]
#         return (p_u / (p_u + p_s).clamp_min(1e-8))     # [B]

#     # ---- General route: log p(" UNSAFE") vs log p(" SAFE") ----
#     def str_logprob_next(target_str):
#         target = tok.encode(target_str, add_special_tokens=False)
#         B, T = input_ids.shape
#         pad = tok.pad_token_id or tok.eos_token_id
#         L = len(target)

#         x = torch.full((B, T + L), pad, dtype=input_ids.dtype, device=mod.device)
#         attn = torch.zeros_like(x)
#         x[:, :T] = input_ids
#         attn[:, :T] = attention_mask

#         for j, tid in enumerate(target):
#             x[:, T + j] = tid
#             attn[:, T + j] = 1

#         labels = x.clone()
#         labels[:, :T] = -100  # score only appended part

#         out = mod(input_ids=x, attention_mask=attn, labels=labels)
#         logits = out.logits[:, T-1:T+L-1, :]
#         logprobs = F.log_softmax(logits, dim=-1)
#         tgt = torch.tensor(target, device=mod.device).unsqueeze(0).expand(B, L)
#         return logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum(1)  # [B]

#     lp_u = str_logprob_next(" UNSAFE")
#     lp_s = str_logprob_next(" SAFE")
#     return torch.sigmoid(lp_u - lp_s)


# --------------------------------------------------------------------- main
def main():
    args = get_args()
    np.random.seed(0);  torch.manual_seed(0);  random.seed(0)

    t0 = time.time()

    # --- LOAD MODEL ---

    print(f"Loading model ... ({time.time()-t0:.3f})")
    tok, mod, safe_token = load_llamaguard(args.model)


    # --- LOAD DATASETS ---

    print(f"Loading datasets ... ({time.time()-t0:.3f})")
    wiki   = load_dataset("wikimedia/wikipedia", "20231101.en", 
                          split=f"train[:{args.wiki}]")
    toxic  = load_dataset("allenai/real-toxicity-prompts",
                           split=f"train[:{args.toxic}]")

    # print(f"[DEBUG] example wiki: {wiki[0]["text"]}")
    # print(f"[DEBUG] example toxic: {toxic[0]["prompt"]}")

    texts  = [x["text"] for x in wiki] + [x["prompt"]["text"] for x in toxic]
    print(f"Total samples before clean: {len(texts):,}")
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    sources= ["wiki"] * len(wiki) + ["toxic"] * len(toxic)
    print(f"Total samples: {len(texts):,}")


    # --- EMBED AND SCORE ---

    print(f"Embedding ... ({time.time()-t0:.3f})")
    all_emb, all_p = [], []
    for i in tqdm(range(0, len(texts), args.batch), desc="Embedding"):
        batch_txt = texts[i:i+args.batch]
        # emb  = embed_batch(tok, mod, batch_txt)
        pu   = unsafe_prob(tok, mod, batch_txt, safe_token)
        # all_emb.append(emb.cpu())
        # all_p.append( pu.cpu())
        all_p.append( pu )

    # just going to pickle whatever this is for now
    print(f"Pickling! FTW")
    with open("unsafe_p.pkl", "wb") as f:
        pickle.dump(all_p, f)

    print(type(all_p), len(all_p))
    
    # Z = torch.cat(all_emb).float().numpy()          # [N, 4096] float32
    y = torch.cat(all_p ).float().numpy()           # [N]       float32
    src = np.array(sources)

    

    # --- SAVE RESULTS ---

    print(f"Saving ... ({time.time()-t0:.3f})")
    # np.savez_compressed(args.out, embeddings=Z, p_unsafe=y, source=src, text=texts)
    np.savez_compressed(args.out, p_unsafe=y)
    # print(f"Saved to {args.out}  (embeddings.shape={Z.shape})")
    print(f"COMPLETE ... ({time.time()-t0:.3f})")

if __name__ == "__main__":
    main()
