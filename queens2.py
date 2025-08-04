#!/usr/bin/env python
import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- Part 1: Backtracking solver with trace logging ----------

def solve_nqueens(n, randomize=False):
    board, trace, steps = [-1]*n, [], 0
    def is_safe(r,c):
        for pr in range(r):
            pc = board[pr]
            if pc==c or abs(pc-c)==abs(pr-r): return False
        return True
    cols = list(range(n))
    def backtrack(r):
        nonlocal steps
        if r==n:
            trace.append("solved")
            return True
        order = cols.copy()
        if randomize: random.shuffle(order)
        for c in order:
            steps += 1
            trace.append(f"state {board[:r]}")
            trace.append(f"place ({r},{c})")
            if is_safe(r,c):
                board[r]=c
                if backtrack(r+1): return True
                trace.append(f"remove ({r},{c})")
                board[r]=-1
            else:
                trace.append(f"remove ({r},{c})")
        return False
    solved = backtrack(0)
    return trace, steps, solved

def generate_dataset(path, n, trials=1000, randomize=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stats = {"total_steps":0, "solved_count":0}
    with open(path,'w') as f:
        for _ in range(trials):
            trace, steps, solved = solve_nqueens(n, randomize)
            stats["total_steps"]   += steps
            stats["solved_count"]  += int(solved)
            f.write(json.dumps({"prompt":f"NQueens N={n}","trace":trace})+"\n")
    print(f"Avg steps: {stats['total_steps']/trials:.1f}, Solved: {stats['solved_count']/trials*100:.1f}%")

# ---------- Part 2: Vocabulary & encoding ----------

def build_vocab(jsonl_path, vocab_path):
    toks = {"<bos>","<eos>"}
    with open(jsonl_path) as f:
        for L in f:
            ex = json.loads(L)
            toks.update(ex["trace"])
    token2id = {t:i for i,t in enumerate(sorted(toks))}
    json.dump(token2id, open(vocab_path,'w'))
    return token2id

def encode_dataset(jsonl_path, token2id, max_len=128):
    X,Y = [],[]
    with open(jsonl_path) as f:
        for L in f:
            ex=json.loads(L)
            seq=["<bos>"]+ex["trace"]+["<eos>"]
            ids=[token2id[t] for t in seq]
            if len(ids)<max_len+1: ids+=[0]*(max_len+1-len(ids))
            else: ids=ids[:max_len+1]
            X.append(ids[:max_len])
            Y.append(ids[1:max_len+1])
    return np.array(X,dtype=np.int64), np.array(Y,dtype=np.int64)

# ---------- Part 3: PyTorch model ----------

class TraceDataset(Dataset):
    def __init__(self,X,Y): self.X, self.Y = torch.from_numpy(X), torch.from_numpy(Y)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i], self.Y[i]

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=256, max_len=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.attn    = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1     = nn.LayerNorm(d_model)
        self.ffn     = nn.Sequential(nn.Linear(d_model,d_ff), nn.ReLU(), nn.Linear(d_ff,d_model))
        self.ln2     = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B,T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        attn_out,_ = self.attn(h,h,h)
        h = self.ln1(h+attn_out)
        h2 = self.ffn(h)
        h = self.ln2(h+h2)
        return self.head(h)

# ---------- Part 4: Training & Expert iteration ----------

def train_pytorch(model, train_loader, val_loader, device, epochs, lr):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    best_val = float('inf'); wait=0; patience=10
    for ep in range(1, epochs+1):
        # train
        model.train()
        cum, tot = 0.0, 0
        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            L = lossf(logits.view(-1,logits.size(-1)), yb.view(-1))
            L.backward(); opt.step()
            cum += L.item()*xb.size(0); tot += xb.size(0)
        train_loss = cum/tot
        # val
        model.eval()
        cum, tot = 0.0, 0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(device), yb.to(device)
                logits = model(xb)
                L = lossf(logits.view(-1,logits.size(-1)), yb.view(-1))
                cum += L.item()*xb.size(0); tot += xb.size(0)
        val_loss = cum/tot
        print(f"Epoch {ep:2d}: train-loss={train_loss:.4f}, val-loss={val_loss:.4f}")
        # early stop
        if val_loss + 1e-4 < best_val:
            best_val, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

def expert_iteration_step(model, token2id, rev_token, n, M, max_len, device, fine_epochs, batch_size, lr):
    new_examples=[]
    model.eval()
    for _ in range(M):
        seq_ids=[token2id["<bos>"],token2id["state []"]]+[0]*(max_len+1-2)
        x=torch.tensor([seq_ids[:max_len]],device=device)
        gen=seq_ids.copy()
        with torch.no_grad():
            for _ in range(max_len):
                logits=model(x)
                nxt=logits[0,-1].argmax().item()
                gen.append(nxt)
                if rev_token[nxt]=="solved": break
                x=torch.cat((x[:,1:],torch.tensor([[nxt]],device=device)),dim=1)
        toks=[rev_token[i] for i in gen]
        if "solved" in toks:
            idx=toks.index("solved")+1
            new_examples.append(toks[1:idx])
    print(f"Expert iteration: {len(new_examples)}/{M} solved")
    if not new_examples: return
    Xn,Yn=[],[]
    for trace in new_examples:
        seq=["<bos>"]+trace+["<eos>"]
        ids=[token2id[t] for t in seq]
        if len(ids)<max_len+1: ids+=[0]*(max_len+1-len(ids))
        else: ids=ids[:max_len+1]
        Xn.append(ids[:max_len]); Yn.append(ids[1:max_len+1])
    Xn=torch.tensor(Xn,device=device); Yn=torch.tensor(Yn,device=device)
    loader=DataLoader(torch.utils.data.TensorDataset(Xn,Yn), batch_size=batch_size, shuffle=True)
    opt=torch.optim.Adam(model.parameters(), lr=lr); lossf=nn.CrossEntropyLoss()
    model.train()
    for ep in range(1,fine_epochs+1):
        cum, tot = 0.0, 0
        for xb,yb in loader:
            xb,yb=xb.to(device), yb.to(device)
            opt.zero_grad()
            logits=model(xb)
            L=lossf(logits.view(-1,logits.size(-1)), yb.view(-1))
            L.backward(); opt.step()
            cum+=L.item()*xb.size(0); tot+=xb.size(0)
        print(f"  fine-tune ep{ep}: loss={cum/tot:.4f}")

# ---------- Main & CLI ----------

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode", choices=["bc","expert"], default="bc")
    p.add_argument("--dataset","-d", default="data/nqueens_6_traces.jsonl")
    p.add_argument("--vocab","-v",    default="data/nqueens_vocab.json")
    p.add_argument("--model","-m",    default="models/queens_transformer.pt")
    p.add_argument("--n",      type=int,   default=6)
    p.add_argument("--trials", type=int,   default=1000)
    p.add_argument("--epochs", type=int,   default=20)
    p.add_argument("--batch",  type=int,   default=32)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--M",      type=int,   default=500)
    p.add_argument("--f_epochs",type=int,  default=3)
    p.add_argument("--f_lr",   type=float, default=1e-4)
    args=p.parse_args()

    # build/generate if needed
    if not os.path.exists(args.dataset):
        generate_dataset(args.dataset, args.n, args.trials)
    if not os.path.exists(args.vocab):
        build_vocab(args.dataset, args.vocab)
    token2id=json.load(open(args.vocab))
    rev_token={i:t for t,i in token2id.items()}
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=TinyTransformer(len(token2id), max_len=128).to(device)

    if args.mode=="bc":
        # load existing or train
        if os.path.exists(args.model):
            model.load_state_dict(torch.load(args.model,map_location=device))
        X,Y=encode_dataset(args.dataset, token2id, max_len=128)
        # split 80/10/10
        N=len(X); idx=np.arange(N); np.random.shuffle(idx)
        n1=int(0.8*N); n2=int(0.1*N)
        tr,va=idx[:n1], idx[n1:n1+n2]
        Xtr,Ytr=X[tr],Y[tr]; Xva,Yva=X[va],Y[va]
        tr_loader=DataLoader(TraceDataset(Xtr,Ytr), batch_size=args.batch, shuffle=True)
        va_loader=DataLoader(TraceDataset(Xva,Yva), batch_size=args.batch)
        train_pytorch(model, tr_loader, va_loader, device, args.epochs, args.lr)
        torch.save(model.state_dict(), args.model)
        print(f"BC model saved to {args.model}")

    else:  # expert iteration
        if os.path.exists(args.model):
            model.load_state_dict(torch.load(args.model,map_location=device))
        expert_iteration_step(
            model, token2id, rev_token,
            n=args.n, M=args.M,
            max_len=128, device=device,
            fine_epochs=args.f_epochs,
            batch_size=args.batch,
            lr=args.f_lr
        )
        torch.save(model.state_dict(), args.model)
        print(f"Expert‑iter model updated at {args.model}")
