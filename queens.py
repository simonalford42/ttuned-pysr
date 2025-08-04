#!/usr/bin/env python
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse

# ---------- Part 1: Backtracking solver with trace logging ----------

def solve_nqueens(n, randomize=False):
    board = [-1] * n
    trace = []
    steps = 0

    def is_safe(r, c):
        for prev_r in range(r):
            prev_c = board[prev_r]
            if prev_c == c or abs(prev_c - c) == abs(prev_r - r):
                return False
        return True

    cols = list(range(n))
    def backtrack(r):
        nonlocal steps
        if r == n:
            trace.append("solved")
            return True
        order = cols.copy()
        if randomize:
            random.shuffle(order)
        for c in order:
            steps += 1
            trace.append(f"state {board[:r]}")
            trace.append(f"place ({r},{c})")
            if is_safe(r, c):
                board[r] = c
                if backtrack(r+1):
                    return True
                trace.append(f"remove ({r},{c})")
                board[r] = -1
            else:
                trace.append(f"remove ({r},{c})")
        return False

    solved = backtrack(0)
    return trace, steps, solved

# ---------- Part 2: Simulation and statistics ----------

def simulate(n, trials=1, randomize=False):
    stats = {"total_steps": 0, "solved_count": 0}
    traces = []
    for _ in range(trials):
        trace, steps, solved = solve_nqueens(n, randomize=randomize)
        stats["total_steps"]   += steps
        stats["solved_count"]  += int(solved)
        traces.append({
            "prompt": f"NQueens N={n}",
            "trace": trace
        })
    stats["avg_steps"]      = stats["total_steps"] / trials
    stats["percent_solved"] = stats["solved_count"] / trials * 100
    return stats, traces

def generate_dataset(path, n, trials=1000, randomize=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stats, traces = simulate(n, trials, randomize=randomize)
    with open(path, 'w') as f:
        for ex in traces:
            f.write(json.dumps(ex) + "\n")
    print(f"Generated {trials} traces for N={n}")
    print(f"Average steps: {stats['avg_steps']:.1f}, Solved: {stats['percent_solved']:.1f}%")
    return stats

# ---------- Part 3: Vocabulary & encoding ----------

def build_vocab(jsonl_path, vocab_path):
    tokens = {"<bos>", "<eos>"}
    with open(jsonl_path) as f:
        for line in f:
            ex = json.loads(line)
            tokens.update(ex["trace"])
    token2id = {tok: i for i, tok in enumerate(sorted(tokens))}
    with open(vocab_path, 'w') as f:
        json.dump(token2id, f)
    return token2id

def encode_dataset(jsonl_path, token2id, max_len=128):
    X, Y = [], []
    with open(jsonl_path) as f:
        for line in f:
            ex = json.loads(line)
            seq = ["<bos>"] + ex["trace"] + ["<eos>"]
            ids = [token2id[t] for t in seq]
            # pad/truncate to max_len+1
            if len(ids) < max_len+1:
                ids += [0] * (max_len+1 - len(ids))
            else:
                ids = ids[:max_len+1]
            X.append(ids[:max_len])
            Y.append(ids[1:max_len+1])
    return np.array(X, dtype=np.int64), np.array(Y, dtype=np.int64)

# ---------- Part 4: PyTorch model & training loop ----------

class TraceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=256, max_len=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.attn    = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1     = nn.LayerNorm(d_model)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln2     = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h   = self.tok_emb(x) + self.pos_emb(pos)
        attn_out, _ = self.attn(h, h, h)
        h = self.ln1(h + attn_out)
        h2 = self.ffn(h)
        h  = self.ln2(h + h2)
        return self.head(h)

def train_pytorch(jsonl_path, vocab_path, model_path,
                  max_len=128, batch_size=32, epochs=20, lr=1e-3):
    token2id = json.load(open(vocab_path))
    X, Y = encode_dataset(jsonl_path, token2id, max_len)

    # ─── split into train / val / test ─────────
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val,   Y_val   = X[val_idx],   Y[val_idx]
    X_test,  Y_test  = X[test_idx],  Y[test_idx]

    train_loader = DataLoader(TraceDataset(X_train, Y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TraceDataset(X_val,   Y_val),
                              batch_size=batch_size)
    test_loader  = DataLoader(TraceDataset(X_test,  Y_test),
                              batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TinyTransformer(len(token2id), max_len=max_len).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    lossf  = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        model.train()
        cum_loss, total, correct = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)                          # (B, T, V)
            loss   = lossf(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            opt.step()

            cum_loss += loss.item() * xb.size(0)
            preds = logits.argmax(-1)
            # 'exact-sequence' accuracy
            correct += (preds == yb).all(dim=1).sum().item()
            total   += xb.size(0)

        # validation pass
        model.eval()
        val_loss, val_total, val_correct = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                L = lossf(logits.view(-1, logits.size(-1)),
                          yb.view(-1))
                val_loss += L.item() * xb.size(0)
                preds = logits.argmax(-1)
                val_correct += (preds == yb).all(dim=1).sum().item()
                val_total   += xb.size(0)
        print(f"Epoch {ep:2d} | train-loss={cum_loss/total:.4f} "
              f"| val-loss={val_loss/val_total:.4f} "
              f"| train-acc={(correct/total)*100:.2f}% "
              f"| val-acc={(val_correct/val_total)*100:.2f}%")

    torch.save(model.state_dict(), model_path)
    print(f"\nModel weights saved to {model_path}")


def expert_iteration_step(model, token2id, rev_token, n=6,
                          M=500, max_len=128,
                          device=None, fine_epochs=3,
                          batch_size=32, lr=1e-4):
    # ensure device
    device = device or next(model.parameters()).device

    # 1) Decode M times
    new_examples = []
    for _ in range(M):
        # start with <bos> + initial state []
        seq_ids = [token2id["<bos>"], token2id["state []"]]
        seq_ids += [0] * (max_len+1 - len(seq_ids))
        x = torch.tensor([seq_ids[:max_len]], device=device)  # shape (1, T)
        generated = seq_ids.copy()

        model.eval()
        with torch.no_grad():
            for _ in range(max_len):
                logits = model(x)                            # (1, T, V)
                nxt = logits[0, -1].argmax().item()
                generated.append(nxt)
                if rev_token[nxt] == "solved":
                    break
                # shift window in
                x = torch.cat((x[:,1:], torch.tensor([[nxt]], device=device)), dim=1)

        # 2) filter only if solved
        toks = [rev_token[i] for i in generated]
        if "solved" in toks:
            # drop leading <bos>, keep everything up to "solved"
            idx = toks.index("solved") + 1
            new_examples.append({
                "prompt": f"NQueens N={n}",
                "trace": toks[1:idx]  # e.g. ["state []","place (0,2)",...,"solved"]
            })

    print(f"Expert step: {len(new_examples)}/{M} solved by model → fine‑tuning")

    if not new_examples:
        print("No new solved examples, skipping fine‑tune")
        return

    # 3) build a small DataLoader
    # encode them just like before:
    Xn, Yn = [], []
    for ex in new_examples:
        seq = ["<bos>"] + ex["trace"] + ["<eos>"]
        ids = [token2id[t] for t in seq]
        if len(ids) < max_len+1:
            ids += [0]*(max_len+1-len(ids))
        else:
            ids = ids[:max_len+1]
        Xn.append(ids[:max_len])
        Yn.append(ids[1:max_len+1])
    Xn = torch.tensor(Xn, device=device, dtype=torch.long)
    Yn = torch.tensor(Yn, device=device, dtype=torch.long)
    dsn = torch.utils.data.TensorDataset(Xn, Yn)
    loader = DataLoader(dsn, batch_size=batch_size, shuffle=True)

    # 4) fine‑tune
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1, fine_epochs+1):
        cum, tot = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)                          # (B,T,V)
            L = lossf(logits.view(-1, logits.size(-1)), yb.view(-1))
            L.backward()
            opt.step()
            cum += L.item() * xb.size(0)
            tot += xb.size(0)
        print(f"  fine‑tune epoch {epoch}: loss={cum/tot:.4f}")

# ---------- Main ----------

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
