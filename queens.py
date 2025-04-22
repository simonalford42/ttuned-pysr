import json
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout
from tensorflow.keras.models import Model

# ---------- Part 1: Backtracking solver with trace logging ----------

def solve_nqueens(n, randomize=False):
    """
    Solve N-Queens via backtracking, logging a trace.
    Returns: trace (list of tokens), steps (int), solved (bool)
    """
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
                # backtrack
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
        stats["total_steps"] += steps
        stats["solved_count"] += int(solved)
        traces.append({
            "prompt": f"NQueens N={n}",
            "trace": trace
        })
    stats["avg_steps"] = stats["total_steps"] / trials
    stats["percent_solved"] = stats["solved_count"] / trials * 100
    return stats, traces

# ---------- Part 3: Generate dataset ----------

def generate_dataset(output_path, n, trials=1000, randomize=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stats, traces = simulate(n, trials, randomize=randomize)
    with open(output_path, 'w') as f:
        for ex in traces:
            f.write(json.dumps(ex) + '\n')
    print(f"Generated {trials} traces for N={n}")
    print(f"Average steps: {stats['avg_steps']:.2f}, Solved: {stats['percent_solved']:.1f}%")
    return stats

# ---------- Part 4: Build vocabulary and data pipeline ----------

def build_vocab(jsonl_path, vocab_path):
    tokens = set()
    with open(jsonl_path) as f:
        for line in f:
            ex = json.loads(line)
            tokens.add("<bos>")
            tokens.add("<eos>")
            for t in ex['trace']:
                tokens.add(t)
    tokens = sorted(tokens)
    token2id = {t: i for i, t in enumerate(tokens)}
    with open(vocab_path, 'w') as f:
        json.dump(token2id, f)
    return token2id


def encode_dataset(jsonl_path, token2id, max_len=128):
    X, y = [], []
    with open(jsonl_path) as f:
        for line in f:
            ex = json.loads(line)
            seq = ['<bos>'] + ex['trace'] + ['<eos>']
            ids = [token2id[t] for t in seq]
            ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
            X.append(ids[:-1])
            y.append(ids[1:])
    return np.array(X), np.array(y)

# ---------- Part 5: Define Transformer model in Keras ----------

def create_transformer_model(vocab_size, max_len=128, d_model=128, num_heads=4, dff=256):
    inputs = Input(shape=(max_len,), dtype='int32')
    x = Embedding(vocab_size, d_model)(inputs)

    # simple positional encoding
    positions = tf.range(start=0, limit=max_len, delta=1)
    pos_emb = Embedding(max_len, d_model)(positions)
    x = x + pos_emb

    # single transformer block
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)
    ffn_output = Dense(dff, activation='relu')(x)
    ffn_output = Dense(d_model)(ffn_output)
    x = LayerNormalization(epsilon=1e-6)(x + ffn_output)

    outputs = Dense(vocab_size, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ---------- Part 6: Training loop ----------

def train_model(jsonl_path, vocab_path, model_path,
                max_len=128, batch_size=32, epochs=10):
    token2id = json.load(open(vocab_path))
    X, y = encode_dataset(jsonl_path, token2id, max_len=max_len)
    vocab_size = len(token2id)

    model = create_transformer_model(vocab_size, max_len)
    model.summary()

    model.fit(
        X, y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1
    )
    model.save(model_path)
    print(f"Model saved to {model_path}")

# ---------- Example Usage ----------

if __name__ == '__main__':
    # 1) generate dataset
    dataset_file = 'data/nqueens_6_traces.jsonl'
    vocab_file = 'data/nqueens_vocab.json'
    model_file = 'models/nqueens_transformer'

    stats = generate_dataset(dataset_file, n=6, trials=1000, randomize=True)
    token2id = build_vocab(dataset_file, vocab_file)
    train_model(dataset_file, vocab_file, model_file, epochs=20)
