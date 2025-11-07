import torch
import os
import time
import numpy as np
import uuid
import pickle
import itertools
import sys
import json

import matplotlib.pyplot as plt
import ast
import re

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

WARNINGS = set()


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved {filename}')


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')


def freeze_module(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def ckpt_path(version, seed=None, glob=False):
    if glob:
        return "results/" + str(version) + '_*'

    if seed is None:
        try:
            # detect which seeds are present by looking for directories of form 'results/{version}_{seed}'
            seeds = [int(d.split('_')[-1])
                     for d in os.listdir('results')
                     if (d.startswith(str(version) + '_')
                         and os.path.isdir(os.path.join('results', d)))]
            if len(seeds) == 0:
                raise ValueError(f"No results found for version {version}")
            if len(seeds) > 1:
                print(f'Warning: multiple seeds found for version {version}: {seeds}. Using first seed.')

            seed = seeds[0]
            if seed != 0:
                print('Using seed:', seed)
        except FileNotFoundError:
            # we might be running from figures/
            try:
                # detect which seeds are present by looking for directories of form 'results/{version}_{seed}'
                seeds = [int(d.split('_')[-1])
                         for d in os.listdir('../results')
                         if (d.startswith(str(version) + '_')
                             and os.path.isdir(os.path.join('../results', d)))]
                if len(seeds) == 0:
                    raise ValueError(f"No results found for version {version}")
                if len(seeds) > 1:
                    print(f'Warning: multiple seeds found for version {version}: {seeds}. Using first seed.')

                seed = seeds[0]
                if seed != 0:
                    print('Using seed:', seed)

            except FileNotFoundError:
                print('Defaulting to seed=0')
                seed = 0

    return "results/" + str(version) + '_' + str(seed)


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def get_operators(operator_set: str):
    if operator_set == "arith":
        binary_operators = ["+", "-", "*"]
        unary_operators = []
    elif operator_set == "full":
        binary_operators = ["+", "-", "*", "/", "^"]
        unary_operators = ["sin", "cos", "exp", "log", "sqrt"]
    else:
        raise ValueError(f"Unknown operator set: {operator_set}")

    return binary_operators, unary_operators


def get_script_execution_command():
    return 'python ' + ' '.join(sys.argv)


def warn(s):
    if s not in WARNINGS:
        print('WARNING:', s)
    WARNINGS.add(s)


def hash_tensor(t):
    return (t * torch.arange(torch.numel(t)).reshape(t.shape)**2).sum() % 1000


_SEGMENT_RE = re.compile(r"([^\[\]]+)|(\[(\d+)\])")


def _coerce_value(raw):
    """Best-effort convert a string to Python type (int/float/bool/None/list/dict).
    Accepts lowercase true/false/null/none. Falls back to original string.
    """
    if isinstance(raw, (int, float, bool)) or raw is None:
        return raw
    if not isinstance(raw, str):
        return raw
    low = raw.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def _parse_keypath(path: str):
    """Parse a dot/bracket path like 'a.b[0].c' into ["a","b",0,"c"]."""
    parts = []
    for segment in path.split('.'):
        if not segment:
            continue
        for m in _SEGMENT_RE.finditer(segment):
            key, bracket, idx = m.groups()
            if key is not None:
                parts.append(key)
            elif idx is not None:
                parts.append(int(idx))
    return parts


def apply_override(cfg: dict, path: str, value):
    """Set cfg[path] = value creating intermediate dicts/lists as needed.
    Supports dict keys and list indices via [N].
    """
    tokens = _parse_keypath(path)
    if not tokens:
        return
    cur = cfg
    for i, tok in enumerate(tokens):
        is_last = i == len(tokens) - 1
        if is_last:
            if isinstance(tok, int):
                if not isinstance(cur, list):
                    # Replace non-list with a new list
                    cur_list = []
                    # cannot assign back to parent here without full path context; assume list exists
                    cur = cur_list
                if tok >= len(cur):
                    cur.extend([None] * (tok - len(cur) + 1))
                cur[tok] = value
            else:
                if isinstance(cur, dict):
                    cur[tok] = value
                else:
                    raise TypeError(f"Cannot set key '{tok}' on non-dict container at path '{path}'")
        else:
            nxt = tokens[i + 1]
            want_list = isinstance(nxt, int)
            if isinstance(tok, int):
                if not isinstance(cur, list):
                    cur = []
                if tok >= len(cur):
                    cur.extend([None] * (tok - len(cur) + 1))
                if cur[tok] is None or (want_list and not isinstance(cur[tok], list)) or (not want_list and not isinstance(cur[tok], dict)):
                    cur[tok] = [] if want_list else {}
                cur = cur[tok]
            else:
                if tok not in cur or cur[tok] is None or (want_list and not isinstance(cur[tok], list)) or (not want_list and not isinstance(cur[tok], dict)):
                    cur[tok] = [] if want_list else {}
                cur = cur[tok]


def parse_overrides(overrides):
    """Parse a list of 'key=val' strings into (path, value) pairs with coerced values."""
    parsed = []
    for ov in overrides or []:
        if '=' not in ov:
            raise ValueError(f"Override must be in key=val format: {ov}")
        key, val = ov.split('=', 1)
        parsed.append((key.strip(), _coerce_value(val.strip())))
    return parsed


def assert_equal(*args):
    # iterate through adjacent pairs and check for equality
    for a, b in zip(args[:-1], args[1:]):
        if np.ndarray in [type(a), type(b)]:
            assert np.array_equal(a, b), f'a != b: a:{a}, b:{b}'
        elif torch.Tensor in [type(a), type(b)]:
            assert torch.equal(a, b), f'a != b: a:{a}, b:{b}'
        else:
            assert a == b, f'a != b: a:{a}, b:{b}'


def num_params(model):
    return sum([torch.prod(torch.tensor(p.shape))
                for p in list(model.parameters())])


def save_model(model, path, overwrite=False):
    if not overwrite:
        path = next_unused_path(path)
    torch.save(model, path)
    print('Saved model at ' + path)
    return path


def load_model(path):
    model = torch.load(path, map_location=DEVICE)
    print('Loaded model from ' + path)
    return model


def generate_uuid():
    return uuid.uuid4().hex


def next_unused_path(path, extend_fn=lambda i: f'__({i})'):
    last_dot = path.rindex('.')
    extension = path[last_dot:]
    file_name = path[:last_dot]

    i = 0
    while os.path.isfile(path):
        path = file_name + extend_fn(i) + extension
        i += 1

    return path


class Timing(object):
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        dt = time.time() - self.start
        if isinstance(self.message, str):
            message = self.message
        elif callable(self.message):
            message = self.message(dt)
        else:
            raise ValueError("Timing message should be string function")
        print(f"{message} in {dt:.1f} seconds")


def print_structure(obj, indent=0, max_list_items=3, max_dict_items=10):
  """Print the structure of a nested Python object (dict/list/etc)."""
  prefix = "  " * indent

  if isinstance(obj, dict):
      print(f"{prefix}dict {{")
      items = list(obj.items())[:max_dict_items]
      for key, value in items:
          print(f"{prefix}  '{key}': ", end="")
          if isinstance(value, (dict, list)):
              print()
              print_structure(value, indent + 2, max_list_items, max_dict_items)
          else:
              print(f"{type(value).__name__}", end="")
              if hasattr(value, 'shape'):  # numpy arrays
                  print(f" shape={value.shape} dtype={value.dtype}")
              elif isinstance(value, (str, int, float, bool)):
                  print(f" = {repr(value)[:50]}")
              else:
                  print()
      if len(obj) > max_dict_items:
          print(f"{prefix}  ... ({len(obj) - max_dict_items} more keys)")
      print(f"{prefix}}}")

  elif isinstance(obj, list):
      print(f"{prefix}list(length={len(obj)}) [")
      if len(obj) > 0:
          # Show first few items
          for i, item in enumerate(obj[:max_list_items]):
              print(f"{prefix}  [{i}]: ", end="")
              if isinstance(item, (dict, list)):
                  print()
                  print_structure(item, indent + 2, max_list_items, max_dict_items)
              else:
                  print(f"{type(item).__name__}", end="")
                  if hasattr(item, 'shape'):
                      print(f" shape={item.shape} dtype={item.dtype}")
                  elif isinstance(item, (str, int, float, bool)):
                      print(f" = {repr(item)[:50]}")
                  else:
                      print()
          if len(obj) > max_list_items:
              print(f"{prefix}  ... ({len(obj) - max_list_items} more items)")
      print(f"{prefix}]")

  else:
      print(f"{prefix}{type(obj).__name__}", end="")
      if hasattr(obj, 'shape'):
          print(f" shape={obj.shape} dtype={obj.dtype}")
      else:
          print()


def resolve_model_dir(path: str) -> str:
    """Return a directory that contains an HF model config.json.

    Accepts either a leaf model dir (itself has config.json), a parent directory
    that contains a "final_model" subdir, or scans subdirectories for a config.json.
    """
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    final = os.path.join(path, "final_model")
    if os.path.isfile(os.path.join(final, "config.json")):
        return final
    candidates = []
    try:
        for name in sorted(os.listdir(path)):
            sub = os.path.join(path, name)
            if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "config.json")):
                has_weights = (
                    os.path.isfile(os.path.join(sub, "pytorch_model.bin")) or
                    os.path.isfile(os.path.join(sub, "model.safetensors"))
                )
                # Prefer final_model, then dirs that have weights
                rank = 0 if name == "final_model" else (1 if has_weights else 2)
                candidates.append((rank, sub))
    except Exception:
        pass
    if candidates:
        candidates.sort()
        return candidates[0][1]
    return path


if __name__ == '__main__':
    c = torch.tensor(float('-inf'))
    print(logaddexp(c, c))
