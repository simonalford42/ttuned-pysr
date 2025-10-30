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


# batched covariance calculation:
# https://stackoverflow.com/a/71357620/4383594
def batch_cov(points):
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)


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

def gpu_check():
    print_torch_device()
    torch.arange(3).to(DEVICE)


def warn(s):
    if s not in WARNINGS:
        print('WARNING:', s)
    WARNINGS.add(s)


def hash_tensor(t):
    return (t * torch.arange(torch.numel(t)).reshape(t.shape)**2).sum() % 1000


class CustomDictOne(dict):
    def __init__(self,*arg,**kw):
        super(CustomDictOne, self).__init__(*arg, **kw)


def log(s: str):
    with open('log.txt', 'r+') as f:
        f.write(s)


def print_torch_device():
    if torch.cuda.is_available():
        print('Using torch device ' + torch.cuda.get_device_name(DEVICE))
    else:
        print('Using torch device CPU')


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


def logaddexp(tensor, other, mask=[1, 1]):
    if type(mask) in [list, tuple]:
        mask = torch.tensor(mask)

    assert mask.shape == (2, ), 'Invalid mask shape'

    a = torch.max(tensor, other)
    # if max is -inf, set a to zero, to avoid making nan's
    a = torch.where(a == float('-inf'), torch.zeros(a.shape), a)

    return a + ((tensor - a).exp()*mask[0] + (other - a).exp()*mask[1]).log()


def log1minus(x):
    """
    Returns log(1 - x.exp())
    This is the logged version of finding 1 - p
    """
    return torch.log1p(-x.exp())


def compare_tensors(t1, t2):
    # (a, b, c, d), (a, b, c, d) -> (a, b, c, 2d)
    plot_tensor(torch.cat((t1, t2), dim=-1))


def plot_tensor(t):
    if t.dim() == 2:
        plot_2D_tensor(t)
    else:
        init_shape = t.shape[:-2]
        for init_dim_values in itertools.product(*map(range, init_shape)):
            plot_2D_tensor(t[init_dim_values], label=init_dim_values)


def plot_2D_tensor(t, label=None):
    (y, x) = t.shape
    fig, ax = plt.subplots()
    ax.imshow(t)

    print(t)
    for j in range(y):
        for i in range(x):
            ax.text(i, j, f'{t[j][i].item():.2f}', ha="center", va="center", color="w", fontsize=6)

    if label is not None:
        plt.title(label)
    plt.show()


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


if __name__ == '__main__':
    c = torch.tensor(float('-inf'))
    print(logaddexp(c, c))
