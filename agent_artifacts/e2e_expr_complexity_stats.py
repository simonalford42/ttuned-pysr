#!/usr/bin/env python3
"""
Compute expression complexity stats (depth and size) from e2e-generated trees.

Usage:
  python e2e_expr_complexity_stats.py --n 1000 --seed 0 \
      --env "use_controller=False,env_base_seed=0"

Outputs summary statistics and suggested BasicSR params (max_depth, max_size).
"""
import argparse
import os
import sys
from typing import Dict, Any, List, Union

import numpy as np


def _ensure_e2e_on_path():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    e2e_dir = os.path.join(repo_dir, "e2e_sr")
    if e2e_dir not in sys.path:
        sys.path.insert(0, e2e_dir)


_ensure_e2e_on_path()

# Import from main codebase (parent directory)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from generate_expressions import E2EDataGenerator  # noqa: E402
from symbolicregression.envs.generators import Node, NodeList  # noqa: E402


def parse_env_overrides(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    out: Dict[str, Any] = {}
    for part in s.split(","):
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid env override '{part}', expected key=value")
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Basic coercions: int, float, bool, str
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                if "." in v or "e" in v.lower():
                    out[k] = float(v)
                else:
                    out[k] = int(v)
            except ValueError:
                out[k] = v
    return out


def node_depth(n: Node) -> int:
    if not n.children:
        return 1
    return 1 + max(node_depth(c) for c in n.children)


def per_output_stats(tree: Union[Node, NodeList]) -> List[Dict[str, int]]:
    if isinstance(tree, NodeList):
        nodes = tree.nodes
    else:
        nodes = [tree]
    stats: List[Dict[str, int]] = []
    for t in nodes:
        stats.append({
            "size": len(t),
            "depth": node_depth(t),
        })
    return stats


def summarize(values: List[int]) -> Dict[str, float]:
    arr = np.array(values)
    return {
        "min": int(arr.min()) if arr.size else 0,
        "p50": float(np.percentile(arr, 50)) if arr.size else 0.0,
        "p90": float(np.percentile(arr, 90)) if arr.size else 0.0,
        "p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "max": int(arr.max()) if arr.size else 0,
        "mean": float(arr.mean()) if arr.size else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="Compute e2e expr depth/size stats")
    ap.add_argument("--n", type=int, default=10000, help="Number of expressions to sample")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for env RNG")
    ap.add_argument("--env", type=str, default="use_controller=False", help="Comma-separated env overrides key=value")
    args = ap.parse_args()

    env_overrides = parse_env_overrides(args.env)
    env_overrides.setdefault("env_base_seed", args.seed)

    gen = E2EDataGenerator(env_overrides=env_overrides, seed=args.seed)

    sizes: List[int] = []
    depths: List[int] = []
    per_expr_max_sizes: List[int] = []
    per_expr_max_depths: List[int] = []

    for i in range(args.n):
        sample = gen.generate_expression(train=True)
        tree = sample["tree"]
        stats = per_output_stats(tree)
        # accumulate per-output stats
        for s in stats:
            sizes.append(s["size"])
            depths.append(s["depth"])
        # per-target (NodeList) maxima
        per_expr_max_sizes.append(max(s["size"] for s in stats))
        per_expr_max_depths.append(max(s["depth"] for s in stats))

    size_summary = summarize(sizes)
    depth_summary = summarize(depths)
    per_expr_size_summary = summarize(per_expr_max_sizes)
    per_expr_depth_summary = summarize(per_expr_max_depths)

    print("=== e2e Expression Complexity (per-output) ===")
    print(f"size  -> min={size_summary['min']} mean={size_summary['mean']:.2f} p90={size_summary['p90']:.1f} p95={size_summary['p95']:.1f} max={size_summary['max']}")
    print(f"depth -> min={depth_summary['min']} mean={depth_summary['mean']:.2f} p90={depth_summary['p90']:.1f} p95={depth_summary['p95']:.1f} max={depth_summary['max']}")

    print("\n=== Target-Level Maxima (NodeList aggregated) ===")
    print(f"size  -> p90={per_expr_size_summary['p90']:.1f} p95={per_expr_size_summary['p95']:.1f} max={per_expr_size_summary['max']}")
    print(f"depth -> p90={per_expr_depth_summary['p90']:.1f} p95={per_expr_depth_summary['p95']:.1f} max={per_expr_depth_summary['max']}")

    # Suggested BasicSR settings (conservative: p95 of per-target maxima)
    suggested_max_size = int(np.ceil(per_expr_size_summary["p95"]))
    suggested_max_depth = int(np.ceil(per_expr_depth_summary["p95"]))
    print("\n=== Suggested BasicSR Params ===")
    print(f"max_size={suggested_max_size}")
    print(f"max_depth={suggested_max_depth}")


if __name__ == "__main__":
    main()
