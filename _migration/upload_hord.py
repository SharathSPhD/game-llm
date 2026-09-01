#!/usr/bin/env python3
"""Publish prabhasa-samskrutam HORD m2/m3 checkpoints to public HF repos.

Uploads final.pt + anchor_*.pt (skips redundant latest.pt) for the real
trained arms only (m2 199M/650M-tok, m3 353M/275M-tok). One repo per scale:
  qbz506/p-s-hord-m2   (baseline + treatment)
  qbz506/p-s-hord-m3   (baseline + treatment + aux)
Each .pt self-describes (embeds config/arm/step/tokens_seen + optimizer state),
so it is resumable/reproducible with the prabhasa codebase on GitHub.
"""
import os, glob, torch, json
from huggingface_hub import HfApi, create_repo

CKPT = "/home/sharaths/projects/prabhasa-samskrutam/data/checkpoints"
USER = "qbz506"
REPOS = {
    "m2": {"repo": f"{USER}/p-s-hord-m2", "arms": ["baseline", "treatment"]},
    "m3": {"repo": f"{USER}/p-s-hord-m3", "arms": ["baseline", "treatment", "aux"]},
}
api = HfApi()


def card(m, meta):
    return f"""---
license: apache-2.0
tags:
- prabhasa
- hord
- sanskrit
- paninian
- mamba
- ablation
- research-checkpoint
language:
- sa
- en
---

# prabhasa-samskrutam — HORD {m}

Research training checkpoints from the **prabhasa-samskrutam** HORD series
(Pāṇinian-structured hybrid Mamba/attention language models). Published under
the pseudonym **p-s**. These are **raw PyTorch training checkpoints**, not
packaged inference models.

## Arms (baseline vs treatment ablation)

{meta}

## What each `.pt` contains

Each checkpoint is a dict with keys: `model` (state_dict), `opt` (optimizer
state), `config` (architecture), `arm`, `step`, `tokens_seen`, `cursor`. It is
fully **resumable**.

```python
import torch
ckpt = torch.load("treatment/final.pt", map_location="cpu", weights_only=False)
print(ckpt["config"], ckpt["step"], ckpt["tokens_seen"])
# model = build_model(**ckpt["config"]); model.load_state_dict(ckpt["model"])
```

## Reproduce / load

Architecture and tokenizer live in the code repo:
<https://github.com/SharathSPhD/prabhasa-samskrutam>
(`src/prabhasa/` for the model; tokenizer under
`src/prabhasa/application/tokenizer`). `anchor_*.pt` files are the MMD magnetic
anchor snapshots referenced during training.

*Byte-level tokenizer (vocab_size 256), structured channels (n_roles 16).*
"""


def main():
    for m, spec in REPOS.items():
        repo = spec["repo"]
        print(f"\n=== {repo} ===")
        # gather files + build per-arm metadata
        rows, files = [], []
        for arm in spec["arms"]:
            adir = os.path.join(CKPT, m, arm)
            fin = os.path.join(adir, "final.pt")
            if not os.path.exists(fin):
                print(f"  SKIP {arm}: no final.pt"); continue
            d = torch.load(fin, map_location="cpu", weights_only=False)
            c = d.get("config", {})
            npar = sum(v.numel() for v in d["model"].values() if hasattr(v, "numel")) // 1_000_000
            rows.append(f"- **{arm}**: {npar}M params, step {d.get('step')}, "
                        f"{d.get('tokens_seen',0)//1_000_000}M tokens, "
                        f"d_model={c.get('d_model')}, n_layers={c.get('n_layers')}")
            for f in glob.glob(os.path.join(adir, "final.pt")) + glob.glob(os.path.join(adir, "anchor_*.pt")):
                files.append((f, f"{arm}/{os.path.basename(f)}"))
        if not files:
            print("  nothing to upload"); continue
        create_repo(repo, repo_type="model", private=False, exist_ok=True)
        api.upload_file(path_or_fileobj=card(m, "\n".join(rows)).encode(),
                        path_in_repo="README.md", repo_id=repo, repo_type="model")
        for local, remote in files:
            sz = os.path.getsize(local) / 1073741824
            print(f"  uploading {remote} ({sz:.1f}G) ...", flush=True)
            api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                            repo_id=repo, repo_type="model")
        print(f"  DONE https://hf.co/{repo}")


if __name__ == "__main__":
    main()
