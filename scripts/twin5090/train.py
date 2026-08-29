"""Remote dispatcher for SPEC 0022 jobs on the RTX 5090.

The submit convention on the target launches `python train.py` detached in the
job directory, so this file is the whole remote control surface: it reads
job.json, which holds a list of stages (each a full exp39 argv), and runs them
sequentially in one process. Phase 1 is therefore a single detached job — Arm
E to 2.5B tokens, then Arm T to 2.5B tokens — that survives the submitting
session, honours the one-training-job rule by construction, and stops at the
first stage that exits non-zero so a failed arm can never silently hand a
corrupted state to the next.

The allocator setting matches the scale-run practice recorded in SPEC 0022;
it must be set before torch initialises CUDA, which is why it happens here
rather than in the trainer.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main() -> int:
    with open("job.json") as fh:
        job = json.load(fh)
    for i, stage_args in enumerate(job["stages"]):
        print(f"=== stage {i + 1}/{len(job['stages'])}: {stage_args}", flush=True)
        t0 = time.time()
        sys.argv = ["exp39"] + list(stage_args)
        try:
            runpy.run_path("experiments/exp39_twin_1b.py", run_name="__main__")
            rc = 0
        except SystemExit as exc:
            rc = int(exc.code or 0)
        print(f"=== stage {i + 1} exit {rc} after "
              f"{(time.time() - t0) / 3600:.2f}h", flush=True)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
