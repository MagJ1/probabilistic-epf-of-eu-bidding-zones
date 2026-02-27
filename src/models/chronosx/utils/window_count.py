import os
import math
import pandas as pd

def _get_world_size() -> int:
    # works for accelerate/torchrun; safe fallback
    for k in ("WORLD_SIZE", "SLURM_NTASKS"):
        if k in os.environ:
            try: return int(os.environ[k])
            except: pass
    return 1

def estimate_total_windows(entries, context_len, pred_len, stride_hours, anchor_hour):
    total = 0
    for ts in entries:
        T = int(len(ts["target"]))
        t_min = int(context_len)
        t_max = int(T - pred_len)
        if t_max < t_min:
            continue

        step = int(stride_hours)
        if anchor_hour is None:
            first_t = t_min
        else:
            start_period = ts["start"]
            first_t = None
            for t in range(t_min, min(t_min + 24, t_max + 1)):
                try:
                    if (start_period + t).hour == int(anchor_hour):
                        first_t = t
                        break
                except Exception:
                    if (start_period.to_timestamp() + pd.Timedelta(hours=t)).hour == int(anchor_hour):
                        first_t = t
                        break
            if first_t is None:
                continue

            # optional: if anchor is set, enforce at least 24 stride semantics
            if step < 24:
                step = 24

        total += (t_max - first_t) // step + 1
    return int(total)

def estimate_total_windows_from_train_list(train_list, context_len, pred_len, stride, anchor_hour):
    # train_list is list[list[dict]] (sources)
    return sum(
        estimate_total_windows(src_entries, context_len, pred_len, stride, anchor_hour)
        for src_entries in train_list
    )

def compute_steps_per_epoch(total_windows, per_device_bs, grad_accum, world_size):
    eff_bs = int(per_device_bs) * int(grad_accum) * int(world_size)
    return int(math.ceil(total_windows / eff_bs))