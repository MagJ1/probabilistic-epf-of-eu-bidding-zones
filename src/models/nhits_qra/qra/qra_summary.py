from __future__ import annotations

def pack_summary(*,
                  split, device, Qte_per_h, mean_crps, mean_es,mean_ece,
                  berk, taus_train_used, taus_tgt, summary_params,
                  N, anchor_hour, origin_stride):
    """
    `berk` is a dict with keys:
      - "per_h": list of per-horizon dicts:
           {"h", "LR", "df", "p", "T_used", "mu", "rho", "sigma2"}
      - "fisher": {"stat", "df", "p", "k"}  (and optionally "hmp": float)
      - (optionally) "pooled_diagnostic": {...} if ever adding it
    """
    # Stable insertion order for readability
    out = {
        "run": {
            "split": str(split),
            "device": str(device),
        },
        "data": {
            "n_horizons": int(len(Qte_per_h)),
            "N_origins": int(N),
        },
        "filter": {
            "anchor_hour": (None if anchor_hour is None else int(anchor_hour)),
            "origin_stride": int(origin_stride),
        },
        "taus": {
            "train": [float(t) for t in taus_train_used],
            "target": [float(t) for t in taus_tgt],
        },
        "metrics": {
            "CRPS_mean": float(mean_crps),
            "ES_mean": float(mean_es),
            "ECE_mean": float(mean_ece),
        },
        "calibration": {
            "berkowitz": {
                # Per-horizon detailed results
                "per_h": [
                    {
                        "h": int(r.get("h")),
                        "LR": (None if r.get("LR") is None else float(r.get("LR"))),
                        "df": int(r.get("df", 3)),
                        "p": (None if r.get("p") is None else float(r.get("p"))),
                        "T_used": (None if r.get("T_used") is None else int(r.get("T_used"))),
                        "mu": (None if r.get("mu") is None else float(r.get("mu"))),
                        "rho": (None if r.get("rho") is None else float(r.get("rho"))),
                        "sigma2": (None if r.get("sigma2") is None else float(r.get("sigma2"))),
                    }
                    for r in berk.get("per_h", [])
                ],
                # Fisher combination across horizons
                "fisher": {
                    "stat": (None if berk.get("fisher", {}).get("stat") is None
                             else float(berk["fisher"]["stat"])),
                    "df": int(berk.get("fisher", {}).get("df", 0)),
                    "p": (None if berk.get("fisher", {}).get("p") is None
                          else float(berk["fisher"]["p"])),
                    "k": int(berk.get("fisher", {}).get("k", 0)),
                    # Include HMP if present
                    "hmp": (None if berk.get("fisher", {}).get("hmp") is None
                            else float(berk["fisher"]["hmp"])),
                    "note": "Fisher assumes independence; horizons are dependent -> interpret as approximate."
                },
            }
        },
        "model": {
            "qra_params": summary_params,
        },
    }

    if str(split) == "val":
        out["val_mean_crps"] = float(mean_crps)
        out["val_mean_ece"]  = float(mean_ece)

    # If later adding a pooled diagnostic block, keeping it optional:
    pooled = berk.get("pooled_diagnostic")
    if pooled is not None:
        out["calibration"]["berkowitz"]["pooled_diagnostic"] = {
            "LR": (None if pooled.get("LR") is None else float(pooled["LR"])),
            "p": (None if pooled.get("p") is None else float(pooled["p"])),
            "T_used": int(pooled.get("T_used", 0)),
            "note": str(pooled.get("note", "diagnostic only")),
        }

    return out