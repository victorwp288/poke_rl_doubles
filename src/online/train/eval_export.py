import csv
import json
import pathlib

import numpy as np


def _export_eval_metrics(best_dir, settings):
    eval_file = pathlib.Path(best_dir) / "evaluations.npz"
    if not eval_file.exists():
        print("[eval export] evaluations.npz not found", flush=True)
        return
    data = np.load(eval_file, allow_pickle=True)
    out = pathlib.Path("outputs/eval")
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "ppo_eval.csv"
    jsonl = out / "ppo_eval.jsonl"
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as cf, open(jsonl, "a") as jf:
        w = csv.writer(cf)
        if not exists:
            w.writerow(
                [
                    "settings_id",
                    "eval_index",
                    "timesteps",
                    "mean_reward",
                    "reward_std",
                    "mean_ep_length",
                ]
            )
        sid = str(settings.get("policy_path", "unknown"))
        for i, ts in enumerate(data["timesteps"]):
            rewards = data["results"][i]
            ep_lengths = data["ep_lengths"][i]
            mr = float(rewards.mean())
            sr = float(rewards.std())
            ml = float(ep_lengths.mean())
            w.writerow([sid, i, int(ts), mr, sr, ml])
            jf.write(
                json.dumps(
                    {
                        "settings_id": sid,
                        "eval_index": i,
                        "timesteps": int(ts),
                        "mean_reward": mr,
                        "reward_std": sr,
                        "mean_ep_length": ml,
                    }
                )
                + "\n"
            )


__all__ = ["_export_eval_metrics"]
