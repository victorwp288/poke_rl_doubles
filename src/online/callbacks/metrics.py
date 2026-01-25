import time
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback


class RewardMetricsCallback(BaseCallback):
    """Record reward-related stats to the logger and emit concise console summaries.

    This inspects `infos` from the rollout buffer, which is robust for both DummyVecEnv
    and SubprocVecEnv.
    """

    def __init__(self, summary_interval_sec: float = 5.0, summary_window: int = 50):
        super().__init__()
        self.summary_interval_sec = float(summary_interval_sec)
        self.summary_window = max(int(summary_window), 1)
        self._last_summary_time: float | None = None
        self._recent: deque[dict] = deque(maxlen=self.summary_window)
        self._best_mean_score: float | None = None

    def _record_from_infos(self, infos):
        if not infos:
            return
        repaired_count = 0
        for info in infos:
            if not isinstance(info, dict):
                continue
            if info.get("repaired_action"):
                repaired_count += 1
            stats = info.get("battle_stats")
            if not isinstance(stats, dict):
                continue
            for key, value in stats.items():
                try:
                    v = float(value)
                except Exception:
                    continue
                self.logger.record(f"reward_components/{key}", v)

            result = stats.get("result")
            if result not in {"win", "loss", "draw"}:
                continue
            entry = {
                "score": float(stats.get("score", 0.0)),
                "delta": float(stats.get("delta", 0.0)),
                "result": result,
                "turn": int(stats.get("turn", 0)),
            }
            self._recent.append(entry)
        if infos:
            rate = repaired_count / max(len(infos), 1)
            self.logger.record("safety/repaired_rate", rate)
            if self.logger:
                self.logger.record("safety/repaired_count", repaired_count)

    def _summary_from_recent(self):
        if not self._recent:
            return {}
        n = len(self._recent)
        mean_score = sum(e["score"] for e in self._recent) / n
        mean_delta = sum(e["delta"] for e in self._recent) / n
        wins = sum(1 for e in self._recent if e.get("result") == "win")
        losses = sum(1 for e in self._recent if e.get("result") == "loss")
        draws = sum(1 for e in self._recent if e.get("result") == "draw")
        total_results = wins + losses + draws
        win_rate = wins / total_results if total_results else 0.0
        mean_turn = sum(e["turn"] for e in self._recent) / n
        last = self._recent[-1]
        return {
            "n": n,
            "mean_score": mean_score,
            "mean_delta": mean_delta,
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "mean_turn": mean_turn,
            "last_score": last.get("score", 0.0),
            "last_delta": last.get("delta", 0.0),
            "last_result": last.get("result", "-") or "-",
            "last_turn": last.get("turn", 0),
        }

    def _maybe_print_summary(self):
        if self.summary_interval_sec <= 0.0:
            return
        now = time.time()
        if (
            self._last_summary_time is not None
            and now - self._last_summary_time < self.summary_interval_sec
        ):
            return
        self._last_summary_time = now

        summary = self._summary_from_recent()
        if not summary:
            return

        mean_score = summary["mean_score"]
        if self._best_mean_score is None or mean_score > self._best_mean_score:
            self._best_mean_score = mean_score
            best_flag = " *best*"
        else:
            best_flag = ""

        t = int(self.num_timesteps)
        print(
            f"[online train]{best_flag} t={t} n={summary['n']} "
            f"mean_score={mean_score:.3f} win_rate={summary['win_rate']:.3f} "
            f"mean_delta={summary['mean_delta']:.3f} mean_turn={summary['mean_turn']:.1f} "
            f"last_score={summary['last_score']:.3f} last_result={summary['last_result']} "
            f"last_turn={summary['last_turn']}",
            flush=True,
        )

    def _on_step(self):
        infos = self.locals.get("infos") or []
        self._record_from_infos(infos)
        self._maybe_print_summary()
        return True


__all__ = ["RewardMetricsCallback"]
