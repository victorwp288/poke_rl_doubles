import contextlib
import shutil
from collections import deque
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class RollingCheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_freq,
        save_path,
        name_prefix,
        keep_last=1,
        fixed_filename=None,
    ):
        super().__init__()
        self.save_freq = max(int(save_freq), 0)
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.keep_last = max(int(keep_last), 1)
        self.fixed_filename = fixed_filename
        self._next_save = 0
        self._saved_paths: deque[Path] = deque()

    def _on_training_start(self) -> None:
        if self.save_freq == 0:
            return
        self.save_path.mkdir(parents=True, exist_ok=True)
        self._next_save = self.num_timesteps + self.save_freq

    def _remove_excess(self) -> None:
        while len(self._saved_paths) > self.keep_last:
            old_path = self._saved_paths.popleft()
            with contextlib.suppress(FileNotFoundError):
                old_path.unlink()

    def _save_checkpoint(self) -> None:
        filename = f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
        target = self.save_path / filename
        self.model.save(str(target))
        self._saved_paths.append(target)
        self._remove_excess()
        if self.fixed_filename:
            latest_path = self.save_path / self.fixed_filename
            shutil.copy(str(target), str(latest_path))

    def _on_step(self) -> bool:
        if self.save_freq == 0:
            return True
        if self.num_timesteps >= self._next_save:
            self._save_checkpoint()
            self._next_save += self.save_freq
        return True


class CopyBestModelCallback(BaseCallback):
    def __init__(self, source_dir: Path, destination: Path):
        super().__init__()
        self.source = Path(source_dir) / "best_model.zip"
        self.destination = Path(destination)
        self._last_copied_stamp: int | None = None

    def _init_callback(self) -> None:
        self.destination.parent.mkdir(parents=True, exist_ok=True)

    def _copy_if_updated(self) -> None:
        if not self.source.exists():
            return
        stamp = self.source.stat().st_mtime_ns
        if stamp == self._last_copied_stamp:
            return
        shutil.copy(self.source, self.destination)
        self._last_copied_stamp = stamp
        print(f"[online eval] new best model copied to {self.destination}", flush=True)

    def _on_step(self) -> bool:
        self._copy_if_updated()
        return True


__all__ = ["CopyBestModelCallback", "RollingCheckpointCallback"]
