import contextlib


def _safe_close(env):
    if env is None:
        return
    with contextlib.suppress(AssertionError, RuntimeError):
        env.close()


__all__ = ["_safe_close"]
