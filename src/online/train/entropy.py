def _entropy_schedule_fn(settings):
    start = float(settings.get("entropy_coef_start", settings.get("entropy_coef", 0.0)))
    final = float(settings.get("entropy_coef_final", start))
    anneal_steps = int(settings.get("entropy_anneal_steps", 0))
    total = int(settings.get("total_timesteps", 1))
    if anneal_steps <= 0 or start == final:
        return start

    def schedule(progress_remaining: float) -> float:
        # progress_remaining goes 1 -> 0 over training
        current_step = (1.0 - progress_remaining) * total
        frac = min(max(current_step / max(anneal_steps, 1), 0.0), 1.0)
        return start + (final - start) * frac

    return schedule


__all__ = ["_entropy_schedule_fn"]
