import torch

from .utils import _slot_loss


def _train_epoch(
    model,
    loader,
    device,
    loss_fn,
    optimizer,
    log_every,
    grad_clip_norm,
    obs_noise_std,
    numeric_feature_mask,
    scheduler,
    grad_accum_steps,
    use_amp,
    amp_dtype,
):
    model.train()
    total_loss = 0.0
    total_items = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    steps = 0
    total_steps = len(loader)
    remainder = total_steps % grad_accum_steps
    last_window = remainder if remainder > 0 else grad_accum_steps
    last_window_start = total_steps - last_window + 1

    for step, (obs, actions, mask, weights) in enumerate(loader, 1):
        steps = step
        obs = obs.to(device)
        actions = actions.to(device)
        mask = mask.to(device)
        if weights is not None:
            weights = weights.to(device)
        if obs_noise_std:
            noise = torch.randn_like(obs) * obs_noise_std
            if numeric_feature_mask is not None:
                noise = noise * numeric_feature_mask
            inputs = obs + noise
        else:
            inputs = obs
        is_step = step % grad_accum_steps == 0
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            loss = _slot_loss(model(inputs), actions, mask, loss_fn, sample_weights=weights)
            # Use the actual accumulation window size for the final partial window
            accum_divisor = (
                last_window if (remainder > 0 and step >= last_window_start) else grad_accum_steps
            )
            loss = loss / accum_divisor

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if is_step:
            if grad_clip_norm:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * accum_divisor * obs.size(0)
        total_items += obs.size(0)
        if log_every and step % log_every == 0:
            print(f"step {step}: loss {loss.item() * accum_divisor:.4f}")
    remainder = steps % grad_accum_steps
    if remainder:
        if grad_clip_norm:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
    return total_loss / max(1, total_items)


def _evaluate(model, loader, device, loss_fn, use_amp, amp_dtype):
    if loader is None:
        return None
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for obs, actions, mask, weights in loader:
            obs = obs.to(device)
            actions = actions.to(device)
            mask = mask.to(device)
            if weights is not None:
                weights = weights.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                loss = _slot_loss(model(obs), actions, mask, loss_fn, sample_weights=weights)
            total_loss += loss.item() * obs.size(0)
            total_items += obs.size(0)
    if total_items == 0:
        return None
    return total_loss / total_items


__all__ = ["_evaluate", "_train_epoch"]
