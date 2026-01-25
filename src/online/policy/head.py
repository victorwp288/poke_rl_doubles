from torch import nn

_DEFAULT_HEAD_HIDDEN = 512


def _policy_device(policy):
    return next(policy.parameters()).device


def _action_linear_head(policy):
    action_net = policy.action_net
    if isinstance(action_net, nn.Linear):
        return action_net
    if isinstance(action_net, nn.Sequential) and action_net:
        for module in reversed(action_net):
            if isinstance(module, nn.Linear):
                return module
    return action_net


def configure_action_head(policy, head_hidden_dim):
    """Replace the SB3 action_net with a two-layer MLP head while keeping output dims the same."""
    action_dims = getattr(policy.action_dist, "action_dims", [])
    if not action_dims:
        return
    total_actions = sum(action_dims)
    current = _action_linear_head(policy)
    if not isinstance(current, nn.Linear):
        return
    in_features = current.in_features
    head_hidden_dim = int(head_hidden_dim)
    if head_hidden_dim <= 0:
        return
    device = _policy_device(policy)
    new_head = nn.Sequential(
        nn.Linear(in_features, head_hidden_dim * 2),
        nn.GELU(),
        nn.Linear(head_hidden_dim * 2, total_actions),
    ).to(device)
    policy.action_net = new_head


def _infer_action_head_hidden_dim_from_params(params):
    policy_params = params.get("policy", {}) if isinstance(params, dict) else {}
    if isinstance(policy_params, dict):
        weight = policy_params.get("action_net.0.weight")
        if weight is not None and weight.ndim == 2 and weight.shape[0] % 2 == 0:
            return int(weight.shape[0] // 2)
    return None


__all__ = [
    "_DEFAULT_HEAD_HIDDEN",
    "_action_linear_head",
    "_infer_action_head_hidden_dim_from_params",
    "_policy_device",
    "configure_action_head",
]
