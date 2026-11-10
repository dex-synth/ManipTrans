"""Opt-in finite checks for the H2R entrypoints; native defaults are unchanged."""

import torch


def normalize_valid(normalizer, values, valid):
    """Update moments using valid rows only, then normalize the fixed-size batch."""
    if int(valid.sum()) > 1:
        normalizer.train()
        normalizer(values[valid])
    normalizer.eval()
    return normalizer(values)


def require_finite(values, context):
    values = {name: value for name, value in values.items() if value is not None}
    if not values:
        return
    device = next(iter(values.values())).device
    finite = torch.stack([value.isfinite().all().to(device) for value in values.values()])
    if not finite.all():
        bad = [name for name, valid in zip(values, finite.tolist()) if not valid]
        raise FloatingPointError(f"Non-finite {context}: {', '.join(bad)}")


def check_gaussian(mu, logstd, value):
    require_finite(dict(mu=mu, logstd=logstd, value=value, sigma=logstd.exp()), "policy output")
    if not (logstd.exp() > 0).all():
        raise FloatingPointError("Zero policy standard deviation")


def check_model_result(module, args, output):
    require_finite({k: v for k, v in output.items() if isinstance(v, torch.Tensor)}, "policy result")


def enable_checks(model):
    for module in model.modules():
        module.check_finite = True
    if not getattr(model, "_finite_result_hook", False):
        model.register_forward_hook(check_model_result)
        model._finite_result_hook = True
    require_finite(model.state_dict(), "model state")
    return model
