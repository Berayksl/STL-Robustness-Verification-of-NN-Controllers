import functools
import jax.numpy as jnp
import torch.nn as nn
import jax_verify
import numpy as np
 
def pytorch_policy_to_jax(torch_model: nn.Module):
    params = []
 
    # Hidden layers
    for layer_name in ["linear1", "linear2", "linear3", "linear4"]:
        layer = getattr(torch_model, layer_name)
        W = layer.weight.detach().cpu().numpy().T   # transpose for JAX
        b = layer.bias.detach().cpu().numpy()
        params.append((W, b))
 
    # Output layers
    mean_layer = torch_model.mean_linear
    mean_W = mean_layer.weight.detach().cpu().numpy().T
    mean_b = mean_layer.bias.detach().cpu().numpy()
    params.append((mean_W, mean_b))

    if hasattr(torch_model, 'log_std_linear'):
        log_std_layer = torch_model.log_std_linear
        log_W = log_std_layer.weight.detach().cpu().numpy().T
        log_b = log_std_layer.bias.detach().cpu().numpy()
        params.append((log_W, log_b))

    return functools.partial(relu_policy_nn, params)
 
 
def relu_policy_nn(params, inputs):
    """
    params: list of (W, b)
      [linear1, linear2, linear3, linear4, mean_linear, (optional) log_std_linear]
    inputs: jnp.array of shape (batch, 3)
    """
    x = inputs
    for W, b in params[:4]:
        x = jnp.dot(x, W) + b
        x = jnp.maximum(x, 0)

    mean_W, mean_b = params[4]
    mean = jnp.dot(x, mean_W) + mean_b

    if len(params) > 5:
        log_W, log_b = params[5]
        log_std = jnp.dot(x, log_W) + log_b
    else:
        log_std = None

    return mean, log_std



def jax_interval_to_np_range(interval: jax_verify.IntervalBound) -> np.ndarray:
  return np.vstack([interval.lower, interval.upper]).T