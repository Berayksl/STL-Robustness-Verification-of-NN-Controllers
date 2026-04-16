import functools
import jax.numpy as jnp
import torch.nn as nn
 
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
    log_std_layer = torch_model.log_std_linear
 
    mean_W = mean_layer.weight.detach().cpu().numpy().T
    mean_b = mean_layer.bias.detach().cpu().numpy()
    params.append((mean_W, mean_b))
 
    log_W = log_std_layer.weight.detach().cpu().numpy().T
    log_b = log_std_layer.bias.detach().cpu().numpy()
    params.append((log_W, log_b))
 
    return functools.partial(relu_policy_nn, params)
 
 
def relu_policy_nn(params, inputs):
    """
    params: list of (W, b)
      [linear1, linear2, linear3, linear4, mean_linear, log_std_linear]
    inputs: jnp.array of shape (batch, 3)
    """
    x = inputs
    # Hidden layers
    for W, b in params[:4]:
        x = jnp.dot(x, W) + b
        x = jnp.maximum(x, 0)
 
    # Outputs
    mean_W, mean_b = params[4]
    log_W, log_b   = params[5]
 
    mean    = jnp.dot(x, mean_W) + mean_b
    log_std = jnp.dot(x, log_W) + log_b
 
    # # Optional: clamp log_std like in PyTorch
    # log_std = jnp.clip(log_std, a_min=torch_model.log_std_min, a_max=torch_model.log_std_max)
    return mean, log_std