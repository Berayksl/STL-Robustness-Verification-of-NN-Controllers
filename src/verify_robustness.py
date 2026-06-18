import time

import jax_verify
from matplotlib.patches import Rectangle
import functools
from JAX_scripts.torch_to_jax import pytorch_policy_to_jax, relu_policy_nn, jax_interval_to_np_range
from JAX_scripts.JAXController import JAXPolicyController
from networks import PolicyNetwork

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import tkinter as tk
from tkinter import filedialog
import os
import jax
import jax.numpy as jnp
from collections import OrderedDict
import ast
import yaml

import nfl_veripy.analyzers as analyzers  # noqa: E402
import nfl_veripy.constraints as constraints  # noqa: E402
import nfl_veripy.dynamics as dynamics
from nfl_veripy.utils.utils import get_plot_filename

def select_model_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

    file_path = filedialog.askopenfilename(
        title="Select Configuration File",
        initialdir=model_dir,
        filetypes=[("Model Files", "*.pth"), ("All Files", "*.*")])
    return file_path

def step_1_batch(states: jnp.ndarray, jax_controller) -> jnp.ndarray:
    actions = jax.vmap(jax_controller)(states)
    next_states = jax.vmap(dyn.dynamics_step_jnp)(states, actions)
    return next_states

def plot_Tstep_samples_and_bounds(initial_state_range,
    step_fn, num_steps, bounds):
    num_samples = 1000

    # Sample initial states
    xt = np.stack([
        np.random.uniform(initial_state_range[0,0], initial_state_range[0,1], size=num_samples),
        np.random.uniform(initial_state_range[1,0], initial_state_range[1,1], size=num_samples),
        np.random.uniform(initial_state_range[2,0], initial_state_range[2,1], size=num_samples)
    ], axis=1)

    xt_jax = jnp.array(xt)

    plt.figure(figsize=(6,6))
    plt.plot(xt[:,0], xt[:,1], 'o', alpha=0.3, label='t=0')

    # T-step propagation
    for t in range(num_steps):
        xt_jax = step_fn(xt_jax, jax_controller.get_action)
        xt_np = np.array(xt_jax)
        # print(xt_np)
        plt.plot(xt_np[:,0], xt_np[:,1], 'o', alpha=0.3, label=f't={t+1}')

    # Plot bounds if provided
    if bounds is not None:
        rect = Rectangle(
            (bounds[0,0], bounds[1,0]),
            bounds[0,1]-bounds[0,0],
            bounds[1,1]-bounds[1,0],
            fc="None",
            linewidth=2,
            edgecolor="red"
        )
        plt.gca().add_patch(rect)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"T-step samples with JAX controller (T={num_steps})")
    plt.axis("equal")
    plt.show()

def build_policy_sequential(hidden_size=64, num_actions=2):
    return nn.Sequential(OrderedDict([
        ("linear1", nn.Linear(3, hidden_size)),
        ("relu1", nn.ReLU()),

        ("linear2", nn.Linear(hidden_size, hidden_size)),
        ("relu2", nn.ReLU()),

        ("linear3", nn.Linear(hidden_size, hidden_size)),
        ("relu3", nn.ReLU()),

        ("linear4", nn.Linear(hidden_size, hidden_size)),
        ("relu4", nn.ReLU()),

        ("mean_linear", nn.Linear(hidden_size, num_actions)),
        # ("log_std_linear", nn.Linear(hidden_size, num_actions)),
    ]))

# Convert jax to torch
def jax_to_torch_weight(jax_kernel):
    return torch.tensor(np.array(jax_kernel), dtype=torch.float32).T

def jax_to_torch_bias(jax_bias):
    return torch.tensor(np.array(jax_bias), dtype=torch.float32)


# def state_propagation(model_path, state_range):

#     config_file = '/home/bera/nfl_veripy/src/nfl_veripy/_static/example_configs/playground.yaml'
#     with open(config_file, mode="r", encoding="utf-8") as file:
#             params = yaml.load(file, yaml.Loader)

#     dyn = dynamics.get_dynamics_instance(
#         params["system"]["type"], params["system"]["feedback"]
#     )

#     if not isinstance(state_range, list):
#         state_range = state_range.tolist()

#     state_range.append([0,0])

#     params["analysis"]["initial_state_range"] = state_range
    
#     device = "cpu"

#     checkpoint = torch.load(model_path, map_location=device)

#     torch_layers = [
#     'linear1', 'linear2', 'linear3', 'linear4',
#     'mean_linear', 'log_std_linear']

#     jax_params = {}

#     for layer_name in torch_layers:
#         weight = checkpoint[f'{layer_name}.weight'].numpy().T   # transpose for Flax
#         bias   = checkpoint[f'{layer_name}.bias'].numpy()
#         jax_params[layer_name] = {'kernel': jnp.array(weight), 'bias': jnp.array(bias)}

#     hidden_size = 128
#     num_actions = 2

#     policy = build_policy_sequential(hidden_size = hidden_size, num_actions=num_actions)

#     sd = policy.state_dict()

#     sd["linear1.weight"] = jax_to_torch_weight(jax_params["linear1"]["kernel"])
#     sd["linear1.bias"]   = jax_to_torch_bias(jax_params["linear1"]["bias"])

#     sd["linear2.weight"] = jax_to_torch_weight(jax_params["linear2"]["kernel"])
#     sd["linear2.bias"]   = jax_to_torch_bias(jax_params["linear2"]["bias"])

#     sd["linear3.weight"] = jax_to_torch_weight(jax_params["linear3"]["kernel"])
#     sd["linear3.bias"]   = jax_to_torch_bias(jax_params["linear3"]["bias"])

#     sd["linear4.weight"] = jax_to_torch_weight(jax_params["linear4"]["kernel"])
#     sd["linear4.bias"]   = jax_to_torch_bias(jax_params["linear4"]["bias"])

#     sd["mean_linear.weight"] = jax_to_torch_weight(jax_params["mean_linear"]["kernel"])
#     sd["mean_linear.bias"]   = jax_to_torch_bias(jax_params["mean_linear"]["bias"])

#     policy.load_state_dict(sd)

#     policy.eval()

#     analyzer = analyzers.ClosedLoopAnalyzer(policy, dyn)
#     analyzer.partitioner = params["analysis"]["partitioner"]
#     analyzer.propagator = params["analysis"]["propagator"]
#     analyzer.visualizer = params["visualization"]
    
#     initial_state_range = np.array(
#         params["analysis"]["initial_state_range"])
    
#     initial_state_set = constraints.state_range_to_constraint(
#         initial_state_range, params["analysis"]["propagator"]["boundary_type"]
#     )

#     stats = {}
#     reachable_sets, analyzer_info = analyzer.get_reachable_set(
#                 initial_state_set, t_max=params["analysis"]["t_max"])
    
    
#     stats["reachable_sets"] = reachable_sets
#     bounds = []

#     for i in range(params['analysis']['t_max']):
#         bounds.append(reachable_sets.constraints[i].range[:2])
#         # print("Overapprox range", reachable_sets.constraints[i].range)
#         # cell_count =  len(reachable_sets.constraints[i].cells)
#         # print("number of rectangles",cell_count)
#         # for j in range(cell_count):
#         #     print(f" set{i} sub-set{j} : {reachable_sets.constraints[i].cells[j].range} ")   
            
#         # print("---------------------------------")

# #     analyzer.visualizer.plot_filename = get_plot_filename(params)
# #     analyzer.visualize(
# #         initial_state_set,
# #         reachable_sets,
# #         analyzer.propagator.network,
# #         **analyzer_info,
# # )
    
#     return bounds

def state_propagation(model_path, state_range, goals, obstacles):
    config_file = '/home/bera/nfl_veripy/src/nfl_veripy/_static/example_configs/playground.yaml'
    with open(config_file, mode="r", encoding="utf-8") as file:
            params = yaml.load(file, yaml.Loader)

    dyn = dynamics.get_dynamics_instance(
        params["system"]["type"], params["system"]["feedback"]
    )

    if not isinstance(state_range, list):
        state_range = state_range.tolist()

    state_range.append([0,0])

    params["analysis"]["initial_state_range"] = state_range
    
    device = "cpu"

    checkpoint = torch.load(model_path, map_location=device)

    torch_layers = [
    'linear1', 'linear2', 'linear3', 'linear4',
    'mean_linear', 'log_std_linear']

    jax_params = {}

    for layer_name in torch_layers:
        weight = checkpoint[f'{layer_name}.weight'].numpy().T   # transpose for Flax
        bias   = checkpoint[f'{layer_name}.bias'].numpy()
        jax_params[layer_name] = {'kernel': jnp.array(weight), 'bias': jnp.array(bias)}

    hidden_size = 128
    num_actions = 2

    policy = build_policy_sequential(hidden_size = hidden_size, num_actions=num_actions)

    sd = policy.state_dict()

    sd["linear1.weight"] = jax_to_torch_weight(jax_params["linear1"]["kernel"])
    sd["linear1.bias"]   = jax_to_torch_bias(jax_params["linear1"]["bias"])

    sd["linear2.weight"] = jax_to_torch_weight(jax_params["linear2"]["kernel"])
    sd["linear2.bias"]   = jax_to_torch_bias(jax_params["linear2"]["bias"])

    sd["linear3.weight"] = jax_to_torch_weight(jax_params["linear3"]["kernel"])
    sd["linear3.bias"]   = jax_to_torch_bias(jax_params["linear3"]["bias"])

    sd["linear4.weight"] = jax_to_torch_weight(jax_params["linear4"]["kernel"])
    sd["linear4.bias"]   = jax_to_torch_bias(jax_params["linear4"]["bias"])

    sd["mean_linear.weight"] = jax_to_torch_weight(jax_params["mean_linear"]["kernel"])
    sd["mean_linear.bias"]   = jax_to_torch_bias(jax_params["mean_linear"]["bias"])

    policy.load_state_dict(sd)

    policy.eval()

    # jax_params_list = [
    #     (jax_params["linear1"]["kernel"],     jax_params["linear1"]["bias"]),
    #     (jax_params["linear2"]["kernel"],     jax_params["linear2"]["bias"]),
    #     (jax_params["linear3"]["kernel"],     jax_params["linear3"]["bias"]),
    #     (jax_params["linear4"]["kernel"],     jax_params["linear4"]["bias"]),
    #     (jax_params["mean_linear"]["kernel"], jax_params["mean_linear"]["bias"]),
    #     (jax_params["log_std_linear"]["kernel"], jax_params["log_std_linear"]["bias"]),
    # ]

    jax_policy_fn = pytorch_policy_to_jax(policy)
    #jax_policy_fn = functools.partial(relu_policy_nn, jax_params_list)
    action_range = jnp.array(params.get("action_range", [0.6, 0.6]))
    dt = float(params["system"].get("dt", 1.0))
    # dyn_B maps action (1,2) -> state contribution (1,3); dyn_c adds dt to time dim
    dyn_B = jnp.array([[dt, 0.0], [0.0, dt], [0.0, 0.0]])  # (3,2)
    dyn_c = jnp.array([[0.0, 0.0, dt]])                      # (1,3)

    def closed_loop_step(state):
        # Only lax.max (relu) is supported by backward CROWN — no lax.min, lax.tanh, lax.slice.
        # clip(x, -a, a) = relu(x+a) - relu(x-a) - a  (relu-only, no min/max needed)
        state_2d = jnp.reshape(state, (1, 3))
        mean, _ = jax_policy_fn(state_2d)                                              # (1,2)
        action = jnp.maximum(mean + action_range, 0.0) - jnp.maximum(mean - action_range, 0.0) - action_range
        next_state = state_2d + action @ dyn_B.T + dyn_c                               # (1,3)


        return jnp.reshape(next_state, (3,))
    

    analyzer = analyzers.ClosedLoopAnalyzer(policy, dyn)
    analyzer.partitioner = params["analysis"]["partitioner"]
    analyzer.propagator = params["analysis"]["propagator"]
    analyzer.visualizer = params["visualization"]

    initial_state_range = np.array(
        params["analysis"]["initial_state_range"])

    initial_state_set = constraints.state_range_to_constraint(
        initial_state_range, params["analysis"]["propagator"]["boundary_type"]
    )

    stats = {}
    # reachable_sets, analyzer_info = analyzer.get_reachable_set(
    #             initial_state_set, t_max=params["analysis"]["t_max"])

    lower = jnp.array(initial_state_range[:, 0], dtype=jnp.float32)
    upper = jnp.array(initial_state_range[:, 1], dtype=jnp.float32)
    jax_input_bound = jax_verify.IntervalBound(lower, upper)
    num_steps = 40

    #state step fn:
    step_fn_state = functools.partial(closed_loop_step_T, num_steps, jax_policy_fn, action_range, dyn_B, dyn_c, None)

    #predicate step fns:
    step_fn_goal = functools.partial(closed_loop_step_T, num_steps, jax_policy_fn, action_range, dyn_B, dyn_c, lambda s: goal_sdf(s, goals[0]["center"], goals[0]["radius"]))
    step_fn_obs1 = functools.partial(closed_loop_step_T, num_steps, jax_policy_fn, action_range, dyn_B, dyn_c, lambda s: obstacle_sdf(s, obstacles[0]["center"], obstacles[0]["radius"]))
    step_fn_obs2 = functools.partial(closed_loop_step_T, num_steps, jax_policy_fn, action_range, dyn_B, dyn_c, lambda s: obstacle_sdf(s, obstacles[1]["center"], obstacles[1]["radius"]))

    print('here')
    init_time = time.time()
    output_bounds_state = jax_verify.backward_crown_bound_propagation(step_fn_state, jax_input_bound)
    end_time = time.time()
    print(f"State propagation bounds computed in {end_time - init_time:.4f} seconds")
    print('here1')
    output_bounds_goal = jax_verify.backward_crown_bound_propagation(step_fn_goal, jax_input_bound)
    print('here2')
    output_bounds_obs1 = jax_verify.backward_crown_bound_propagation(step_fn_obs1, jax_input_bound)
    print('here3')
    output_bounds_obs2 = jax_verify.backward_crown_bound_propagation(step_fn_obs2, jax_input_bound)
    
    all_bounds = np.hstack([output_bounds_goal, output_bounds_obs1, output_bounds_obs2])
    
    
    return all_bounds


def dynamics_step_jnp(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    """
    state: (..., 3) -> x, y, time
    action: (..., 2) -> vx, vy
    returns next_state: (..., 3)
    """
    dt = 1
    x_next = state[..., 0] + action[..., 0] * dt
    y_next = state[..., 1] + action[..., 1] * dt
    t_next = state[..., 2] + dt
    return jnp.stack([x_next, y_next, t_next], axis=-1)

class SingleIntegratorDynamics:
    def __init__(self, dt=1.0):
        self.dt = dt

    def dynamics_step_jnp(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """
        state: (..., 3) -> x, y, time
        action: (..., 2) -> vx, vy
        returns next_state: (..., 3)
        """
        x_next = state[..., 0] + action[..., 0] * self.dt
        y_next = state[..., 1] + action[..., 1] * self.dt
        t_next = state[..., 2] + self.dt
        return jnp.stack([x_next, y_next, t_next], axis=-1)

_ex = jnp.array([1., 0., 0.], dtype=jnp.float32)
_ey = jnp.array([0., 1., 0.], dtype=jnp.float32)

def goal_sdf(state, center, radius):
    """Robustness of reaching goal: positive when inside goal region."""
    x = jnp.dot(state, _ex)
    y = jnp.dot(state, _ey)
    dist = jnp.sqrt((x - jnp.float32(center[0]))**2 + (y - jnp.float32(center[1]))**2)
    return jnp.float32(radius) - dist

def obstacle_sdf(state, center, radius):
    """Robustness of avoiding obstacle: positive when outside obstacle."""
    x = jnp.dot(state, _ex)
    y = jnp.dot(state, _ey)
    dist = jnp.sqrt((x - jnp.float32(center[0]))**2 + (y - jnp.float32(center[1]))**2)
    return dist - jnp.float32(radius)

def closed_loop_step_T(num_steps, jax_policy_fn, action_range, dyn_B, dyn_c, predicate_fn, state):
    """
    Propagates closed-loop dynamics for num_steps steps and evaluates predicate_fn
    at each step. Output: (3 + num_steps,) where output[:3] is the final state
    and output[3:] are per-step robustness values [rob_t1, ..., rob_tN].

    Usage:
        step_fn = functools.partial(closed_loop_step_T,
                                    num_steps, jax_policy_fn, action_range,
                                    dyn_B, dyn_c, predicate_fn)
        result = jax_verify.backward_crown_bound_propagation(step_fn, input_interval)
        rob_lower, rob_upper = result.lower[3:], result.upper[3:]
    """
    def single_step(carry, _x):
        s = carry
        s_2d = jnp.reshape(s, (1, 3))
        mean, _ = jax_policy_fn(s_2d)
        action = jnp.maximum(mean + action_range, 0.0) - jnp.maximum(mean - action_range, 0.0) - action_range
        next_s = jnp.reshape(s_2d + action @ dyn_B.T + dyn_c, (3,))
        if predicate_fn is not None:
            rob = predicate_fn(next_s)
            return next_s, rob  # (new_carry, output) — scan requires both
        else:
            return next_s, 0.0  # dummy robustness output when predicate_fn is None 

    state_seq, robustness_seq = jax.lax.scan(single_step, state, None, length=num_steps)

    if predicate_fn is not None:
        return robustness_seq
    else:
        return state_seq  # when predicate_fn is None, return final state instead of robustness sequence

if __name__ == "__main__":
    from simulator import Continuous2DEnv
    from SAC import SAC


    agent_init_loc = [-7.0, 4.0] #initial location of the agent
	#Static goal region:
    goal_region_radius = 1
    goals = {0: {'center': (3, -7), 'radius': goal_region_radius, 'movement':{'type':'static'}}, #goal region for the agent
    }

    obstacles = {
	0: {'label': 1,'center': (-3, -1.6), 'radius': 3, 'u_max': 0, 'remaining_time': 100, 'movement':{'type': 'static'}, 'color': 'blue'},
	1: {'label': 1,'center': (2, 3), 'radius': 2, 'u_max': 0, 'remaining_time': 100, 'movement':{'type': 'static'}, 'color': 'blue'}
	}

    disturbance_interval = [0, 0]

    #config dictionary for the environment
    config = {
        'init_loc':agent_init_loc, #initial location of the agent (x, y)
        "width": 10.0,
        "height": 10.0,
        "dt": 1,
        "render": False,
        'dt_render': 0.03,
        'goals': goals, #goal regions for the agent
        "obstacles": obstacles, #obstacle regions in the environment
        "randomize_loc": False, #whether to randomize the agent location at the end of each episode
        'deterministic': False,
        'auto_entropy':True,
        "dynamics": "single integrator", #dynamics model to use
        "targets": None, #target regions for the agent to visit
        "disturbance": disturbance_interval #disturbance range in both x and y directions [w_min, w_max]
    }

    action_range = [0.6, 0.6] #action range for the RL model (for the neural network output layer) [3,3] for case-1, [4,4] for case-2

    hyperparameters = {
                'gamma': 0.99,
                'tau': 0.005,
                'hidden_size': 64, 
                'buffer_size': int(1e6),
                'batch_size': 128,
                'max_timesteps_per_episode': 40, 
                'num_episodes': 200,
                'n_updates_per_iteration': 1,
                'deterministic': False,
                'auto_entropy':True,
                'action_range': action_range,
                'action_clip' : True,
                'CBF': False,
                'CBF_params': None,
                'online': False,
                'disturbance': disturbance_interval
                }
  
    env = Continuous2DEnv(config)
    model_path = select_model_file()
    device = "cpu"
                
    # model = SAC(env = env, **hyperparameters)

	# # Load in the actor model saved by the PPO algorithm
    # model.load_model(model_path)
    # pytorch_policy = model.policy_net
    # pytorch_policy.eval()
    
    # # Convert to JAX
    # jax_policy_fn = pytorch_policy_to_jax(pytorch_policy)
    
    # # Wrap in a controller
    # jax_controller = JAXPolicyController(jax_policy_fn, action_range=action_range, deterministic=True)
    
    # # Now you can use jax_controller to get actions
    # sample_state = np.array([[0.1, 0.2, 1]])  # Example state
    # action = jax_controller.get_action(sample_state)
    # print("Action from JAX controller:", action)

    # dyn = SingleIntegratorDynamics(dt=1.0)

    # initial_state_range = np.array([
    # [-7, -6.8],   # x
    # [4, 3.8],   # y
    # [0.0, 0.0]    # t
    # ])
    # num_steps = 20

    # plot_Tstep_samples_and_bounds(initial_state_range, step_1_batch, num_steps, bounds=None)

    config_file = '/home/bera/nfl_veripy/src/nfl_veripy/_static/example_configs/playground.yaml'
    with open(config_file, mode="r", encoding="utf-8") as file:
            params = yaml.load(file, yaml.Loader)
            
    
    dyn = dynamics.get_dynamics_instance(
        params["system"]["type"], params["system"]["feedback"]
    )

    verify_robustness(model_path)




