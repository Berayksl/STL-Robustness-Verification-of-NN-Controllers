from SAC import SAC
import torch
from simulator import Continuous2DEnv
import tkinter as tk
from tkinter import filedialog
import os
from networks import PolicyNetwork
import numpy as np
from dynamics import SingleIntegratorDynamics

def select_model_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

    file_path = filedialog.askopenfilename(
        title="Select Configuration File",
        initialdir=model_dir,
        filetypes=[("Model Files", "*.pth"), ("All Files", "*.*")])
    return file_path




def step_robustness_fn(state, policy_net, goals):
    #Returns the robustness value for the next state
    action = policy_net.get_action(state, deterministic=True)
    dynamics = SingleIntegratorDynamics(x = state[0], y = state[1], dt=1)
    next_state = dynamics.update(action)
    dist = np.linalg.norm(next_state - np.array(goals[0]['center']))
    rho = goals[0]['radius'] - dist 

    return rho



if __name__ == "__main__":
    disturbance_interval = [0, 0]
    action_range = np.array([0.5, 0.5])
    goal_region_radius = 1

    goals = {
	0: {'center': (3, -7), 'radius': goal_region_radius, 'movement':{'type':'static'}}, #goal region for the agent
	}

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
    
    model_path = select_model_file()

    policy_net = PolicyNetwork()
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()





