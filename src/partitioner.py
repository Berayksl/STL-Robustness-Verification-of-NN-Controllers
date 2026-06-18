import time

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#from calculate_robustness import get_robustness
from calculate_robustness_v2 import get_robustness_bounds, get_task_robustness
from verify_robustness import state_propagation
import tkinter as tk
from tkinter import filedialog
import os
 
# ============================================================
#  PartitionNode CLASS (Binary tree node)
# ============================================================
 
class PartitionNode:
    def __init__(self, cell, robustness_range, parent=None):
        self.cell = np.array(cell)        # shape (2,2): [[xmin,xmax],[ymin,ymax]]
        self.parent = parent              # pointer to parent PartitionNode
        self.left = None                  # left child
        self.right = None                 # right child
        self.verified = False             
        self.ignore = False
        self.robustness_range = robustness_range[robustness_range.argsort()]
        
        self.area = (self.cell[0,1] - self.cell[0,0]) * (self.cell[1,1] - self.cell[1,0])
        
        if self.robustness_range[0] >= 0: #if lower bound of robustness is positive -> verify
            self.verified = True
            
            
        if self.robustness_range[1] < 0: #if upper bound of robustness is negative -> ignore that node completely
            self.ignore = True
            
        if self.area < 0.01 and not self.verified:
            self.ignore = True
      
 
    def is_leaf(self):
        return (self.left is None) and (self.right is None)
 
    def __repr__(self):
        return f"PartitionNode(cell={self.cell}, verified={self.verified}, robustness={self.robustness_range}, ignore={self.ignore}, area={self.area})"
    
    
 
# ============================================================
#  RELATIONSHIP CHECKING FUNCTIONS
# ============================================================
 
def is_parent(parent, child):
    """
    Return True if 'parent' is the direct parent of 'child'.
    """
    return child.parent is parent
 
 
def is_ancestor(ancestor, node):
    """
    Return True if 'ancestor' is ANY ancestor of 'node'
    (i.e., parent, grandparent, great-grandparent...).
    """
    current = node.parent
    while current is not None:
        if current is ancestor:
            return True
        current = current.parent
    return False
 
 
def path_to_root(node):
    """
    Return the list of nodes from the given node up to the root.
    """
    path = []
    current = node
    while current is not None:
        path.append(current)
        current = current.parent
    return path  # leaf → ... → root
 
def leaves_to_struct(leaves):
    """
    Convert a list of PartitionNode leaves into a struct-like dictionary.
    Keys = attribute names
    Values = list of attribute values (in the same order as leaves)
    """
 
    if len(leaves) == 0:
        return {}
 
    struct = {}
 
    # Get all attributes from the first leaf (excluding private/internal)
    attributes = [
        attr for attr in dir(leaves[0])
        if not attr.startswith("_")
        and not callable(getattr(leaves[0], attr))
    ]
 
    # Initialize lists
    for attr in attributes:
        struct[attr] = []
 
    # Fill struct with ordered values
    for leaf in leaves:
        for attr in attributes:
            struct[attr].append(getattr(leaf, attr))
 
    return struct
 
 
 
# ============================================================
#  SPLITTING LOGIC
# ============================================================
 
def split_cell(cell):
    """
    Split a 2D rectangle in half.
    Rule:
        - If dy > dx: split horizontally (cut y)
        - Else:       split vertically   (cut x)
    """
    x0, x1 = cell[0]
    y0, y1 = cell[1]
 
    dx = x1 - x0
    dy = y1 - y0
 
    if dy > dx:
        # horizontal split (cut Y)
        y_mid = (y0 + y1) / 2
        cell1 = np.array([[x0, x1], [y0, y_mid]])
        cell2 = np.array([[x0, x1], [y_mid, y1]])
    else:
        # vertical split (cut X)
        x_mid = (x0 + x1) / 2
        cell1 = np.array([[x0, x_mid], [y0, y1]])
        cell2 = np.array([[x_mid, x1], [y0, y1]])
 
    return cell1, cell2
 
 
# ============================================================
#  RECURSIVE ADAPTIVE SPLITTING USING BINARY TREE
# ============================================================
 
def recursive_adaptive_partition_tree(model_path, input_range):
    """
    Build a binary tree by always splitting a leaf node.
    Each split produces 2 child nodes.
    
    Returns:
        root  : root PartitionNode
        leaves: list of leaf PartitionNodes
    """
    
    '''
    TODO:
    - Implement Bound Propation code to the root
    - Find robustness bound for root
    '''
    goals = {0: {'center': (3, -7), 'radius': 1, 'movement':{'type':'static'}}}#goal region for the agent}

    obstacles = {
	0: {'label': 0,'center': (-3, -1.6), 'radius': 3, 'u_max': 0, 'remaining_time': 100, 'movement':{'type': 'static'}, 'color': 'blue'},
	1: {'label': 1,'center': (2, 3), 'radius': 2, 'u_max': 0, 'remaining_time': 100, 'movement':{'type': 'static'}, 'color': 'blue'}
	}

    STL_task = """
        signal x, z, y    # signal names
        mux := x[t] > 0  # obstacle-1

        muz := z[t] > 0   # obstacle-2

        muy := y[t] > 0   

        phi1 := alw_[0, 30] mux # always (G works too)
        phi2 := alw_[0, 30] muz # always (G !obstacle-2)
        phi3 := ev_[0, 30] muy  # eventually (or F) 
        phi := phi1 and phi2 and phi3    # boolean combination 
        """

    all_bounds = state_propagation(model_path ,input_range, goals, obstacles)

    bounds = all_bounds[0]
    pred1_robustness_bounds = all_bounds[1]
    pred2_robustness_bounds = all_bounds[2]
    pred3_robustness_bounds = all_bounds[3]

    #pred1_robustness_bounds = [get_robustness_bounds(bounds[i], obstacles[0]) for i in range(len(bounds))] #robustness bounds for predicate 1 (G !obstacle1)
    # pred2_robustness_bounds = [get_robustness_bounds(bounds[i], obstacles[1]) for i in range(len(bounds))] #robustness bounds for predicate 2 (G !obstacle2)
    # pred3_robustness_bounds = [get_robustness_bounds(bounds[i], goals[0]) for i in range(len(bounds))] #robustness bounds for predicate 3 (F goal)


    # for i in range(len(bounds)):
    #     #for predicate 1
    #     min_rho, max_rho = get_robustness_bounds(bounds[i], obstacles[0])
    #     pred1_robustness_bounds.append((min_rho, max_rho))

    #     #for predicate 2
    #     min_rho, max_rho = get_robustness_bounds(bounds[i], obstacles[1])
    #     pred2_robustness_bounds.append((min_rho, max_rho))

    #     #for predicate 3
    #     min_rho, max_rho = get_robustness_bounds(bounds[i], goals)
    #     pred3_robustness_bounds.append((min_rho, max_rho))


    #combine lower and upper bounds of each predicate in one list:
    rho_mins = []
    rho_maxs = []
    for i in range(len(bounds)):
        rho_mins.append((pred1_robustness_bounds[i][0], pred2_robustness_bounds[i][0], pred3_robustness_bounds[i][0]))
        rho_maxs.append((pred1_robustness_bounds[i][1], pred2_robustness_bounds[i][1], pred3_robustness_bounds[i][1]))


    print("Rho mins:", rho_mins)
    print("Rho maxs:", rho_maxs)

    task_rho_min = get_task_robustness(STL_task, rho_mins)
    task_rho_max = get_task_robustness(STL_task, rho_maxs)

    print("Task robustness min:", task_rho_min)
    print("Task robustness max:", task_rho_max)


    # mins = [b[0] for b in pred1_robustness_bounds]
    # maxs = [b[1] for b in pred1_robustness_bounds]

    # rho_min = max(mins)
    # rho_max = max(maxs)

    root = PartitionNode(input_range, robustness_range=np.array([task_rho_min, task_rho_max]))
    leaves = [root]   # list of current leaves
    all_leaves_tracked = [root]
    verified_cell = []
 
    if root.verified == True:
        verified_cell.append(root)
        
 
    while len(leaves) > 0:
        print(len(leaves))
        
        #First, check if we still need to continue splitting
        all_nodes_to_check  = leaves_to_struct(leaves)
        nodes_to_ignore = [i for i, v in enumerate(all_nodes_to_check["ignore"]) if v is True] #indices
 
        # remove from paritioning list
        for index in sorted(nodes_to_ignore, reverse=True):
            del leaves[index]
        
        if len(leaves) == 0:
            break
        
        else:
            
            all_nodes_to_check  = leaves_to_struct(leaves)
            
            verified_node = [i for i, v in enumerate(all_nodes_to_check["verified"]) if v is True]
            
            #Collect all verified cells and
            for k in verified_node:
                verified_cell.append(leaves[k])
            
            # remove from paritioning list
            for index in sorted(verified_node, reverse=True):
                del leaves[index]
            
            if len(leaves) == 0:
                break
            
            else:
                
                all_nodes_to_check  = leaves_to_struct(leaves)
                
                all_robustness_to_be_verify = np.array(all_nodes_to_check["robustness_range"])
                idx_to_verify = all_robustness_to_be_verify[:,0].argmax()
                
                # print(all_robustness_to_be_verify)
                # print(idx_to_verify)
                # print(leaves)
                
                node_to_split = leaves.pop(idx_to_verify)
                
                
                # split rectangle
                cell1, cell2 = split_cell(node_to_split.cell)
                
                
                '''
                TODO:
                
                - Implement Bound Propation code to cell1 and cell2
                - Find robustness bound for cell1 and cell2
                '''
                cell1_bounds = state_propagation(model_path ,cell1)
                cell2_bounds = state_propagation(model_path ,cell2)

                cell1_pred1_robustness_bounds = [get_robustness_bounds(cell1_bounds[i], obstacles[0]) for i in range(len(cell1_bounds))] #robustness bounds for predicate 1 (G !obstacle1)
                cell1_pred2_robustness_bounds = [get_robustness_bounds(cell1_bounds[i], obstacles[1]) for i in range(len(cell1_bounds))] #robustness bounds for predicate 2 (G !obstacle2)
                cell1_pred3_robustness_bounds = [get_robustness_bounds(cell1_bounds[i], goals[0]) for i in range(len(cell1_bounds))] #robustness bounds for predicate 3 (F goal)

                cell2_pred1_robustness_bounds = [get_robustness_bounds(cell2_bounds[i], obstacles[0]) for i in range(len(cell2_bounds))] #robustness bounds for predicate 1 (G !obstacle1)
                cell2_pred2_robustness_bounds = [get_robustness_bounds(cell2_bounds[i], obstacles[1]) for i in range(len(cell2_bounds))] #robustness bounds for predicate 2 (G !obstacle2)
                cell2_pred3_robustness_bounds = [get_robustness_bounds(cell2_bounds[i], goals[0]) for i in range(len(cell2_bounds))] #robustness bounds for predicate 3 (F goal)

                cell1_rho_mins = []
                cell1_rho_maxs = []
                cell2_rho_mins = []
                cell2_rho_maxs = []
                for i in range(len(cell1_bounds)):
                    cell1_rho_mins.append((-cell1_pred1_robustness_bounds[i][1], -cell1_pred2_robustness_bounds[i][1], cell1_pred3_robustness_bounds[i][0]))
                    cell1_rho_maxs.append((-cell1_pred1_robustness_bounds[i][0], -cell1_pred2_robustness_bounds[i][0], cell1_pred3_robustness_bounds[i][1]))

                    cell2_rho_mins.append((-cell2_pred1_robustness_bounds[i][1], -cell2_pred2_robustness_bounds[i][1], cell2_pred3_robustness_bounds[i][0]))
                    cell2_rho_maxs.append((-cell2_pred1_robustness_bounds[i][0], -cell2_pred2_robustness_bounds[i][0], cell2_pred3_robustness_bounds[i][1]))

                cell1_task_rho_min = get_task_robustness(STL_task, cell1_rho_mins)
                cell1_task_rho_max = get_task_robustness(STL_task, cell1_rho_maxs)

                cell2_task_rho_min = get_task_robustness(STL_task, cell2_rho_mins)
                cell2_task_rho_max = get_task_robustness(STL_task, cell2_rho_maxs)


                # create children nodes
                left_child = PartitionNode(cell1, parent=node_to_split, robustness_range=np.array([cell1_task_rho_min, cell1_task_rho_max]))
                right_child = PartitionNode(cell2, parent=node_to_split, robustness_range=np.array([cell2_task_rho_min, cell2_task_rho_max]))
 
                # attach children
                node_to_split.left = left_child
                node_to_split.right = right_child
 
                # add new leaves
                leaves.append(left_child)
                leaves.append(right_child)

                # add new leaves
                all_leaves_tracked.append(left_child)
                all_leaves_tracked.append(right_child)

    
            print('verified cells:', len(verified_cell))
    return verified_cell, all_leaves_tracked
 
 
# ============================================================
#  PLOTTING
# ============================================================
 
def plot_cells(leaves, color='blue', ax=None, fill_in=False):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
 
    for node in leaves:
        cell = node.cell
        (x0, x1), (y0, y1) = cell[0], cell[1]
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=fill_in, edgecolor=color
        )
        ax.add_patch(rect)
 
 
    return ax
    
    
def select_model_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

    file_path = filedialog.askopenfilename(
        title="Select Configuration File",
        initialdir=model_dir,
        filetypes=[("Model Files", "*.pth"), ("All Files", "*.*")])
    return file_path

 
# ============================================================
# EXAMPLE USAGE
# ============================================================
 
if __name__ == "__main__":
 
    # input_range = np.array([
    #     [-2.0, 2.0],     # x-range
    #     [-2.0, 2.0],    # y-range
    # ])
    model_path = select_model_file()
    input_range = [[7, 10], [4, 7]]

    init_time = time.time()
    verified_cell, all_leaves_tracked = recursive_adaptive_partition_tree(model_path, input_range)

    end_time = time.time()
    print(f"Total time taken: {end_time - init_time:.4f} seconds")

    print("Verified Cell:",verified_cell)
    print("All leaves tracked:",all_leaves_tracked)

 
    # Show all leaves
    for leaf in verified_cell:
        print(leaf)
 
    # Plot
    ax = plot_cells(all_leaves_tracked, color='blue', fill_in=False)
    ax = plot_cells(verified_cell, color='white', ax=ax, fill_in=True)
    ax.set_xlim(input_range[0][0], input_range[0][1])
    ax.set_ylim(input_range[1][0], input_range[1][1])
    ax.set_title("Partitioning Result")
    #plt.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()
 
    #print(all_leaves_tracked)
    
 
    # 
    # recursive_adaptive_partition_tree(model_path, input_range)