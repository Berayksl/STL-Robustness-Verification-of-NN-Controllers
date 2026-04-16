import numpy as np
import stlrom

def min_distance_to_goal(point, box):
    """
    Project a point onto a rectangular (axis-aligned) box.
    """
    point = np.asarray(point)
    box = np.asarray(box)
    lower = box[:, 0]
    upper = box[:, 1]

    closest_point = np.minimum(np.maximum(point, lower), upper)

    min_distance = np.linalg.norm(closest_point - point)
    return min_distance

def max_distance_to_goal(point, box):
    """
    Compute the furthest point in an axis-aligned box from a query point.
 
    Parameters:
        point: (n,) array
        box: (n,2) array  [[l1,u1], [l2,u2], ...]
 
    Returns:
        y: the furthest point in the box from `point`
    """
    point = np.asarray(point)
    box = np.asarray(box)
 
    lower = box[:, 0]
    upper = box[:, 1]
    mid   = (lower + upper) / 2
 
    # choose opposite endpoint from where point lies
    y = np.where(point >= mid, lower, upper)
    
    max_distance = np.linalg.norm(y - point)
    return max_distance


def get_robustness_bounds(state_boundary, goal):   
    #returns the best and worst case robustness values of a predicate for a given state boundary and goal region
    min_dist = min_distance_to_goal(goal['center'], state_boundary)
    max_dist = max_distance_to_goal(goal['center'], state_boundary)

    #dists = np.linalg.norm(corners - center, axis=1)
    # max_dist = dists.max()
    # min_dist = dists.min()
    worst_rho = goal['radius'] - max_dist
    best_rho = goal['radius'] - min_dist

    return worst_rho, best_rho



def get_task_robustness(task, signals):
    stl_driver =stlrom.STLDriver()
    stl_driver.parse_string(task)

    #add the samples:
    for i in range(len(signals)):
        stl_driver.add_sample([i, signals[i][0], signals[i][1], signals[i][2]])   #format [t, pred1_val, pred2_val, ...]

    phi = stl_driver.get_monitor("phi") #overall task
    robustness = phi.eval_rob()

    return robustness       