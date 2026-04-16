import numpy as np

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
 
 

def get_robustness(state_boundary, goal):    
    min_dist = min_distance_to_goal(goal[0]['center'], state_boundary)
    max_dist = max_distance_to_goal(goal[0]['center'], state_boundary)

    #dists = np.linalg.norm(corners - center, axis=1)
    # max_dist = dists.max()
    # min_dist = dists.min()
    worst_rho = goal[0]['radius'] - max_dist
    best_rho = goal[0]['radius'] - min_dist

    return worst_rho, best_rho




if __name__ == "__main__":
    state_boundary = np.array([[1.0, 5.0],
                               [-9.0, -5.0]])
    []
    goal = {0:{'center': (3, -7), 'radius': 1}}

    worst_rho, best_rho = get_robustness(state_boundary, goal)
    print("Worst-case Robustness value:", worst_rho)
    print("Best-case Robustness value:", best_rho)