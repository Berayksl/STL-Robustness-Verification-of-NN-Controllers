import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- paste your PartitionNode class here if needed ---

 
class PartitionNode:
    def __init__(self, cell, robustness_range, parent=None):
        self.cell = np.array(cell)        # shape (2,2): [[xmin,xmax],[ymin,ymax]]
        self.parent = parent              # pointer to parent PartitionNode
        self.left = None                  # left child
        self.right = None                 # right child
        self.verified = False             
        self.ignore = False
        #self.robustness_range = robustness_range[robustness_range.argsort()]
        self.robustness_range = np.array(robustness_range)
        self.robustness_range = self.robustness_range[self.robustness_range.argsort()]

        
        self.area = (self.cell[0,1] - self.cell[0,0]) * (self.cell[1,1] - self.cell[1,0])
        
        if self.robustness_range[0] >= 0: #if lower bound of robustness is positive -> verify
            self.verified = True
            
            
        if self.robustness_range[1] < 0: #if upper bound of robustness is negative -> ignore that node completely
            self.ignore = True
            
        if self.area < 0.25 and not self.verified:
            self.ignore = True
      
 
    def is_leaf(self):
        return (self.left is None) and (self.right is None)
 
    def __repr__(self):
        return f"PartitionNode(cell={self.cell}, verified={self.verified}, robustness={self.robustness_range}, ignore={self.ignore}, area={self.area})"
    

verified_cells = [
    PartitionNode(cell=[[-6.5, -6.375],[3.25, 3.5]], robustness_range=[0.65043789, 0.76777267]),
    PartitionNode(cell=[[-6.375, -6.25],[3.25, 3.5]], robustness_range=[0.52649952, 0.80433681]),
    PartitionNode(cell=[[-6.5, -6.375],[3.0, 3.25]], robustness_range=[0.54142565, 0.83527637]),
    PartitionNode(cell=[[-6.375, -6.25],[3.0, 3.25]], robustness_range=[0.50552483, 0.82442729]),
    PartitionNode(cell=[[-6.25, -6.0],[3.0, 3.25]], robustness_range=[0.3878898, 0.92807574]),
    PartitionNode(cell=[[-6.125, -6.0],[3.5, 3.75]], robustness_range=[0.47558954, 0.8863225]),
    PartitionNode(cell=[[-6.125, -6.0],[3.75, 4.0]], robustness_range=[0.36655963, 0.93544126]),
    PartitionNode(cell=[[-6.75, -6.5],[3.25, 3.5]], robustness_range=[0.42939676, 0.87094259]),
    PartitionNode(cell=[[-7.0, -6.75],[3.25, 3.5]], robustness_range=[0.50680493, 0.81681848]),
]

fig, ax = plt.subplots(figsize=(8, 8))

# Plot each verified cell as a blue rectangle
for node in verified_cells:
    xmin, xmax = node.cell[0]
    ymin, ymax = node.cell[1]
    width = xmax - xmin
    height = ymax - ymin

    rect = patches.Rectangle((xmin, ymin), width, height,
                             linewidth=1,
                             edgecolor='black',
                             facecolor='blue',
                             alpha=0.4)
    ax.add_patch(rect)

    # Optionally: mark center point
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    ax.plot(cx, cy, 'ko', markersize=3)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Verified Cells (filled in blue)")
ax.set_aspect('equal', adjustable='box')

plt.grid(True)
plt.show()
