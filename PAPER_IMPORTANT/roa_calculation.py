# ====================================================
# This File:
# Draw Region of Attraction with different NN models
# In paper, we only use theta = 0.001 ROA
#  The program also shows the data points numbers in ROA.
# ====================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import torch

from network import P_Net



# ============Select Theta================

# theta = 0.01
# file_name = "compare_opt_001.csv"
# model_weights = 'trained_model_theta0.01_loss10_epoch9000_1.3507456893012637.pth'

theta = 0.001
file_name = "compare_opt_0001.csv"
model_weights = 'trained_model_theta0.001_loss10_epoch7500_1.1457103604362124.pth'

# theta=0.0001
# file_name = "compare_opt_00001.csv"
# model_weights = 'trained_model_theta0.0001_loss10_epoch3500_1.1061427281016396.pth'

print(f'theta = {theta}')
# ========================================





# ================
# Load NN weights
# ================
# CHECK GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(f"Using device: {device}")

model = P_Net(output_size=4).to(device)


# model.load_state_dict(torch.load('weights/finetune_new_0.1/trained_model_theta0.1_loss10_epoch30500_3.426541805267334.pth',map_location=device))
model.load_state_dict(torch.load(model_weights,map_location=device))
model.eval()





# Generate data Grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

X, Y = np.meshgrid(x, y)
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)
data_input = np.concatenate([X,Y],1)
# r = np.zeros((xt.shape[0],1))

data_points = data_input*np.pi/180.0
num_pints = data_points.shape[0]

print(f'there are {num_pints} data points are applied!')


##---------Loop parameters--------------
# print(data_points.shape)
# batch size: data point size
Bds=np.array([0,0.1]).reshape(2,1)
Bd = np.tile(Bds, (num_pints, 1, 1))

r = torch.zeros((num_pints,1),dtype=torch.float32)

iteration = 50
boundary = 5*np.pi/180.0
v_dot_set = []
x_set = []
error_set = []
v_dot_set = []
mask_set = []

mask_final = np.abs(data_points[:,0]) < 1e9
# start iterations
for t in range(iteration):
    if t == 0:
        xt = data_points
    with torch.no_grad():
        x_tem = torch.tensor(xt)
        x_tem = x_tem.type(torch.float32)
        r = torch.zeros((xt.shape[0],1),dtype=torch.float32)
        output = model(torch.cat((x_tem,r),1).to(device))
        if not device.type == 'cuda':
            output = output.numpy()
        else:
            output = output.detach().cpu().numpy()


        Ad= np.array([np.array([xt[:,0]+0.1*xt[:,1]]),[-0.1*9.8*np.sin(xt[:,0]) + 1*xt[:,1]]])

        p_t = np.array([[output[:,0],output[:,1]],[output[:,1],output[:,2]]])

        u_t = output[:,3].reshape(-1,1,1)

        # (bs, 2, 1)
        xtt =  Ad.transpose(2,0,1) + Bd @ u_t

        # our previous V
        # v_dot = (A@xt + B@u_t).T @ p_t @ (A@xt + B@u_t) - (xt.T@p_t@xt)

        # find error for each point
        xtt = xtt.squeeze(2)
        
        # find roa
        mask_1 = xtt >= -boundary
        mask_1 = np.logical_and(mask_1[:,0],mask_1[:,1])
        mask_2 = xtt <= boundary
        mask_2 = np.logical_and(mask_2[:,0],mask_2[:,1])

        mask = np.logical_and(mask_1,mask_2)
        mask_final = np.logical_and(mask,mask_final)

        # only consider ROA points for next iteration.
        xt = xtt


# show ROA
points = data_points[mask_final]
print('there are {} data points are in ROA'.format(points.shape[0]))
points = points*180/np.pi


# introduce latex
plt.rcParams['text.usetex'] = True

# control number font size on axis.
plt.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "Times New Roman"

fig, ax = plt.subplots()
# plt.scatter(data_points[:,0]*180/np.pi,data_points[:,1]*180/np.pi)
plt.scatter(points[300,0],points[300,1])
plt.xlabel(r'$x_1$ (deg)', fontsize=16)
plt.ylabel(r'$x_2$ (deg/s)', fontsize=16)
# ax.set_title(r'Region of Attraction')


# create edge
edge = patches.Rectangle([-5,-5],width=10,height=10,facecolor='blue')
ax.add_patch(edge)


# Choose ROA for different theta
# ===================================================================
# Here we only use theta=0.001, because it fits all theta conditions.
# ===================================================================
if theta == 0.01:
    # Define the vertices of the polygon
    # vertices = [[-2.3, 5], [-1.6, -5], [2.3, -5], [1.6, 5]]
    vertices = [[-2.3, 5],[-2.14,2.55],[-2.15,0.4],[-2.12,-1.06],[-1.87,-2.8],
                  [-1.78,-3.83],[-1.6, -5], [2.3, -5],[2.14,-2.55],[2.15,-0.4],[2.12,1.06],
                  [1.87,2.8], [1.78,3.83],[1.6, 5]]

elif theta == 0.001:
    vertices = [[-2.3, 5],[-2.14,2.55],[-2.15,0.4],[-2.12,-1.06],[-1.87,-2.8],
                  [-1.6, -5], [2.3, -5],[2.14,-2.55],[2.15,-0.4],[2.12,1.06],
                  [1.87,2.8], [1.6, 5]]
elif theta == 0.0001:
    vertices = [[-2.4, 5],[-2.14,2.55],[-2.15,0.4],[-2.12,-1.06],[-1.87,-2.8],
                  [-1.6, -5], [2.4, -5],[2.14,-2.55],[2.15,-0.4],[2.12,1.06],
                  [1.87,2.8], [1.6, 5]]

# Create a polygon patch
polygon = patches.Polygon(vertices, edgecolor='black', facecolor='orange')

# Add the polygon patch to the axis
ax.add_patch(polygon)

# Set the aspect of the plot to equal
# ax.set_aspect('equal')
plt.xticks(np.arange(-5, 6, 1))
plt.yticks(np.arange(-5, 6, 1))

plt.tight_layout()
plt.savefig('paper_figures/roa.png')
plt.show()

