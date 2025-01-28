'''
This file draws:

===============================================
These figures will be inserted into final paper.
===============================================
USE DATA FROM MATLAB (OPT RESULT) :
For All Theta:
1. Given a reference r and a random start point, plot figures of x1 and x2 change vs time.
2. Same condition, trajectory of (x1,x2)
3. Same condition, Delta V of this process.

==============================================================
Summary of Errors between xi and x_r with 10000 random points.
==============================================================
'''


import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time 
# from network_ori import P_Net
from network import P_Net



# CHECK GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(f"Using device: {device}")

model1 = P_Net(output_size=4).to(device)
# model2 = P_Net(output_size=4).to(device)
model3 = P_Net(output_size=4).to(device)
# test saved model
# Create an instance of the model
# Load the saved model's state dictionary
# model.load_state_dict(torch.load('trained_model_theta0.01_0.21945167709165908.pth'))

# model.load_state_dict(torch.load('weights/finetune_new_0.1/trained_model_theta0.1_loss10_epoch30500_3.426541805267334.pth',map_location=device))
# model2.load_state_dict(torch.load('trained_model_theta0.001_loss10_epoch7500_1.1457103604362124.pth',map_location=device))
model3.load_state_dict(torch.load('trained_model_theta0.0001_loss10_epoch3500_1.1061427281016396.pth',map_location=device))
model1.load_state_dict(torch.load('trained_model_theta0.01_loss10_epoch9000_1.3507456893012637.pth',map_location=device))

# Set the model to evaluation mode (if needed)
model1.eval()
# model2.eval()
model3.eval()


# init summary data
num_points = 1
iteration = 51
u_set = []
x1_set = []
x2_set = []
delta_x_set = []
v_dot_set = []
arrow_trajectory = []

for model in [model1,model3]:
    # for each model

    Bd=np.array([0,0.1]).reshape(2,1)
    # Ad=lambda xt: [[1, 0.1],[-0.1*9.8*np.cos(xt), 1]]

    for i in range(num_points):
        # x1 = 10 * (np.random.rand() - 0.5)
        # x2 = 10 * (np.random.rand() - 0.5)
        # x0 = np.array([[x1],[x2]]).reshape(1,2)
        r = torch.tensor([[0.*np.pi/180.0]],dtype=torch.float32)
        
        x0 = np.array([[2*np.pi/180.0],[-5*np.pi/180.0]]).reshape(1,2)
        p_tt = np.eye(2)
        start_time = time.time()
        for t in range(iteration):
            if t == 0:
                xt = x0
            with torch.no_grad():
                x_tem = torch.tensor(xt)
                x_tem = x_tem.type(torch.float32)
                output = model(torch.cat((x_tem.reshape(1,2),r),1).to(device))
                if not device.type == 'cuda':
                    output = output.numpy()
                else:
                    output = output.detach().cpu().numpy()

                # k, _, _ = dlqr(A, B, Q, R)
                # Ad= np.array([np.array([1, 0.1]),[-0.1*9.8*np.cos(xt.numpy()), 1]])
                xt = xt.reshape(2,1)
                Ad= np.array([np.array([xt[0,0]+0.1*xt[1,0]]),[-0.1*9.8*np.sin(xt[0,0]) + 1*xt[1,0]]])

                p_t = np.array([[output[0,0],output[0,1]],[output[0,1],output[0,2]]])

                u_t = output[0,3].reshape(1,1)
                xt = xt.reshape(2,1)
                # uncomment this line if you want to compare to dlqr.
                # u_t = -torch.from_numpy(k).type(torch.float32) @ xt
                xtt =  Ad+Bd @ u_t
                x1_set.append(xt[0,0])
                x2_set.append(xt[1,0])
                # our previous V

                v_dot = np.sqrt(np.linalg.norm(xtt.T @ p_t @ xtt ,axis=1)) - np.sqrt(np.linalg.norm(xt.T @ p_tt @ xt,axis=1))
                v_dot_set.append(v_dot)
                u_set.append(u_t.item())
                ##======REMEMBER TO CHEKCK THETA HERE (0.01)======
                arrow_trajectory.append(np.linalg.norm(xt))
                delta_x_set.append((xtt-xt))
                xt = xtt
                p_tt = p_t
        end_time = time.time()
        print(f'NN method cost: {end_time-start_time}s for one theta')


# transfer to degrees
delta_x_set = np.array(delta_x_set).squeeze(2)
delta_x_set = delta_x_set*180/np.pi
u_set = np.array(u_set)*180/np.pi
x1_set = np.array(x1_set)*180/np.pi
x2_set = np.array(x2_set)*180/np.pi
v_dot_set = np.array(v_dot_set).reshape(-1)*180/np.pi
arrow_trajectory = np.array(arrow_trajectory)*180/np.pi


times = 0.1 * np.array(list(range(iteration)))



# OPEN MATLAB DATA 

# show label data
df1 = pd.read_csv('2_-5_graph_timedomain0.01.csv', header=None)
df3 = pd.read_csv('2_-5_graph_timedomain0.0001.csv', header=None)
# Assuming the first row is the header, adjust if needed
data1 = df1.values  
data3 = df3.values  


data_x_set = []
data_deltax_set = []
data_normx_set = []
data_u_set = []
data_deltav_set = []

for data in [data1,data3]:
    for row in range(data.shape[0]//2):
        tmp_x = [data[2*row:2*row+2,i] for i in range(data.shape[1]) if i%5==0]
        tmp_data_u = [data[2*row,i+3] for i in range(data.shape[1]) if i%5==0]
        tmp_data_normx = [data[2*row,i+2] for i in range(data.shape[1]) if i%5==0]
        tmp_data_delta_v = [data[2*row,i+4] for i in range(data.shape[1]) if i%5==0]
        tmp_delta_x = [data[2*row:2*row+2,i+1] for i in range(data.shape[1]) if i%5==0]
        # tmp_data_u = np.array(data_u)
        data_x_set.extend(tmp_x)
        data_u_set.extend(tmp_data_u)
        data_normx_set.extend(tmp_data_normx)
        data_deltav_set.extend(tmp_data_delta_v)
        data_deltax_set.extend(tmp_delta_x)


data_x_set = np.array(data_x_set)*180/np.pi
data_u_set = np.array(data_u_set)*180/np.pi
data_normx_set = np.array(data_normx_set)*180/np.pi
data_deltav_set = np.array(data_deltav_set)*180/np.pi
data_deltax_set = np.array(data_deltax_set)*180/np.pi

# =================================================
# Plot Results: x1-t, x2-t, u-t, Delta V - t, x1-x2
# =================================================
# introduce latex
plt.rcParams['text.usetex'] = True

# control number font size on axis.
plt.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "Times New Roman"


#  -------------x1-t theta0.01----------------
plt.plot(times,x1_set[:iteration],'r',label=r'$NN\theta$=0.01')
plt.plot(times,data_x_set[:iteration,0],'b',label=r'$OPT\theta$=0.01')
# plt.plot(times,x1_set[iteration:iteration*2],'b',label=r'$\theta$=0.0001')
# plt.plot(times,x1_set[iteration*2:],'b',label=r'$\theta$=0.0001')
plt.grid(linestyle = '--')
plt.xlim(0,np.max(times))
plt.xlabel(r'Time (s)', fontsize=16)
plt.ylabel(r'$x_1$ (deg)', fontsize=16)
plt.legend()
# plt.title('x1 with respect to time')
plt.tight_layout()
plt.savefig('paper_figures/theta001x1.png')
plt.show()
plt.close()


# ----------IMPORTANT! SHOW IN PAPER!---------------------------
#  -------------x1-t theta0.0001 Appear in Paper----------------
plt.subplot(121)
plt.plot(times,x1_set[iteration:],'r',label=r'NN')
plt.plot(times,data_x_set[iteration:,0],'b',label=r'OPT')
# plt.plot(times,x1_set[iteration:iteration*2],'b',label=r'$\theta$=0.0001')
# plt.plot(times,x1_set[iteration*2:],'b',label=r'$\theta$=0.0001')
plt.grid(linestyle = '--')
plt.xlim(0,np.max(times))
plt.xlabel(r'Time (s)', fontsize=16)
plt.ylabel(r'$x_1$ (deg)', fontsize=16)
plt.legend()
# plt.title('x1 with respect to time')
plt.tight_layout()
# plt.savefig('paper_figures/NNOPTx1.png')
# plt.show()
# plt.close()

#  -------------x2-t theta0.0001 Appear in Paper----------------
plt.subplot(122)
plt.plot(times,x2_set[iteration:],'r',label=r'NN')
plt.plot(times,data_x_set[iteration:,1],'b',label=r'OPT')
plt.grid(linestyle = '--')
plt.xlim(0,np.max(times))
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel(r'$x_2$ (deg/s)', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('paper_figures/NNOPTx1x2.png')
# plt.title('x2 with respect to time')
plt.show()
plt.close()


exit()

#  -------------x2-t theta0.01----------------
plt.plot(times,x2_set[:iteration],'r',label=r'$NN\theta$=0.01')
plt.plot(times,data_x_set[:iteration,1],'b',label=r'$OPT\theta$=0.01')
plt.grid(linestyle = '--')
plt.xlim(0,np.max(times))
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel(r'$x_2$ (deg/s)', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('paper_figures/theta001x2.png')
# plt.title('x2 with respect to time')
plt.show()
plt.close()

#  -------------x2-t theta0.0001 Appear in Paper----------------
plt.plot(times,x2_set[iteration:],'r',label=r'NN-based')
plt.plot(times,data_x_set[iteration:,1],'b',label=r'One-Step-Ahead')
plt.grid(linestyle = '--')
plt.xlim(0,np.max(times))
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel(r'$x_2$ (deg/s)', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('paper_figures/NNOPTx2.png')
# plt.title('x2 with respect to time')
plt.show()
plt.close()

#  -------------u-t theta0.01----------------
plt.plot(times,u_set[:iteration],'r',label=r'$NN\theta$=0.01')
plt.plot(times,data_u_set[:iteration],'b',label=r'$OPT\theta$=0.01')
plt.grid(linestyle = '--')
plt.xlim(0,np.max(times))
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel(r'$u$ (N$\cdot$m)', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('paper_figures/theta001ut.png')
# plt.title('x2 with respect to time')
plt.show()
plt.close()

#  -------------u-t theta0.0001----------------
plt.plot(times,u_set[iteration:],'r',label=r'$NN\theta$=0.0001')
plt.plot(times,data_u_set[iteration:],'b',label=r'$OPT\theta$=0.0001')
plt.grid(linestyle = '--')
plt.xlim(0,np.max(times))
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel(r'$u$ (N$\cdot$m)', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('paper_figures/theta00001ut.png')
# plt.title('x2 with respect to time')
plt.show()
plt.close()


# introduce latex
# plt.rcParams['text.usetex'] = False

# control number font size on axis.
plt.rcParams.clear()
plt.rcParams.update()


# -----------x1x2 theta0.01---------------------------
fig, ax = plt.subplots()
# arrow_trajectory = np.array(delta_x_set)*180/np.pi
ax.quiver(x1_set[:iteration], x2_set[:iteration],delta_x_set[:iteration,0], delta_x_set[:iteration,1],
        color="r",label=r'$NN\theta$=0.01', angles='xy',scale_units='xy', scale=2, width=.007,headaxislength=4)
ax.quiver(data_x_set[:iteration,0], data_x_set[:iteration,1],
          data_deltax_set[:iteration,0], data_deltax_set[:iteration,1],
        color="b",label=r'$OPT\theta$=0.01', angles='xy',scale_units='xy', scale=2, width=.007,headaxislength=4)
# ax.quiver(x1_set[iteration:], x2_set[iteration*2:],delta_x_set[iteration*2:,0], delta_x_set[iteration*2:,1],
#         color="b",label=r'$\theta$=0.0001', angles='xy',scale_units='xy', scale=2, width=.007,headaxislength=4)
plt.legend()
plt.grid(linestyle = '--')
plt.xlabel(r'$x_1$ (deg)', fontsize=16)
plt.ylabel(r'$x_2$ (deg/s)', fontsize=16)


# plt.title('input data changes with random points')
plt.tight_layout()
plt.savefig('paper_figures/theta001x1x2.png')
plt.show()
plt.close()


# -----------x1x2 theta0.01---------------------------
fig, ax = plt.subplots()
# arrow_trajectory = np.array(delta_x_set)*180/np.pi
ax.quiver(x1_set[iteration:], x2_set[iteration:],delta_x_set[iteration:,0], delta_x_set[iteration:,1],
        color="r",label=r'$NN\theta$=0.0001', angles='xy',scale_units='xy', scale=2, width=.007,headaxislength=4)
ax.quiver(data_x_set[iteration:,0], data_x_set[iteration:,1],
          data_deltax_set[iteration:,0], data_deltax_set[iteration:,1],
        color="b",label=r'$OPT\theta$=0.0001', angles='xy',scale_units='xy', scale=2, width=.007,headaxislength=4)
# ax.quiver(x1_set[iteration:], x2_set[iteration*2:],delta_x_set[iteration*2:,0], delta_x_set[iteration*2:,1],
#         color="b",label=r'$\theta$=0.0001', angles='xy',scale_units='xy', scale=2, width=.007,headaxislength=4)
plt.legend()
plt.grid(linestyle = '--')
plt.xlabel(r'$x_1$ (deg)', fontsize=16)
plt.ylabel(r'$x_2$ (deg/s)', fontsize=16)

# plt.title('input data changes with random points')
plt.tight_layout()
plt.savefig('paper_figures/theta00001x1x2.png')
plt.show()
plt.close()


# Delta V: should always less than 0 idealy.
# ------------delta v theta0.01-------------------
fig, ax = plt.subplots()
plt.plot(times,v_dot_set[:iteration],'r',label=r'$NN\theta$=0.01')
plt.plot(times,data_deltav_set[:iteration],'b',label=r'$OPT\theta$=0.01')
# plt.plot(times,v_dot_set[iteration*2:],'b',label=r'$\theta$=0.0001')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\Delta{V}$', fontsize=16)
plt.legend()
plt.grid(linestyle = '--')
# plt.title('input data delta V with random points')
plt.xlim(0,5)
plt.tight_layout()
plt.savefig('paper_figures/theta001delta_V.png')
plt.show()
plt.close()

# ------------delta v theta0.0001-------------------
fig, ax = plt.subplots()
plt.plot(times,v_dot_set[iteration:],'r',label=r'$NN\theta$=0.0001')
plt.plot(times,data_deltav_set[iteration:],'b',label=r'$OPT\theta$=0.0001')
# plt.plot(times,v_dot_set[iteration*2:],'b',label=r'$\theta$=0.0001')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\Delta{V}$', fontsize=16)
plt.legend()
plt.grid(linestyle = '--')
# plt.title('input data delta V with random points')
plt.xlim(0,5)
plt.tight_layout()
plt.savefig('paper_figures/theta00001delta_V.png')
plt.show()
plt.close()


# graph of ||x(t)||

# -----------||x(t)|| theta0.01-----------------------
fig, ax = plt.subplots()

print(f'summery: theta=0.01: {np.sum(arrow_trajectory[:iteration])}\n', 
      f'theta=0.001: {np.sum(arrow_trajectory[iteration:iteration*2])}\n',
        f'theta=0.001: {np.sum(arrow_trajectory[iteration*2:])}\n')

plt.plot(times,arrow_trajectory[:iteration],'r',label=r'$NN\theta$=0.01')
plt.plot(times,data_normx_set[:iteration],'b',label=r'$OPT\theta$=0.01')
# plt.plot(times,arrow_trajectory[iteration:iteration*2],'g',label=r'$\theta$=0.001')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\left\Vert{x(t)}\right\Vert$', fontsize=16)
plt.legend()
plt.grid(linestyle = '--')
# plt.title('input data delta V with random points')
plt.xlim(0,5)
plt.tight_layout()
plt.savefig('paper_figures/theta001normx.png')
plt.show()
plt.close()

# -----------||x(t)|| theta0.0001-----------------------
fig, ax = plt.subplots()

print('--------------NN summary---------------------')
print(f'summery: theta=0.01: {np.sum(arrow_trajectory[:iteration])}\n', 
        f'theta=0.0001: {np.sum(arrow_trajectory[iteration:])}\n')
print('--------------OPT summary---------------------')
print(f'summery: theta=0.01: {np.sum(data_normx_set[:iteration])}\n', 
        f'theta=0.0001: {np.sum(data_normx_set[iteration:])}\n')
# print(f'summery: theta=0.01: {np.sum(arrow_trajectory[:iteration])}\n', 
#       f'theta=0.001: {np.sum(arrow_trajectory[iteration:iteration*2])}\n',
#         f'theta=0.001: {np.sum(arrow_trajectory[iteration*2:])}\n')

plt.plot(times,arrow_trajectory[iteration:],'r',label=r'$NN\theta$=0.0001')
plt.plot(times,data_normx_set[iteration:],'b',label=r'$OPT\theta$=0.0001')
# plt.plot(times,arrow_trajectory[iteration:iteration*2],'g',label=r'$\theta$=0.001')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\left\Vert{x(t)}\right\Vert$', fontsize=16)
plt.legend()
plt.grid(linestyle = '--')
# plt.title('input data delta V with random points')
plt.xlim(0,5)
plt.tight_layout()
plt.savefig('paper_figures/theta00001normx.png')
plt.show()
plt.close()