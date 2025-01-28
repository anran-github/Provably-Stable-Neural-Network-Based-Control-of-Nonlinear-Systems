'''
This file draws:

(Comment) 1. How location changes with every start points (some are not points in dataset)    
    with OPT method and NN method. Find max of delta p and delta u.

===============================================
TIME DOMAIN NN vs LQR
===============================================

'''


import torch
import matplotlib.pyplot as plt
import numpy as np
from control import dlqr

# from network_ori import P_Net
from network import P_Net



# CHECK GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(f"Using device: {device}")

model1 = P_Net(output_size=4).to(device)
model2 = P_Net(output_size=4).to(device)
model3 = P_Net(output_size=4).to(device)
# test saved model
# Create an instance of the model
# Load the saved model's state dictionary
# model.load_state_dict(torch.load('trained_model_theta0.01_0.21945167709165908.pth'))

# model.load_state_dict(torch.load('weights/finetune_new_0.1/trained_model_theta0.1_loss10_epoch30500_3.426541805267334.pth',map_location=device))
model2.load_state_dict(torch.load('trained_model_theta0.001_loss10_epoch7500_1.1457103604362124.pth',map_location=device))
model3.load_state_dict(torch.load('trained_model_theta0.0001_loss10_epoch3500_1.1061427281016396.pth',map_location=device))
model1.load_state_dict(torch.load('trained_model_theta0.01_loss10_epoch9000_1.3507456893012637.pth',map_location=device))

# Set the model to evaluation mode (if needed)
model1.eval()
model2.eval()
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


# =================init states============================
x0 = np.array([[-2.3*np.pi/180.0],[5*np.pi/180.0]]).reshape(1,2)

for model in [model1,model2,model3]:
    # for each model

    Bd=np.array([0,0.1]).reshape(2,1)


    for i in range(num_points):
        # x1 = 10 * (np.random.rand() - 0.5)
        # x2 = 10 * (np.random.rand() - 0.5)
        # x0 = np.array([[x1],[x2]]).reshape(1,2)
        r = torch.tensor([[0.*np.pi/180.0]],dtype=torch.float32)
        
        p_tt = np.eye(2)

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
                
                # get LQR gain

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


# ======================================
# LQR PART

# LQR GAIN
Q = np.eye(2)*2  # Identity matrix of size 2x2
R = np.array([[0.1]])

Bd=np.array([0,0.1]).reshape(2,1)
# LQR terms
lqr_Ad=lambda x: [[1, 0.1],[-0.1*9.8*np.cos(x[0,0]), 1]]

lqr_x1_set = []
lqr_x2_set = []
lqr_u_set = []
lqr_delta_x_set = []

for t in range(iteration):
    if t == 0:
        xt = x0.T

    # get LQR gain
    
    k, _, _ = dlqr(lqr_Ad(xt),Bd,Q,R)
    Ad= np.array([np.array([xt[0,0]+0.1*xt[1,0]]),[-0.1*9.8*np.sin(xt[0,0]) + 1*xt[1,0]]])

    
    lqr_ut = -k @ xt
    A = np.array([[0,1],[-9.8*np.cos(xt[0,0]),0]])
    B = np.array([0,1]).reshape(2,1)

    xtt =  Ad+Bd @ lqr_ut

    lqr_x1_set.append(xt[0,0])
    lqr_x2_set.append(xt[1,0])
    lqr_u_set.append(lqr_ut.item())
    lqr_delta_x_set.append((xtt-xt))
    xt = xtt




# transfer to degrees
delta_x_set = np.array(delta_x_set).squeeze(2)
delta_x_set = delta_x_set*180/np.pi
u_set = np.array(u_set)*180/np.pi
x1_set = np.array(x1_set)*180/np.pi
x2_set = np.array(x2_set)*180/np.pi
v_dot_set = np.array(v_dot_set).reshape(-1)*180/np.pi

arrow_trajectory = np.array(arrow_trajectory)*180/np.pi

times = 0.1 * np.array(list(range(iteration)))


# same as lqr
lqr_u_set = np.array(lqr_u_set)*180/np.pi
lqr_x1_set = np.array(lqr_x1_set)*180/np.pi
lqr_x2_set = np.array(lqr_x2_set)*180/np.pi
lqr_delta_x_set = np.array(lqr_delta_x_set).squeeze(2)
lqr_delta_x_set = lqr_delta_x_set*180/np.pi


# =================================================
# Plot Results: x1-t, x2-t, u-t, Delta V - t, x1-x2
# =================================================
# introduce latex
plt.rcParams['text.usetex'] = True

# control number font size on axis.
plt.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "Times New Roman"


#  -------------x1-t----------------
plt.plot(times,x1_set[:iteration],'r',label=r'$\theta$=0.01')
plt.plot(times,x1_set[iteration:iteration*2],'g',label=r'$\theta$=0.001')
plt.plot(times,x1_set[iteration*2:],'b',label=r'$\theta$=0.0001')
plt.plot(times,lqr_x1_set[:iteration],'c',label=r'LQR')

plt.grid(linestyle = '--')
plt.xlim(0,np.max(times))
plt.xlabel(r'Time (s)', fontsize=16)
plt.ylabel(r'$x_1$ (deg)', fontsize=16)
plt.legend()
# plt.title('x1 with respect to time')
plt.tight_layout()
plt.savefig('paper_figures/x1.png')
plt.show()
plt.close()

#  -------------x2-t----------------
plt.plot(times,x2_set[:iteration],'r',label=r'$\theta$=0.01')
plt.plot(times,x2_set[iteration:iteration*2],'g',label=r'$\theta$=0.001')
plt.plot(times,x2_set[iteration*2:],'b',label=r'$\theta$=0.0001')
plt.plot(times,lqr_x2_set[:iteration],'c',label=r'LQR')
plt.grid(linestyle = '--')
plt.xlim(0,np.max(times))
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel(r'$x_2$ (deg/s)', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('paper_figures/x2.png')
# plt.title('x2 with respect to time')
plt.show()
plt.close()

#  -------------u-t----------------
plt.plot(times,u_set[:iteration],'r',label=r'$\theta$=0.01')
plt.plot(times,u_set[iteration:iteration*2],'g',label=r'$\theta$=0.001')
plt.plot(times,u_set[iteration*2:],'b',label=r'$\theta$=0.0001')
plt.plot(times,lqr_u_set[:iteration],'c',label=r'LQR')
plt.grid(linestyle = '--')
plt.xlim(0,np.max(times))
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel(r'$u$ (N$\cdot$m)', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('paper_figures/ut.png')
# plt.title('x2 with respect to time')
plt.show()
plt.close()

# introduce latex
# plt.rcParams['text.usetex'] = False

# control number font size on axis.
plt.rcParams.clear()
plt.rcParams.update()

fig, ax = plt.subplots()
# arrow_trajectory = np.array(delta_x_set)*180/np.pi
ax.quiver(x1_set[:iteration], x2_set[:iteration],delta_x_set[:iteration,0], delta_x_set[:iteration,1],
        color="r",label=r'$\theta$=0.01', angles='xy',scale_units='xy', scale=2, width=.007,headaxislength=4)
ax.quiver(x1_set[iteration:iteration*2], x2_set[iteration:iteration*2],
          delta_x_set[iteration:iteration*2,0], delta_x_set[iteration:iteration*2,1],
        color="g",label=r'$\theta$=0.001', angles='xy',scale_units='xy', scale=2, width=.007,headaxislength=4)
ax.quiver(x1_set[iteration*2:], x2_set[iteration*2:],delta_x_set[iteration*2:,0], delta_x_set[iteration*2:,1],
        color="b",label=r'$\theta$=0.0001', angles='xy',scale_units='xy', scale=2, width=.007,headaxislength=4)
ax.quiver(lqr_x1_set[:iteration], lqr_x2_set[:iteration],lqr_delta_x_set[:iteration,0], lqr_delta_x_set[:iteration,1],
        color="c",label=r'LQR', angles='xy',scale_units='xy', scale=2, width=.007,headaxislength=4)
plt.legend()
plt.grid(linestyle = '--')
plt.xlabel(r'$x_1$ (deg)', fontsize=16)
plt.ylabel(r'$x_2$ (deg/s)', fontsize=16)


# plt.title('input data changes with random points')
plt.tight_layout()
plt.savefig('paper_figures/x1x2.png')
plt.show()
plt.close()


# Delta V: should always less than 0 idealy.
fig, ax = plt.subplots()

plt.plot(times,v_dot_set[:iteration],'r',label=r'$\theta$=0.01')
plt.plot(times,v_dot_set[iteration:iteration*2],'g',label=r'$\theta$=0.001')
plt.plot(times,v_dot_set[iteration*2:],'b',label=r'$\theta$=0.0001')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\Delta{V}$', fontsize=16)
plt.legend()
plt.grid(linestyle = '--')
# plt.title('input data delta V with random points')
plt.xlim(0,5)
plt.tight_layout()
plt.savefig('paper_figures/delta_V.png')
plt.show()
plt.close()


# graph of ||x(t)||
fig, ax = plt.subplots()

print(f'summery: theta=0.01: {np.sum(arrow_trajectory[:iteration])}\n', 
      f'theta=0.001: {np.sum(arrow_trajectory[iteration:iteration*2])}\n',
        f'theta=0.001: {np.sum(arrow_trajectory[iteration*2:])}\n')

plt.plot(times,arrow_trajectory[:iteration],'r',label=r'$\theta$=0.01')
plt.plot(times,arrow_trajectory[iteration:iteration*2],'g',label=r'$\theta$=0.001')
plt.plot(times,arrow_trajectory[iteration*2:],'b',label=r'$\theta$=0.0001')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\left\Vert{x(t)}\right\Vert$', fontsize=16)
plt.legend()
plt.grid(linestyle = '--')
# plt.title('input data delta V with random points')
plt.xlim(0,5)
plt.tight_layout()
plt.savefig('paper_figures/normx.png')
plt.show()
plt.close()