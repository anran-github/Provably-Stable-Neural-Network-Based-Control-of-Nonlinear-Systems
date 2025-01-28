import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from control import lqr, dlqr

from network import P_Net



# CHECK GPU/CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(f"Using device: {device}")

model = P_Net(output_size=4).to(device)

# test saved model
# Create an instance of the model
# Load the saved model's state dictionary
# model.load_state_dict(torch.load('trained_model_data4_0.06362609651620016.pth'))
model.load_state_dict(torch.load('trained_model_theta0.01_new_0.00015815345250302926.pth'))

# Set the model to evaluation mode (if needed)
model.eval()
x0 = torch.tensor([[-0.5],[-0.0]])

x0 = x0.reshape(1,2)


import numpy as np
from scipy.signal import cont2discrete

# Define the continuous-time system matrices (A, B, C, D)
A = np.array([[0.,1.],[0.,0.]])
B = np.array([[0.],[1.]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Define the sample time (Ts)
Ts = 0.1  # Adjust the sample time as needed
# Discretize the system
A, B, _, _, _ = cont2discrete((A, B, C, D), Ts, method='zoh')

A = torch.tensor(A).float()
B = torch.tensor(B).float()
# Define state weighting matrix Q and control weighting matrix R
Q = np.eye(2)  # Identity matrix of size 2x2
R = np.array([[1.0]])

iteration = 1000
res_model = np.zeros((iteration,2)) #for saving data
model.eval()
with torch.no_grad():
    for t in range(iteration):
        if t == 0:
            xt = x0
        output = model(xt.reshape(1,2).to(device))
        if device != 'cpu':
            output = output.detach().cpu()
        p_t = torch.tensor([output[0,0],output[0,1],output[0,1],output[0,2]]).reshape(2,2)
        u_t = output[0,3].reshape(1,1)
        print('p matrix:\n {}'.format(p_t))
        print('u: {}'.format(u_t))
        res_model[t,:] = xt.reshape(1,2)
        xt = xt.reshape(2,1)
        v_dot = xt.T @ (p_t @ A + A.T @ p_t) @ xt + xt.T @ p_t @ B @ u_t + u_t.T @ B.T @ p_t @ xt
        print('v_dot: {}'.format(v_dot))
        xt = A @ xt + B @ u_t
        print('x(t+1): \n {}'.format(xt))
        






# caculate with lqr function

x0 = x0.numpy()
A = A.numpy()
B = B.numpy()

Q = np.eye(2)  # Identity matrix of size 2x2
R = np.array([[1.0]])

res_dlqr = np.zeros((iteration,2))
for t in range(iteration):
    if t == 0:
        xt = x0
    k, _, _ = dlqr(A, B, Q, R)
    xt = xt.reshape(2,1)
    res_dlqr[t,:] = xt.reshape(1,2)
    u_t = -k @ xt
    xt = A @ xt + B @ u_t
    # print('u----->{}'.format(u_t))       
    # print('x(t+1): \n {}'.format(xt))

plt.figure(figsize=(10, 8))
plt.subplot(2,2,1)
plt.plot(list(range(iteration)),res_model[:,0])
plt.title('NN_X1')
plt.subplot(2,2,3)
plt.plot(list(range(iteration)),res_model[:,1])
plt.title('NN_X2')
plt.subplot(2,2,2)
plt.plot(list(range(iteration)),res_dlqr[:,0])
plt.title('LQR_X1')
plt.subplot(2,2,4)
plt.plot(list(range(iteration)),res_dlqr[:,1])
plt.title('LQR_X2')
# plt.savefig('NN_LQR_compare_result.png')
plt.show()

# test_loss = 0.0
# model.eval()
# with torch.no_grad():
#     for inputs, targets in test_loader:
#         outputs = model(inputs.to(device))
#         targets = targets.view([-1,2,1])
#         loss = criterion(outputs, targets.to(device))
#         test_loss += loss.item() * inputs.size(0)

# test_loss /= len(test_loader.dataset)
# print(f'Test Loss: {test_loss:.4f}')