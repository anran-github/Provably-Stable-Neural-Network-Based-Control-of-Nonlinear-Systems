import scipy.io
import matplotlib.pyplot as plt
import numpy as np


# control number font size on axis.
# introduce latex
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "Times New Roman"


# Load the .mat file
mat_LQR = scipy.io.loadmat('drone_experiments/NN_LQR/LQR_experiment2.mat')
mat_NN = scipy.io.loadmat('drone_experiments/NN_LQR/NN_experiment1.mat')

# Access the data from the loaded .mat file
# Assuming the data is stored in a variable named 'data'
SS_LQR = mat_LQR['SS']
SS_NN = mat_NN['SS']
t_LQR = mat_LQR['Time'][1:,0] - mat_LQR['Time'][0,0]
t_NN = mat_NN['Time'][1:,0] - mat_NN['Time'][0,0]

# Plot the data
# t = np.linspace()
start = 1090
start_NN = 1100
end = 2400
end_NN = 1965
# X direction
plt.subplot(311)
plt.plot(t_LQR[start:end]-t_LQR[start],SS_LQR[0,start:end],label='LQR')
plt.plot(t_NN[start_NN:end_NN]-t_NN[start_NN],SS_NN[0,start_NN:end_NN],label='NN Model')
plt.legend()
plt.ylabel('X Position [m]')
plt.title('Drone Position Changes with Time.')
plt.grid(linestyle = '--')


# Y direction
plt.subplot(312)
plt.plot(t_LQR[start:end]-t_LQR[start],SS_LQR[1,start:end],label='LQR')
plt.plot(t_NN[start_NN:end_NN]-t_NN[start_NN],SS_NN[1,start_NN:end_NN],label='NN Model')
plt.ylabel('Y Position [m]')
plt.legend()
plt.grid(linestyle = '--')



# Z direction 
plt.subplot(313)
plt.plot(t_LQR[start:end]-t_LQR[start],SS_LQR[2,start:end],label='LQR')
plt.plot(t_NN[start_NN:end_NN]-t_NN[start_NN],SS_NN[2,start_NN:end_NN],label='NN Model')
plt.legend()
# plt.ylim((1.5,2))
plt.xlabel('Time [s]')
plt.ylabel('Z Position [m]')
plt.grid(linestyle = '--')


# plt.savefig('drone_experiments/NN_LQR/Drone_NN_LQR.png',dpi=500)
plt.show()


# Analysis control effort.
control_lqr = mat_LQR['CI']
control_nn = mat_NN['CI']

for i in range(3):
    if i == 2:
        i += 1
    # print(i)
    # control_lqr = controls_lqr[i,:]
    # control_nn = controls_nn[i,:]
    lqr_ctrl_sum = np.sum(np.abs(control_lqr[i,start:end]))
    nn_ctrl_sum = np.sum(np.abs(control_nn[i,start:end]))

    plt.subplot(211)
    plt.plot(t_LQR[start:end]-t_LQR[start],control_lqr[i,start:end],label='LQR')
    _,x = plt.xlim()
    y,_ = plt.ylim()
    plt.annotate(f'Integrate Value:{lqr_ctrl_sum:.1f}',[x/3,y+0.01],bbox=dict(boxstyle="round", fc="none", ec="gray"))
    plt.legend()
    plt.ylabel('LQR Input')
    plt.grid()
    plt.title('Control Signal Changes with time')
    plt.subplot(212)
    plt.plot(t_LQR[start:end]-t_LQR[start],control_nn[i,start:end],label='NN')
    _,x = plt.xlim()
    y,_ = plt.ylim()
    plt.annotate(f'Integrate Value:{nn_ctrl_sum:.1f}',[x/3,y+0.01],bbox=dict(boxstyle="round", fc="none", ec="gray"))
    plt.grid()
    plt.ylabel('NN Input')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.show()
    plt.close()
