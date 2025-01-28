import scipy.io
import matplotlib.pyplot as plt
import numpy as np


# control number font size on axis.
# introduce latex
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "Times New Roman"


# Load the .mat file
mat = scipy.io.loadmat('drone_experiments/experiment3.mat')

# Access the data from the loaded .mat file
# Assuming the data is stored in a variable named 'data'
SS = mat['SS']
t = mat['Time'][1:,0] - mat['Time'][0,0]


# Plot the data
# t = np.linspace()
end = 1500

# X direction
plt.figure()
plt.plot(t[:end],SS[0,:end])
plt.xlabel('Time [s]')
plt.ylabel('X Position [m]')
plt.title('X Position Changes with Time.')
plt.grid(linestyle = '--')
plt.tight_layout()
plt.show()
plt.close()

# Y direction
plt.figure()
plt.plot(t[:end],SS[1,:end])
plt.xlabel('Time [s]')
plt.ylabel('Y Position [m]')
plt.title('Y Position Changes with Time.')
plt.grid(linestyle = '--')
plt.tight_layout()
plt.show()

# Z direction 
plt.plot(t[:end],SS[2,:end])
plt.xlabel('Time [s]')
plt.ylabel('Z Position [m]')
plt.title('Z Position Changes with Time.')
plt.grid(linestyle = '--')
plt.tight_layout()
plt.show()
