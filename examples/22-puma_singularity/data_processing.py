###
### Puma data processing 
###

import numpy as np 
import matplotlib.pyplot as plt 

# 'tab:blue'
# 'tab:orange'
# 'tab:green'
# 'tab:red'
# 'tab:purple'
# 'tab:brown'
# 'tab:pink'
# 'tab:gray'
# 'tab:olive'
# 'tab:cyan'

dq_upper = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
dq_lower = -dq_upper

dq_upper_zone_2 = dq_upper - 0.1 * 3
dq_upper_zone_1 = dq_upper - 0.1 * 5

dq_lower_zone_2 = dq_lower + 0.1 * 3
dq_lower_zone_1 = dq_lower + 0.1 * 5

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
data = np.loadtxt('../../build/examples/22-puma_singularity/puma_singularity_overhead.csv', delimiter=",", dtype=float, skiprows=1)
# data = np.loadtxt('puma_singularity_overhead.csv', delimiter=",", dtype=float, skiprows=1)
# time, current_joint_velocity, current_position, current_velocity, desired_position, current_orientation, desired_orientation,
# q_curr, q_des, safety_torques, sent_torques, command_torques
offsets = [1, 6, 6, 6, 3, 3, 3, 1]

# create dictionary of data
data_dict_keys = ['time', 'singular_values', 'singular_direction', 'desired_acceleration', 'current_position', 'desired_position', 'orientation_error', 'dot_product']
data_dict = {}

col = 0
for i in range(len(data_dict_keys)):
    end_col = col + offsets[i]
    data_dict[data_dict_keys[i]] = data[:, col:end_col]
    col += offsets[i]
    
# condition number 
condition_numbers = np.zeros_like(data_dict['singular_values'])
for i in range(np.shape(data_dict['singular_values'])[0]):
    for j in range(6):
        condition_numbers[i, j] = data_dict['singular_values'][i, j] / data_dict['singular_values'][i, 0]
    
# Plots
fig, ax = plt.subplots(3, 1)

for i in range(3):
    ax[0].plot(data_dict['time'][10:], data_dict['current_position'][10:, i], color=colors[i])

for i in range(3):  
    ax[0].plot(data_dict['time'][10:], data_dict['desired_position'][10:, i], color=colors[i], linestyle='--')
    
ax[0].legend(['X', 'Y', 'Z'])
ax[0].grid()
ax[0].set_title('Actual vs. Desired Position')
ax[0].set_ylabel('Position (m)')

ax[1].plot(data_dict['time'], data_dict['orientation_error'])
# ax[1].legend(['X', 'Y', 'Z'])
ax[1].grid()
ax[1].set_title('Orientation Error')
ax[1].set_ylim([-0.3, 0.3])
ax[1].set_ylabel('Orientation Error (rad)')

ax[2].plot(data_dict['time'], condition_numbers)
ax[2].axhline(y=6e-2, linestyle='--', color='tab:green')
ax[2].axhline(y=6e-3, linestyle='--', color='tab:red')
ax[2].set_title('Singular Value Ratios $\sigma / \sigma_{max}$')
ax[2].set_ylim([0, 0.3])
ax[2].grid()
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Relative Magnitude')

# Plots
# plt.figure()
# for i in range(3):
#     plt.plot(data_dict['time'], data_dict['current_position'][:, i], color=colors[i])
# for i in range(3):
#     plt.plot(data_dict['time'], data_dict['desired_position'][:, i], color=colors[i], linestyle='--')
# plt.xlabel('Time (s)')
# plt.ylabel('Position (m)')
# plt.title('Actual vs. Desired Position')
# plt.grid()
# plt.legend(['X', 'Y', 'Z'])

# plt.figure()
# for i in range(3):
#     plt.plot(data_dict['time'], data_dict['orientation_error'][:, i], color=colors[i])
# plt.xlabel('Time (s)')
# plt.ylabel('Angle (rad)')
# plt.title('Orientation Error')
# plt.grid()
# plt.legend(['X', 'Y', 'Z'])

# plt.figure()
# plt.plot(data_dict['time'], data_dict['dot_product'])
# plt.xlabel('Time (s)')
# plt.ylabel('Magnitude (dot product)')
# plt.grid()
# plt.title('Unit Mass Acceleration and Singular Direction Alignment')

plt.tight_layout()
plt.show()

