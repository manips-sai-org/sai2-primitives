import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('../../build/examples/22-puma-singularity/singular_values.csv')

# Extract time column
time = df.iloc[:, 0]

# Extract data columns
data = df.iloc[:, 1:]

# # Plot each data column against time
# for column in data.columns:
#     plt.plot(time, data[column], label=column)

# Divide each row of data by the first column
normalized_data = data.div(data.iloc[:, 0], axis=0)

# Plot each data column against time
for column in normalized_data.columns:
    plt.plot(time, normalized_data[column], label=column)

# Add labels and legend
plt.xlabel('Time')
plt.ylabel('Data')
plt.ylim([0, 1.2])
# plt.legend(['s0', 's1', 's2', 's3', 's4', 's5'])
plt.legend(['s0', 's1', 's2'])

# Show plot
# plt.title('Singular Values During Singular Motion')
plt.title('Singular Values Ratios During Singular Motion')
plt.grid()
plt.show()
