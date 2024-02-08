import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('../../build/examples/21-singularity/singular_values.csv')

# Extract time column
time = df.iloc[:, 0]

# Extract data columns
data = df.iloc[:, 1:]

# Divide each row of data by the first column
normalized_data = data
# normalized_data = data.div(data.iloc[:, 0], axis=0)

# Divide each row of data by the last column
normalized_data = data.div(data.iloc[:, 5], axis=0)

# Plot each data column against time
for column in normalized_data.columns:
    plt.plot(time, normalized_data[column], label=column)

# Add labels and legend
plt.xlabel('Time')
plt.ylabel('Data')
# plt.legend(['s0', 's1', 's2'])
plt.legend(['e0', 'e1', 'e2', 'e3', 'e4', 'e5'])

# Show plot
# plt.title('Singular Values Ratios During Singular Motion')
plt.title('Eigen Values Ratios During Singular Motion')
plt.grid()
plt.show()
