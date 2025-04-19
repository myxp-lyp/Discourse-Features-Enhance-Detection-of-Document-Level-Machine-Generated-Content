import numpy as np
from sklearn.model_selection import train_test_split

# Load the data from a text file (one feature per line, as a string)
with open("/data/yl7622/MRes/plegbench_data/MGC", "r") as file:
    data = file.readlines()

# Convert list of strings to numpy array
data = np.array(data)

# Split the data into training and temporary data (80% training, 20% temporary)
train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)

# Split the temporary data into validation and testing data (50% validation, 50% testing of the remaining 20%)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save the data to files
with open("/data/yl7622/MRes/plegbench_data/MGC.train", "w") as file:
    file.writelines(train_data)

with open("/data/yl7622/MRes/plegbench_data/MGC.valid", "w") as file:
    file.writelines(val_data)

with open("/data/yl7622/MRes/plegbench_data/MGC.test", "w") as file:
    file.writelines(test_data)

# Optionally, you can print the sizes to verify
print("Training data size:", len(train_data))
print("Validation data size:", len(val_data))
print("Testing data size:", len(test_data))
