import os
import random

def split_data(input_file, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, shuffle=False):
    assert train_ratio + valid_ratio + test_ratio == 1.0, "Ratios should sum up to 1.0"

    with open(input_file, 'r', encoding='utf-8') as file:
        data = file.readlines()

    if shuffle:
        random.shuffle(data)

    total_samples = len(data)
    train_samples = int(total_samples * train_ratio)
    valid_samples = int(total_samples * valid_ratio)
    test_samples = total_samples - train_samples - valid_samples

    train_data = data[:train_samples]
    valid_data = data[train_samples:train_samples + valid_samples]
    test_data = data[train_samples + valid_samples:]

    return train_data, valid_data, test_data

# Example usage:
input_file = '/data/yl7622/MRes/M4/my_data/HPC'
train_data, valid_data, test_data = split_data(input_file)

# Write the split data to separate files
with open('/data/yl7622/MRes/M4/my_data/train.HPC', 'w', encoding='utf-8') as file:
    file.writelines(train_data)

with open('/data/yl7622/MRes/M4/my_data/valid.HPC', 'w', encoding='utf-8') as file:
    file.writelines(valid_data)

with open('/data/yl7622/MRes/M4/my_data/test.HPC', 'w', encoding='utf-8') as file:
    file.writelines(test_data)
