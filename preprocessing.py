# import tensorflow as tf
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

letter_to_number = { 'P':1, 'U':2, 'C':3, 'A':4, 'G':5, 'S':6, 'N':7, 'B':8, 'D':9, 'E':10, 'Z':11, 'Q':12, 'R':13, 'K':14, 'H':15, 'F':16, 'Y':17, 'W':18, 'M':19, 'L':20, 'I':21, 'V':22, 'T':23, '.':24, 'X':25 }
ss_to_number = {'H': 1, 'E':2, 'T': 3, 'S': 4, 'G': 5, 'I':6, 'C':7, '.':8, '-':9}

def split_data():
    sequence_data, sequence_labels = map_to_integer(Path.cwd() / "collected_data" / "data.txt")
    sequence_data = torch.tensor(sequence_data)
    sequence_labels = torch.tensor(sequence_labels)
    print(sequence_data)
    print(sequence_labels)
    train_split = 0.6  # Change to adapt to our amount of data (paper used 90% training 10% testing)

    num_sequences = sequence_data.shape[0]

    rand_indices = torch.randperm(num_sequences)
    sequence_data = sequence_data[rand_indices, :]
    sequence_labels = sequence_labels[rand_indices, :]

    split_index = int(train_split * num_sequences)

    train_seq_data = sequence_data[:split_index][:]
    train_labels = sequence_labels[:split_index][:]
    test_seq_data = sequence_data[split_index:][:]
    test_labels = sequence_labels[split_index:][:]

    return train_seq_data, train_labels, test_seq_data, test_labels
    

def compile_tensor(line, sequence_type):
    line = line.rstrip()
    
    count = 0
    # Represents the number of residues 
    limit = 5

    current_seq = np.zeros(limit)

    line_len = len(line)

    while count < limit:
        if count >= line_len:
            current_seq[count] = 25 if sequence_type == 'SEQUENCE' else 10
        else:
            try:
                number = letter_to_number[line[count]] if sequence_type == 'SEQUENCE' else ss_to_number[line[count]]
            except KeyError:
                number = 25 if sequence_type == 'SEQUENCE' else 10

            current_seq[count] = number
        count += 1

    return current_seq


def map_to_integer(data_file : str):
    
    all_sequences = []
    sequence_labels = []

    sequence_type = 'SEQUENCE'
    with open(data_file, 'r') as unprocessed_data:
        for line in unprocessed_data:
            if line.startswith('>'):
                sequence_type = 'SEQUENCE'
                continue
            if line.startswith('<'):
                sequence_type = 'LABEL'
                continue

            if sequence_type == 'SEQUENCE':
                compiled_seq = compile_tensor(line, sequence_type)
                all_sequences.append(compiled_seq)
            else:
                compiled_seq = compile_tensor(line, sequence_type)
                sequence_labels.append(compiled_seq)
    
    return np.array(all_sequences), np.array(sequence_labels)

all_sequences, sequence_labels = map_to_integer(Path.cwd() / "collected_data" / "data.txt")
print(all_sequences.shape)

train_seq_data, train_labels, test_seq_data, test_labels = split_data()
print(train_seq_data)
print(train_labels)
print(test_seq_data)
print(test_labels)