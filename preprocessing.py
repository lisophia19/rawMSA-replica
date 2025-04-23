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
    train_split = 0.6  # Change to adapt to our amount of data (paper used 90% training 10% testing)

    # train_msa = torch.zeros((L, train_split * Y))
    # train_labels = torch.zeros((L, train_split * Y))

    # test_msa = torch.zeros((L, (1-train_split) * Y))
    # test_labels = torch.zeros((L, (1-train_split) * Y))
    
    num_residues = 200
    num_sequences = 25

    rand_seq_indices = torch.randperm(num_sequences)
    rand_res_indices = torch.randperm(num_residues)

    train_split_seq_index = int(train_split * len(rand_seq_indices))
    
    rand_seq_train_indices = rand_seq_indices[0 : train_split_seq_index]
    rand_seq_test_indices = rand_seq_indices[train_split_seq_index : ]

    # train_msa = train_msa[rand_res_train_indices][:,rand_seq_train_indices]
    # train_labels = train_labels[rand_res_train_indices][:, rand_seq_train_indices]

    # test_msa = test_msa[rand_res_test_indices][:,rand_seq_test_indices]
    # test_labels = test_labels[rand_res_test_indices][:,rand_seq_test_indices]

    train_seq_data = torch.gather(sequence_data, 1, rand_seq_train_indices)
    train_labels = torch.gather(sequence_labels, 1, rand_seq_train_indices)

    test_seq_data = torch.gather(sequence_data, 1, rand_seq_test_indices)
    test_labels = torch.gather(sequence_labels, 1, rand_seq_test_indices)

    return train_seq_data, train_labels, test_seq_data, test_labels
    

def compile_tensor(line, sequence_type):
    line = line.rstrip()
    
    count = 0
    # Represents the number of residues 
    limit = 200

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

# all_sequences, sequence_labels = map_to_integer(Path.cwd() / "collected_data" / "data.txt")
# print(all_sequences)

train_seq_data, train_labels, test_seq_data, test_labels = split_data()
print(train_seq_data)
print (train_labels)