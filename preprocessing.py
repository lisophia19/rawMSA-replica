# import tensorflow as tf
import torch
import numpy as np
from pathlib import Path
import os
from collections import defaultdict


# train_seq_dict = dict()
# # train_labels_dict = dict()
# test_seq_dict = dict()
# test_labels_dict = dict()

letter_to_number = { 'P':1, 'U':2, 'C':3, 'A':4, 'G':5, 'S':6, 'N':7, 'B':8, 'D':9, 'E':10, 'Z':11, 'Q':12, 'R':13, 'K':14, 'H':15, 'F':16, 'Y':17, 'W':18, 'M':19, 'L':20, 'I':21, 'V':22, 'T':23, '.':24, 'X':25 }
ss_to_number = {'H': 1, 'E':2, 'T': 3, 'S': 4, 'G': 5, 'I':6, 'C':7, '.':8, '-':9}
    

def batch_train_data(train_seq_dict, batch_num : int, file_id : str):
    batch_size = 14
    train_seq_data = train_seq_dict[file_id]
    #train_labels = train_labels_dict[file_name]

    seq_batch = train_seq_data[batch_size * batch_num : batch_size * (batch_num + 1)][:]
    #labels_batch = train_labels[batch_size * batch_num : batch_size * (batch_num + 1)][:]

    return seq_batch


def batch_testdata(test_seq_dict, batch_num : int, file_id : str):
    batch_size = 14
    test_seq_data = test_seq_dict[file_id]
    #train_labels = train_labels_dict[file_name]

    seq_batch = test_seq_data[batch_size * batch_num : batch_size * (batch_num + 1)][:]
    #labels_batch = train_labels[batch_size * batch_num : batch_size * (batch_num + 1)][:]

    return seq_batch

def compile_tensor(line, sequence_type):
    line = line.rstrip()
    
    count = 0
    # Represents the number of residues 
    limit = 200

    current_seq = np.zeros(limit)

    line_len = len(line)

    while count < limit:
        if count >= line_len:
            current_seq[count] = 25 if sequence_type == 'SEQUENCE' else 9
        else:
            try:
                number = letter_to_number[line[count]] if sequence_type == 'SEQUENCE' else ss_to_number[line[count]]
            except KeyError:
                number = 25 if sequence_type == 'SEQUENCE' else 9

            current_seq[count] = number
        count += 1

    return current_seq


def map_to_integer(data_file : str):
    
    all_sequences = []
    sequence_labels = []

    sequence_type = 'SEQUENCE'
    with open(os.path.join(data_file), 'r') as unprocessed_data:
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
    
    return all_sequences, sequence_labels


def gather_master_sequences(all_files : list[str], data_type = "train"):
    master_seq_dict = dict()

    if len(all_files) == 0:
        for file_name in os.listdir(os.path.join(f"{data_type}_data", "collected_master_sequences")):
            file_id = file_name[0:7]

            file_path = os.path.join(f"{data_type}_data", "collected_master_sequences", file_name)

            master_seq, seq_labels = map_to_integer(file_path)

            # print(len(master_seq))

            master_seq_dict[file_id] = (master_seq, seq_labels)
    else:
        for file_name in all_files:
            file_id = file_name[0:7]

            file_path = os.path.join(f"{data_type}_data", "collected_master_sequences", file_name)

            master_seq, seq_labels = map_to_integer(file_path)
            master_seq_dict[file_id] = (master_seq, seq_labels)

            # print(master_seq_dict[file_id])


    return master_seq_dict

def gather_body_sequences():
    min_number_body_seq = -1

    train_seq_dict = dict()
    test_seq_dict = dict()
    #training data collection
    for data_file in os.listdir(os.path.join("train_data","collected_body_sequences")):
        file_path = os.path.join("train_data","collected_body_sequences", data_file)
        sequence_data, _ = map_to_integer(file_path)
        sequence_data_tensor = torch.tensor(sequence_data)

        file_id = data_file[0:7]
        train_seq_dict[file_id] = sequence_data_tensor


        if min_number_body_seq == -1:
            min_number_body_seq = len(sequence_data)
        else:
            min_number_body_seq = min(min_number_body_seq, len(sequence_data))

    #testing data collection
    # for data_file in os.listdir(os.path.join("test_data","collected_body_sequences")):
    #     file_path = os.path.join("test_data","collected_body_sequences", data_file)
    #     sequence_data, _ = map_to_integer(file_path)
    #     sequence_data = torch.tensor(sequence_data)

    #     file_id = data_file[0:7]
    #     test_seq_dict[file_id] = sequence_data

    return train_seq_dict, test_seq_dict, min_number_body_seq

gather_master_sequences(["PF00018.master.txt"])
# train_seq_dict, test_seq_dict, min_number_body_seq = gather_body_sequences()
# min_num_batches = min_number_body_seq // 15
# print(min_number_body_seq)

