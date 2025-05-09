import torch
import numpy as np
import os
import random

# global dicts for converion
letter_to_number = { 'P':1, 'U':2, 'C':3, 'A':4, 'G':5, 'S':6, 'N':7, 'B':8, 'D':9, 'E':10, 'Z':11, 'Q':12, 'R':13, 'K':14, 'H':15, 'F':16, 'Y':17, 'W':18, 'M':19, 'L':20, 'I':21, 'V':22, 'T':23, '.':24, 'X':25 }
ss_to_number = {'H': 1, 'S':2, 'C': 3, '-': 4}

def batch_train_data(train_seq_dict, batch_num : int, file_id : str):
    batch_size = 14
    train_seq_data = train_seq_dict[file_id]

    seq_batch = train_seq_data[batch_size * batch_num : batch_size * (batch_num + 1)][:]

    return seq_batch


def batch_testdata(test_seq_dict, batch_num : int, file_id : str):
    batch_size = 14
    test_seq_data = test_seq_dict[file_id]

    seq_batch = test_seq_data[batch_size * batch_num : batch_size * (batch_num + 1)][:]

    return seq_batch

def compile_tensor(line, sequence_type):
    line = line.rstrip()
    
    count = 0
    # Represents the number of residues we want to analyze for each sequence 
    limit = 200

    current_seq = np.zeros(limit)

    line_len = len(line)

    while count < limit:
        if count >= line_len:
            current_seq[count] = 25 if sequence_type == 'SEQUENCE' else 4
        else:
            try:
                number = letter_to_number[line[count]] if sequence_type == 'SEQUENCE' else ss_to_number[line[count]]
            except KeyError:
                number = 25 if sequence_type == 'SEQUENCE' else 4

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

def shuffle_data(sequences, labels):
    combined_list = list(zip(sequences, labels))
    random.shuffle(combined_list) #shuffle seq and labels tgtr
    sequences_shuffled, labels_shuffled = zip(*combined_list)

    return list(sequences_shuffled), list(labels_shuffled)


def gather_master_sequences(all_files : list[str], data_type = "train"):
    master_seq_dict = dict()

    if len(all_files) == 0:
        for file_name in os.listdir(os.path.join(f"{data_type}_data", "collected_master_sequences")):
            file_id = file_name[0:7]

            file_path = os.path.join(f"{data_type}_data", "collected_master_sequences", file_name)

            master_seq, seq_labels = map_to_integer(file_path)
            master_seq, seq_labels = shuffle_data(master_seq, seq_labels)

            master_seq_dict[file_id] = (master_seq, seq_labels)
    else:
        for file_name in all_files:
            file_id = file_name[0:7]

            file_path = os.path.join(f"{data_type}_data", "collected_master_sequences", file_name)

            master_seq, seq_labels = map_to_integer(file_path)
            master_seq, seq_labels = shuffle_data(master_seq, seq_labels)

            master_seq_dict[file_id] = (master_seq, seq_labels)

    return master_seq_dict

def gather_body_sequences():
    train_seq_dict = dict()
    test_seq_dict = dict()
    val_seq_dict = dict()
    #training data collection
    for data_file in os.listdir(os.path.join("train_data","collected_body_sequences")):
        file_path = os.path.join("train_data","collected_body_sequences", data_file)
        sequence_data, _ = map_to_integer(file_path)
        random.shuffle(sequence_data)
        sequence_data_tensor = torch.tensor(sequence_data)

        file_id = data_file[0:7]
        train_seq_dict[file_id] = sequence_data_tensor

    #testing data collection
    for data_file in os.listdir(os.path.join("test_data","collected_body_sequences")):
        file_path = os.path.join("test_data","collected_body_sequences", data_file)
        sequence_data, _ = map_to_integer(file_path)
        random.shuffle(sequence_data)
        sequence_data = torch.tensor(sequence_data)

        file_id = data_file[0:7]
        test_seq_dict[file_id] = sequence_data

    # val data collection
    for data_file in os.listdir(os.path.join("val_data","collected_body_sequences")):
        file_path = os.path.join("val_data","collected_body_sequences", data_file)
        sequence_data, _ = map_to_integer(file_path)
        random.shuffle(sequence_data)
        sequence_data = torch.tensor(sequence_data)

        file_id = data_file[0:7]
        val_seq_dict[file_id] = sequence_data

    return train_seq_dict, test_seq_dict, val_seq_dict


