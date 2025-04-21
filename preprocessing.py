# import tensorflow as tf
import numpy as np
from pathlib import Path

letter_to_number = { 'P':1, 'U':2, 'C':3, 'A':4, 'G':5, 'S':6, 'N':7, 'B':8, 'D':9, 'E':10, 'Z':11, 'Q':12, 'R':13, 'K':14, 'H':15, 'F':16, 'Y':17, 'W':18, 'M':19, 'L':20, 'I':21, 'V':22, 'T':23, '.':24, 'X':25 }
ss_to_number = {'H': 1, 'E':2, 'T': 3, 'S': 4, 'G': 5, 'I':6, 'C':7, '.':8, '-':9}

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