from collections import defaultdict
from pathlib import Path
import os

def parse_stockholm_with_ss(file_path):
    body_sequences = defaultdict(str)

    master_sequences = defaultdict(str)
    sec_structs = defaultdict(str)

    with open(file_path, 'r') as f:
        previous_line = ""
        for line in f:        
            if not line or (line.startswith('#=GC')):# and previous_line == ""):
                continue
            
            if line.startswith('#=GR'):
                sequence = previous_line.split()

                ss = line.split()

                if ss[2] == 'SS':

                    seq_id = sequence[0]
                    actual_sequence = sequence[1]
                    ss_sequence = ss[3]

                    master_sequences[seq_id] += actual_sequence
                    sec_structs[seq_id] += ss_sequence
            elif previous_line != "" and not previous_line.startswith('#'):
                sequence=previous_line.split()

                seq_id = sequence[0]
                actual_sequence = sequence[1]

                body_sequences[seq_id] += actual_sequence

            previous_line = line


    return master_sequences, sec_structs, body_sequences

def write_fasta_with_ss(sequences, sec_structs, output_path):
    with open(output_path, 'a') as out:
        for seq_id, sequence in sequences.items():
            out.write(f">{seq_id}\n{sequence}\n")
            if seq_id in sec_structs:
                out.write(f"<{seq_id}_SS\n{sec_structs[seq_id]}\n")

def write_fasta_without_ss(sequences, output_path):
    with open(output_path, 'a') as out:
        for seq_id, sequence in sequences.items():
            out.write(f">{seq_id}\n{sequence}\n")
           

def parse_all_files(parent_dir, stockholm_dir, collected_master_dir, collected_body_dir):
    for data_file in (Path.cwd() / parent_dir / collected_master_dir).iterdir():
        Path.unlink(data_file)
    for data_file in (Path.cwd() / parent_dir / collected_body_dir).iterdir():
        Path.unlink(data_file)        
    for root, _, files in os.walk(os.path.join(parent_dir, stockholm_dir)):
        for file in files:
            master_seqs, ss, body_seqs = parse_stockholm_with_ss(os.path.join(parent_dir, stockholm_dir, file))

            updated_master_name = file[0:7] + ".master.txt"
            updated_body_name = file[0:7] + ".body.txt"

            write_fasta_with_ss(master_seqs, ss, os.path.join(parent_dir, collected_master_dir, updated_master_name))
            write_fasta_without_ss(body_seqs, os.path.join(parent_dir, collected_body_dir, updated_body_name))


parse_all_files('train_data', 'stockholm_train_data', 'collected_master_sequences', 'collected_body_sequences')
