from collections import defaultdict
from pathlib import Path

def parse_stockholm_with_ss(file_path):
    sequences = defaultdict(str)
    sec_structs = defaultdict(str)

    with open(file_path, 'r') as f:
        previous_line = ""
        for line in f:        
            if not line or line.startswith('#=GC'):
                continue
            
            if line.startswith('#=GR'):
                sequence = previous_line.split()
                print(sequence)

                ss = line.split()
                print(ss)

                if ss[2] == 'SS':

                    seq_id = sequence[0]
                    actual_sequence = sequence[1]
                    ss_sequence = ss[3]

                    sequences[seq_id] += actual_sequence
                    sec_structs[seq_id] += ss_sequence
            else:
                previous_line = line


    return sequences, sec_structs

def write_fasta_with_ss(sequences, sec_structs, output_path):
    with open(output_path, 'a') as out:
        for seq_id, sequence in sequences.items():
            out.write(f">{seq_id}\n{sequence}\n")
            if seq_id in sec_structs:
                out.write(f"<{seq_id}_SS\n{sec_structs[seq_id]}\n")

def parse_all_files():
    for data_file in (Path.cwd() / "collected_data").iterdir():
        Path.unlink(data_file)
    for data_file in (Path.cwd() / "stockholm_data").iterdir():
        seqs, ss = parse_stockholm_with_ss(data_file)
        write_fasta_with_ss(seqs, ss, Path.cwd() / "collected_data" / "data")

parse_all_files()
