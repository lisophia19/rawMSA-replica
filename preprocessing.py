from Bio import SeqIO
from pathlib import Path

letter_to_number = { 'P':1, 'U':2, 'C':3, 'A':4, 'G':5, 'S':6, 'N':7, 'B':8, 'D':9, 'E':10, 'Z':11, 'Q':12, 'R':13, 'K':14, 'H':15, 'F':16, 'Y':17, 'W':18, 'M':19, 'L':20, 'I':21, 'V':22, 'T':23, '.':24, 'X':25 }


def preprocess_data(stockholm_file : str):
    output_dir = Path.cwd() / "fasta_data"
    output_dir.mkdir(exist_ok=True)

    pfam_id = stockholm_file.name.split(".")[0] # PFAM ID of the organism
    output_filepath = output_dir / (pfam_id + "_fasta") # filepath of the new file

    # fasta to integer:
    map_to_integer(output_filepath)

def map_to_integer(data_file : str):
    count = 0
    limit = 3000
    
    output_dir = Path.cwd() / "preprocessed_data"
    output_dir.mkdir(exist_ok=True)

    output_filepath = output_dir / "preprocessed_data" # filepath of the new file

    with open(data_file, "r") as data_file, open(output_filepath, "a") as output_file:
        for line in data_file:

            if count == limit:
                break

            if line.startswith('>'):
                output_file.write("\n" + line)
                continue

            index = 0
            integers = []
            while index < len(line.rstrip()):
                try:
                    number = letter_to_number[line[index]]
                    integers.append(str(number))
                except KeyError:
                    number = 25      
                index += 1
            output_file.write(" ".join(integers))

map_to_integer(Path.cwd() / "fasta-pbd_list_PF00071.txt")