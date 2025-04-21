from Bio import SeqIO
from pathlib import Path

letter_to_number = { 'P':1, 'U':2, 'C':3, 'A':4, 'G':5, 'S':6, 'N':7, 'B':8, 'D':9, 'E':10, 'Z':11, 'Q':12, 'R':13, 'K':14, 'H':15, 'F':16, 'Y':17, 'W':18, 'M':19, 'L':20, 'I':21, 'V':22, 'T':23, '-':24, 'X':25 }
# dataset_dir = '../dataset/test'


def preprocess_data(stockholm_file : str):
    output_dir = Path.cwd() / "fasta_data"
    output_dir.mkdir(exist_ok=True)

    pfam_id = stockholm_file.name.split(".")[0] # PFAM ID of the organism
    output_filepath = output_dir / (pfam_id + "_fasta") # filepath of the new file
    
    # stockholm to fasta:
    records = SeqIO.parse(stockholm_file, "stockholm")
    count = SeqIO.write(records, output_filepath, "fasta")

    # fasta to integer:
    map_to_integer(output_filepath)

def map_to_integer(fasta_file : str):
    count = 0
    limit = 3000
    
    output_dir = Path.cwd() / "preprocessed_data"
    output_dir.mkdir(exist_ok=True)

    pfam_id = fasta_file.name.split("_")[0] # PFAM ID of the organism
    output_filepath = output_dir / (pfam_id + "_preprocessed") # filepath of the new file

    with open(fasta_file, "r") as fasta_file, open(output_filepath, "w") as output_file:
        for line in fasta_file:

            if count == limit:
                break

            if line.startswith('>'):
                count += 1
                output_file.write(line)
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
            output_file.write(" ".join(integers) + "\n")


for data_file in (Path.cwd() / "preprocessed_data").iterdir(): # remove existing files from the id_lists folder
    Path.unlink(data_file)
for data_file in (Path.cwd() / "stockholm_data").iterdir(): # add new id files to the id_lists folder
    preprocess_data(data_file)


# def parse_file(filepath: str):
#     used_orgs = set() # hashset storing the IDs of organisms that have already been included

#     output_dir = Path.cwd() / "id_lists"
#     output_dir.mkdir(exist_ok=True)

#     pfam_id = filepath.name.split(".")[0] # PFAM ID of the organism
#     pdb_filepath = output_dir / (pfam_id + "_id_list.txt") # filepath of the new id file
    
#     with open(filepath, 'r') as stockholm, open(pdb_filepath, 'a') as output:
#         for line in stockholm:
#             if "_HUMAN" in line and "DR PDB;" in line: # only use human examples with existing PDB IDs
#                 org = line.split()[1].split("/")[1]
#                 pdb_id = line.split()[4]
#                 if org not in used_orgs: # if not already included, add data to file
#                     used_orgs.add(org)
#                     with open(pdb_filepath, "a") as file:
#                         file.write(pdb_id + ";" + org + "\n")

# for id_file in (Path.cwd() / "id_lists").iterdir(): # remove existing files from the id_lists folder
#     Path.unlink(id_file)
# for data_file in (Path.cwd() / "stockholm_data").iterdir(): # add new id files to the id_lists folder
#     parse_file(data_file)
