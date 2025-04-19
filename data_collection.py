import requests
from io import StringIO

dssp_base_url = "https://pdb-redo.eu/dssp/get?format=dssp&_csrf=00Uptl0u9vPJvcig4k6mjw"

def pull_individual_dssp_data(pdb_id:str):
    response = requests.get(f'{dssp_base_url}&pdb-id={pdb_id}')

    if response.status_code == 200:
        print("Request successful!")
        # # print(response.content)
        # path = f'{pdb_id}-dssp-file.txt'
        # dssp_file = open(path, 'a')
        # dssp_file.write(str(response.content))
        # dssp_file.close()
        # return path
        return response.content.decode()
    # else:
    #     raise Exception("Failed to request")



# def organize_parsed_data(curr_fasta_file, pdb_id, species_name):
    
def extract_fasta_and_ss(dssp_data, start_residue, end_residue):
    sequence = []
    structure = []

    dssp_data_io = StringIO(dssp_data)

    next_line = False
    for line in dssp_data_io:         

        if line.startswith('  #  RESIDUE'):  # Skip comment or header lines
            next_line = True
            continue
        if next_line:
            amino_acid = line[13:14]  # Adjust the index if needed based on your DSSP format
            if amino_acid == "!":
                print(line)
                continue
            # print(line)
            if int(line[7:10]) >= start_residue and int(line[7:10]) <= end_residue:
                # Each line corresponds to a residue, the second column should contain the AA
                sequence.append(amino_acid)
                # print(amino_acid)
                ss = line[16:17]
                if ss == " ":
                    ss = "-"
                structure.append(ss)    

    fasta_sequence = ''.join(sequence)
    structure_sequence = ''.join(structure)
    return fasta_sequence, structure_sequence


# Steps:
#   1. Reads the data from the PDB entries of format PDB;SPECIES
#   2. Opens file for both the PDB entries and the FASTA format
#       - PDB entries are used to pull SS data ==> make GET request (1), parse DSSP data (2)
#       - FASTA entries are used to pull the proper sequence
#   3. Write Species\nFASTA\nSS
def read_pdb_entries(family_name_file: str):
    pdb_entries = open(f'id_lists/{family_name_file}', 'r')

    for line in pdb_entries:
        parsed_data = open(f'pulled-pdb-files/fasta-{family_name_file}', 'a')

        print(line)
        # Each line corresponds to a PDB id:
        line = line.split(';')
        pdb_id = line[0]

        residues = line[1].split('-')
        start_residue = int(residues[0])
        end_residue = int(residues[1])


        dssp_data = pull_individual_dssp_data(pdb_id)
        fasta_data, ss_data = extract_fasta_and_ss(dssp_data, start_residue, end_residue)

        # Write fasta_data first, then ss_data

        parsed_data.write(f'{fasta_data}\n')
        parsed_data.write(f'{ss_data}\n\n')

        parsed_data.close()
    
    pdb_entries.close()


read_pdb_entries('pbd_list_PF00069.txt')
read_pdb_entries('pbd_list_PF00071.txt')

