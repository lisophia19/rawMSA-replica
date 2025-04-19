from pathlib import Path


def parse_file(filepath: str):
    used_orgs = set() # hashset storing the IDs of organisms that have already been included

    output_dir = Path.cwd() / "id_lists"
    output_dir.mkdir(exist_ok=True)

    pfam_id = filepath.name.split(".")[0] # PFAM ID of the organism
    pdb_filepath = output_dir / (pfam_id + "_id_list.txt") # filepath of the new id file
    
    with open(filepath, 'r') as stockholm, open(pdb_filepath, 'a') as output:
        for line in stockholm:
            if "_HUMAN" in line and "DR PDB;" in line: # only use human examples with existing PDB IDs
                org = line.split()[1].split("/")[1]
                pdb_id = line.split()[4]
                if org not in used_orgs: # if not already included, add data to file
                    used_orgs.add(org)
                    with open(pdb_filepath, "a") as file:
                        file.write(pdb_id + ";" + org + "\n")

for id_file in (Path.cwd() / "id_lists").iterdir(): # remove existing files from the id_lists folder
    Path.unlink(id_file)
for data_file in (Path.cwd() / "stockholm_data").iterdir(): # add new id files to the id_lists folder
    parse_file(data_file)



