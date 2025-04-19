from pathlib import Path

def parse_file(filepath: str):
    used_orgs = set()

    output_dir = Path.cwd() / "output"
    output_dir.mkdir(exist_ok=True)
    pdb_filepath = output_dir / "pbd_list.txt"

    with open(filepath, 'r') as stockholm, open(pdb_filepath, 'a') as output:
        for line in stockholm:
            if "_HUMAN" in line and "DR PDB;" in line:
                org = line.split()[1].split("/")[1]
                pbd_id = line.split()[4]
                if org not in used_orgs:
                    used_orgs.add(org)
                    with open(pdb_filepath, "a") as file:
                        file.write(pbd_id + ";" + org + "\n")


parse_file("/Users/kelleytu/Documents/CS1470/rawMSA-replica/PF00071.alignment.seed")




