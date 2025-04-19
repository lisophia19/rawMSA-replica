from Bio import SeqIO

letter_to_number = { 'P':1, 'U':2, 'C':3, 'A':4, 'G':5, 'S':6, 'N':7, 'B':8, 'D':9, 'E':10, 'Z':11, 'Q':12, 'R':13, 'K':14, 'H':15, 'F':16, 'Y':17, 'W':18, 'M':19, 'L':20, 'I':21, 'V':22, 'T':23, '-':24, 'X':25 }
# dataset_dir = '../dataset/test'
dataset_file = 'original/kinase.fasta'



def map_to_integer(seq : str):
    count = 0
    limit = 3000

    fasta_format = open(dataset_file, 'r')
    # int_format = open('dataset/kinase_preprocessed.txt', 'a')

    for line in fasta_format:

        if count == limit:
            break

        if line.startswith('>'):
            count += 1
            continue

        index = 0
        while index < len(line.rstrip()):
            # First print for now, then try writing
            try:
                number = letter_to_number[line[index]]
            except KeyError:
                number = 25      
        
            index += 1

            # Prints current number 
            print(str(number), end=';')
        print(f'\n{index}\n')

    fasta_format.close()

map_to_integer('Put path in the future')