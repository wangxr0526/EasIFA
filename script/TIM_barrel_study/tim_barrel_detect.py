from Bio.PDB import PDBParser, PPBuilder, DSSP
import os


def is_tim_barrel(pdb_file):

    if not os.path.isfile(pdb_file):
        raise FileNotFoundError(f"The file {pdb_file} does not exist.")

    parser = PDBParser()
    structure = parser.get_structure("PDB", pdb_file)

    model = structure[0]
    dssp = DSSP(model, pdb_file)

    for chain in model:
        ss_sequence = ""
        for res in chain:
            if res.id[0] == " ":
                try:

                    dssp_key = (chain.id, res.id)
                    ss = dssp[dssp_key][1]
                    ss_sequence += ss
                except KeyError:

                    continue

        if ss_sequence.count("H") >= 8 and ss_sequence.count("E") >= 8:
            return True
    return False


if __name__ == "__main__":

    pdb_path = "test_data"
    pdb_files = [
        os.path.abspath(os.path.join(pdb_path, x)) for x in os.listdir("test_data")
    ]
    print("\n" + "#" * 10 + "\n")
    for pdb_file in pdb_files:
        results = is_tim_barrel(pdb_file)
        print("{} have TIM Barrel? {}\n".format(pdb_file, results))
