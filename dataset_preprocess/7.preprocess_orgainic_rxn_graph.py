from argparse import ArgumentParser
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loaders.rxn_dataloader import ReactionDataset


def main(args):
    debug = args.debug
    use_atom_envs_type = args.use_atom_envs_type
    nb_workers = args.num_workers
    uspto_dataset = ReactionDataset('uspto', debug=debug, nb_workers=nb_workers, use_atom_envs_type=use_atom_envs_type)
    del uspto_dataset
    pistachio_dataset = ReactionDataset('pistachio', debug=debug, nb_workers=nb_workers, use_atom_envs_type=use_atom_envs_type)
    del pistachio_dataset
    uspto_rxnmapper_dataset = ReactionDataset('uspto_rxnmapper', debug=debug, nb_workers=nb_workers, use_atom_envs_type=use_atom_envs_type)
    del uspto_rxnmapper_dataset
    
    print('All Done!')


if __name__ == '__main__':
    parser = ArgumentParser('Preprocess rxn dataset arguements')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--use_atom_envs_type', action="store_true")
    parser.add_argument('--num_workers',
                        type=int,
                        default=14,
                        help='Number of processes for data loading')
    args = parser.parse_args()
    main(args)