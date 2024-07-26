from copy import deepcopy
from functools import partial
import math
from multiprocessing import Pool
import os
import pickle
import numpy as np
from rdkit import Chem
import pandas as pd
from dgllife.utils import WeaveAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import dgl
from dgl.data.utils import Subset
from dgl.data.utils import save_graphs, load_graphs
from collections import Counter

atom_stereo2label = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER: 3,
}

label2atom_stereo = {v: k for k, v in atom_stereo2label.items()}

bond_stereo2label = {
    Chem.rdchem.BondStereo.STEREONONE: 0,
    Chem.rdchem.BondStereo.STEREOANY: 1,
    Chem.rdchem.BondStereo.STEREOZ: 2,
    Chem.rdchem.BondStereo.STEREOE: 3,
    Chem.rdchem.BondStereo.STEREOCIS: 4,
    Chem.rdchem.BondStereo.STEREOTRANS: 5
}

atom_types = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
    'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
    'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
    'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta',
    'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs'
]


def pad_atom_distance_matrix(adm_list):
    max_size = max([adm.shape[0] for adm in adm_list])
    adm_list = [
        torch.tensor(np.pad(adm, (0, max_size - adm.shape[0]),
                            'maximum')).unsqueeze(0).long() for adm in adm_list
    ]
    return torch.cat(adm_list, dim=0)


class NotCanonicalizableSmilesException(ValueError):
    pass


def canonicalize_smiles(smi, remove_atom_mapping=False, erro_raise=False):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        if remove_atom_mapping:
            [
                atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms()
                if atom.HasProp('molAtomMapNumber')
            ]
        return Chem.MolToSmiles(mol)
    else:
        if erro_raise:
            raise NotCanonicalizableSmilesException(
                'Molecule not canonicalizable')
        return ''


def process_reaction(rxn):
    reactants, reagents, products = rxn.split('>')
    try:
        precursors = [
            canonicalize_smiles(r, True, True) for r in reactants.split(".")
        ]
        if len(reagents) > 0:
            precursors += [
                canonicalize_smiles(r, True, True) for r in reagents.split(".")
            ]
        products = [
            canonicalize_smiles(p, True, True) for p in products.split(".")
        ]
    except NotCanonicalizableSmilesException:
        return ""

    joined_precursors = canonicalize_smiles(".".join(sorted(precursors)))
    joined_products = canonicalize_smiles(".".join(sorted(products)))

    return f"{joined_precursors}>>{joined_products}"


def read_txt_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [''.join(x.strip().split()) for x in f.readlines()]


def get_adm(mol, max_distance=6):
    mol_size = mol.GetNumAtoms()
    distance_matrix = np.ones((mol_size, mol_size)) * max_distance + 1
    dm = Chem.GetDistanceMatrix(mol)
    dm[dm > 100] = -1  # remote (different molecule)
    dm[dm > max_distance] = max_distance  # remote (same molecule)
    dm[dm == -1] = max_distance + 1
    distance_matrix[:dm.shape[0], :dm.shape[1]] = dm
    return distance_matrix


def atom_to_vocab(mol, atom):
    """
    Adapted from https://github.com/tencent-ailab/grover/blob/main/grover/data/task_labels.py#L57
    
    Convert atom to vocabulary. The convention is based on atom type and bond type.
    :param mol: the molecular.
    :param atom: the target atom.
    :return: the generated atom vocabulary with its contexts.
    """
    nei = Counter()
    for a in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx())
        nei[str(bond.GetBondType())] += 1  # 这里只保留周围的键，不保留另一端的原子
    keys = nei.keys()
    keys = list(keys)
    keys.sort()
    output = atom.GetSymbol()
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])

    # The generated atom_vocab is too long?
    return output


class ReactionDataset(object):
    def __init__(self,
                 dataset_name,
                 debug=False,
                 nb_workers=None,
                 use_atom_envs_type=False) -> None:
        self.orgainc_rxn_dataset_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                         'dataset/organic_rxn_dataset/'))
        if debug: dataset_name += '_debug'
        self.nb_workers = nb_workers
        self.use_atom_envs_type = use_atom_envs_type
        self.dataset_root = os.path.join(self.orgainc_rxn_dataset_root,
                                         dataset_name)
        print('#########################################################')
        print('Preprocess {} dataset'.format(dataset_name))
        print('#########################################################')
        self.processed_path = os.path.join(self.orgainc_rxn_dataset_root,
                                           f'{dataset_name}_preprocessed')
        self.atom_types = atom_types

        node_featurizer = WeaveAtomFeaturizer(atom_types=atom_types)
        edge_featurizer = CanonicalBondFeaturizer(self_loop=True)

        all_dataset_df = pd.DataFrame()
        for dataset_flag in ['train', 'valid', 'test']:
            dataset_df = pd.read_csv(
                os.path.join(self.dataset_root, dataset_flag,
                             'rxn-smiles.csv'))
            dataset_df.loc[:, 'split'] = [dataset_flag] * len(dataset_df)
            all_dataset_df = pd.concat([all_dataset_df, dataset_df], axis=0)
        all_dataset_df = all_dataset_df.reset_index(drop=True)

        print('#########################################################')
        print('Read {} raw dataset'.format(len(all_dataset_df)))
        print('#########################################################')

        self.train_ids = all_dataset_df.index[all_dataset_df['split'] ==
                                              'train']
        self.val_ids = all_dataset_df.index[all_dataset_df['split'] == 'valid']
        self.test_ids = all_dataset_df.index[all_dataset_df['split'] == 'test']

        print('#########################################################')
        print('Train: {}\nVal: {}\nTest: {}'.format(len(self.train_ids),
                                                    len(self.val_ids),
                                                    len(self.test_ids)))
        print('#########################################################')

        if self.use_atom_envs_type:
            self._get_atom_envs_label_lib(all_dataset_df)

        self._pre_process_number_workers(
            all_dataset_df, partial(mol_to_bigraph, add_self_loop=True),
            node_featurizer, edge_featurizer)

        self._get_feat_info()

    def _get_feat_info(self):
        self.node_dim = self.reactant_graphs[0].ndata['h'].shape[-1]
        self.edge_dim = self.reactant_graphs[0].edata['e'].shape[-1]
        if not hasattr(self, 'num_atom_types'):
            if self.use_atom_envs_type:
                self.atom_envs_label_lib_path = os.path.abspath(
                    os.path.join(self.processed_path,
                                 'atom_envs_label_lib.pkl'))
                with open(self.atom_envs_label_lib_path, 'rb') as f:
                    atom_envs_labels_to_class = pickle.load(f)
                self.num_atom_types = len(atom_envs_labels_to_class)
            else:
                self.num_atom_types = len(atom_types) + 1

    def _get_atom_envs_label_lib(self, org_dataset):
        self.atom_envs_label_lib_path = os.path.abspath(
            os.path.join(self.processed_path, 'atom_envs_label_lib.pkl'))
        self.atom_envs_label_lib_exist = os.path.exists(
            self.atom_envs_label_lib_path)
        if not self.atom_envs_label_lib_exist:
            from pandarallel import pandarallel
            pandarallel.initialize(nb_workers=self.nb_workers,
                                   progress_bar=True)
            print()
            print('Generating atomic environment vocabulary library.')
            reactants, products = map(
                list,
                zip(*org_dataset['rxn_smiles'].apply(lambda x: x.split('>>'))))

            def get_atom_envs_from_one_smiles(smi):
                labels = set()
                mol = Chem.MolFromSmiles(smi)
                for atom in mol.GetAtoms():
                    labels.add(atom_to_vocab(mol, atom))
                return labels

            all_molecules = list(set(reactants + products))
            working_df = pd.DataFrame({'smiles': all_molecules})
            working_df['atom_envs_labels'] = working_df[
                'smiles'].parallel_apply(
                    lambda x: get_atom_envs_from_one_smiles(x))
            atom_envs_labels = set()
            for labels in working_df['atom_envs_labels'].tolist():
                atom_envs_labels.update(labels)
            atom_envs_labels = list(atom_envs_labels)
            atom_envs_labels.sort()
            atom_envs_labels_to_class = {
                k: v
                for v, k in enumerate(atom_envs_labels)
            }
            print('\nGet {}'.format(len(atom_envs_labels_to_class)))

            print('Saving atomic environment vocabulary library to {}'.format(
                self.atom_envs_label_lib_path))
            if not os.path.exists(
                    os.path.dirname(self.atom_envs_label_lib_path)):
                os.makedirs(os.path.dirname(self.atom_envs_label_lib_path))
            with open(self.atom_envs_label_lib_path, 'wb') as f:
                pickle.dump(atom_envs_labels_to_class, f)
        else:
            print('Atomic environment vocabulary detected.')

    def _calculate_features(self, org_dataset, mol_to_graph, node_featurizer,
                            edge_featurizer, react_or_prod):
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=self.nb_workers, progress_bar=True)
        if self.use_atom_envs_type:
            print(
                'Using atomic environment type, reading atomic environment vocabulary from {}'
                .format(self.atom_envs_label_lib_path))
            with open(self.atom_envs_label_lib_path, 'rb') as f:
                atom_envs_labels_to_class = pickle.load(f)
            self.num_atom_types = len(atom_envs_labels_to_class)
        else:
            atom_envs_labels_to_class = None
            self.num_atom_types = len(atom_types) + 1

        def calculate_features(smi):
            mol = Chem.MolFromSmiles(smi)
            fgraph = mol_to_graph(mol,
                                  node_featurizer=node_featurizer,
                                  edge_featurizer=edge_featurizer,
                                  canonical_atom_order=False)
            dgraph = get_adm(mol)
            if atom_envs_labels_to_class is None:
                atom_type_labels = torch.argwhere(
                    fgraph.ndata['h'][:, :self.num_atom_types] == 1)[:, 1]
            else:
                atom_type_labels = []
                for atom in mol.GetAtoms():
                    atom_type_labels.append(
                        atom_envs_labels_to_class[atom_to_vocab(mol, atom)])
                atom_type_labels = torch.tensor(atom_type_labels).long()
                assert fgraph.ndata['h'].shape[0] == atom_type_labels.shape[0]
            return fgraph, dgraph, atom_type_labels

        graphs, dgraphs, atom_type_labels_list = map(
            list,
            zip(*org_dataset[react_or_prod].parallel_apply(
                lambda x: calculate_features(x))))
        return graphs, dgraphs, atom_type_labels_list

    def _pre_process_number_workers(self, org_dataset, mol_to_graph,
                                    node_featurizer, edge_featurizer):

        self.react_processed_exist = os.path.exists(
            os.path.join(self.processed_path, 'reactant_processed.bin'))
        self.prod_processed_exist = os.path.exists(
            os.path.join(self.processed_path, 'product_processed.bin'))

        self.rxn_class_labels = None
        if 'label' in org_dataset.columns.tolist():
            self.rxn_class_labels = torch.from_numpy(org_dataset.label.values)

        if not self.react_processed_exist or not self.prod_processed_exist:
            org_dataset['reactants'], org_dataset['products'] = zip(
                *org_dataset['rxn_smiles'].apply(lambda x: x.split('>>')))
            if not self.react_processed_exist:
                self.reactant_graphs, self.reactant_dgraphs, self.reactant_atom_type_labels_list = self._calculate_features(
                    org_dataset, mol_to_graph, node_featurizer,
                    edge_featurizer, 'reactants')
                save_graphs(
                    os.path.join(self.processed_path,
                                 'reactant_processed.bin'),
                    self.reactant_graphs)
                with open(
                        os.path.join(self.processed_path,
                                     'reactant_dgraphs_processed.pkl'),
                        'wb') as f:
                    pickle.dump(self.reactant_dgraphs, f)
                with open(
                        os.path.join(
                            self.processed_path,
                            'reactant_atom_type_labels_processed.pkl'),
                        'wb') as f:
                    pickle.dump(self.reactant_atom_type_labels_list, f)
                del self.reactant_graphs, self.reactant_dgraphs, self.reactant_atom_type_labels_list
                print(
                    '\nDel self.reactant_graphs, self.reactant_dgraphs, self.reactant_atom_type_labels_list'
                )
            if not self.prod_processed_exist:
                self.product_graphs, self.product_dgraphs, self.product_atom_type_labels_list = self._calculate_features(
                    org_dataset, mol_to_graph, node_featurizer,
                    edge_featurizer, 'products')
                save_graphs(
                    os.path.join(self.processed_path, 'product_processed.bin'),
                    self.product_graphs)
                with open(
                        os.path.join(self.processed_path,
                                     'product_dgraphs_processed.pkl'),
                        'wb') as f:
                    pickle.dump(self.product_dgraphs, f)

                with open(
                        os.path.join(
                            self.processed_path,
                            'product_atom_type_labels_list_processed.pkl'),
                        'wb') as f:
                    pickle.dump(self.product_atom_type_labels_list, f)

            print('Reloading reactant_processed graphs from %s...' %
                  os.path.join(self.processed_path, 'reactant_processed.bin'))
            self.reactant_graphs, _ = load_graphs(
                os.path.join(self.processed_path, 'reactant_processed.bin'))
            with open(
                    os.path.join(self.processed_path,
                                 'reactant_dgraphs_processed.pkl'), 'rb') as f:
                self.reactant_dgraphs = pickle.load(f)
            with open(
                    os.path.join(self.processed_path,
                                 'reactant_atom_type_labels_processed.pkl'),
                    'rb') as f:
                self.reactant_atom_type_labels_list = pickle.load(f)

        if self.react_processed_exist:
            print('Loading reactant_processed graphs from %s...' %
                  os.path.join(self.processed_path, 'reactant_processed.bin'))
            self.reactant_graphs, _ = load_graphs(
                os.path.join(self.processed_path, 'reactant_processed.bin'))
            with open(
                    os.path.join(self.processed_path,
                                 'reactant_dgraphs_processed.pkl'), 'rb') as f:
                self.reactant_dgraphs = pickle.load(f)
            with open(
                    os.path.join(self.processed_path,
                                 'reactant_atom_type_labels_processed.pkl'),
                    'rb') as f:
                self.reactant_atom_type_labels_list = pickle.load(f)

        if self.prod_processed_exist:
            print('Loading product_processed graphs from %s...' %
                  os.path.join(self.processed_path, 'product_processed.bin'))
            self.product_graphs, _ = load_graphs(
                os.path.join(self.processed_path, 'product_processed.bin'))
            with open(
                    os.path.join(self.processed_path,
                                 'product_dgraphs_processed.pkl'), 'rb') as f:
                self.product_dgraphs = pickle.load(f)

            with open(
                    os.path.join(
                        self.processed_path,
                        'product_atom_type_labels_list_processed.pkl'),
                    'rb') as f:
                self.product_atom_type_labels_list = pickle.load(f)

    def __getitem__(self, item):
        if self.rxn_class_labels is None:
            return self.reactant_graphs[item], self.reactant_dgraphs[
                item], self.reactant_atom_type_labels_list[
                    item], self.product_graphs[item], self.product_dgraphs[
                        item], self.product_atom_type_labels_list[item]
        else:
            return self.reactant_graphs[item], self.reactant_dgraphs[
                item], self.reactant_atom_type_labels_list[
                    item], self.product_graphs[item], self.product_dgraphs[
                        item], self.product_atom_type_labels_list[
                            item], self.rxn_class_labels[item]

    def __len__(self):
        return len(self.reactant_graphs)


def get_batch_data(batch, device):
    reactants_batch_graphs, reactants_adms, products_batch_graphs, products_adms, labels = batch

    rts_bg, rts_adms, pds_bg, pds_adms = [
        x.to(device) for x in [
            reactants_batch_graphs, reactants_adms, products_batch_graphs,
            products_adms
        ]
    ]

    labels = [x.to(device) for x in labels]

    rts_node_feats, rts_edge_feats = rts_bg.ndata.pop('h'), rts_bg.edata.pop(
        'e')
    pds_node_feats, pds_edge_feats = pds_bg.ndata.pop('h'), pds_bg.edata.pop(
        'e')

    return rts_bg, rts_adms, rts_node_feats, rts_edge_feats, pds_bg, pds_adms, pds_node_feats, pds_edge_feats, labels


class ReactionDataProcessor:
    def __init__(self, nb_workers=12, device=torch.device('cpu')) -> None:
        self.device = device
        self.nb_workers = nb_workers
        self.node_featurizer = WeaveAtomFeaturizer(atom_types=atom_types)
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.mol_to_graph = partial(mol_to_bigraph, add_self_loop=True)

    def _calculate_features(self, smi):
        mol = Chem.MolFromSmiles(smi)
        fgraph = self.mol_to_graph(mol,
                                   node_featurizer=self.node_featurizer,
                                   edge_featurizer=self.edge_featurizer,
                                   canonical_atom_order=False)
        dgraph = get_adm(mol)

        return fgraph, dgraph

    def _collate_fn(self, precursors_graphs, products_graphs):
        precursors_fgraphs, precursors_dgraphs = map(list,
                                                     zip(*precursors_graphs))
        products_fgraphs, products_dgraphs = map(list, zip(*products_graphs))
        rts_bg = dgl.batch(precursors_fgraphs)
        pds_bg = dgl.batch(products_fgraphs)
        rts_bg.set_n_initializer(dgl.init.zero_initializer)
        pds_bg.set_n_initializer(dgl.init.zero_initializer)

        rts_adms = pad_atom_distance_matrix(precursors_dgraphs)
        pds_adms = pad_atom_distance_matrix(products_dgraphs)
        rts_bg, rts_adms, pds_bg, pds_adms = [x.to(self.device) for x in [rts_bg, rts_adms, pds_bg, pds_adms]]
        rts_node_feats, rts_edge_feats = rts_bg.ndata.pop(
            'h'), rts_bg.edata.pop('e')
        pds_node_feats, pds_edge_feats = pds_bg.ndata.pop(
            'h'), pds_bg.edata.pop('e')
        

        return {
            'rts_bg': rts_bg,
            'rts_adms': rts_adms,
            'rts_node_feats': rts_node_feats,
            'rts_edge_feats': rts_edge_feats,
            'pds_bg': pds_bg,
            'pds_adms': pds_adms,
            'pds_node_feats': pds_node_feats,
            'pds_edge_feats': pds_edge_feats
        }

    def reaction_to_graph(self, rxns):
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=self.nb_workers, progress_bar=True)
        
        rxns = [process_reaction(rxn) for rxn in tqdm(
            rxns,
            desc='Canonicalize Reaction',
            total=len(rxns)
            )]
        precursors, products = zip(*[x.split('>>') for x in rxns])
        num_rxn = len(rxns)
        all_molecules = precursors + products
        
        all_molecules_df = pd.DataFrame(
            {'mols': all_molecules}
        )
        
        molecules_graph_data = list(all_molecules_df['mols'].parallel_apply(lambda x:self._calculate_features(x)))
        # molecules_graph_data = [
        #     self._calculate_features(smi) for smi in tqdm(
        #         all_molecules,
        #         desc='Calculate Graph Features (precursors + products)',
        #         total=len(all_molecules))
        # ]
        precursors_graphs, products_graphs = molecules_graph_data[:
                                                                  num_rxn], molecules_graph_data[
                                                                      num_rxn:]
        return self._collate_fn(precursors_graphs, products_graphs)


class MolGraphsCollator:
    def __init__(self, perform_mask=False, mask_percent=0.15) -> None:
        self.perform_mask = perform_mask
        self.mask_percent = mask_percent

    def _mask_fn(self, graph: dgl.DGLGraph, atom_type_label):
        node_feat, edge_feat = graph.ndata.pop('h'), graph.edata.pop('e')

        mask_node_feat, mask_edge_feat = node_feat, edge_feat
        num_atoms = graph.num_nodes()
        n_mask = math.ceil(num_atoms * self.mask_percent)
        perm = np.random.permutation(num_atoms)[:n_mask]

        mask_edge_indices = []
        for a_idx in perm:
            for (src, dis) in zip(*graph.in_edges(int(a_idx))):
                assert a_idx == dis
                mask_edge_indices.append(graph.edge_ids(src, dis))
                mask_edge_indices.append(graph.edge_ids(dis, src))
        mask_edge_indices = torch.cat(mask_edge_indices).long()

        mask_node_feat[perm] = torch.zeros_like(node_feat[perm],
                                                device=node_feat.device,
                                                dtype=node_feat.dtype)
        mask_edge_feat[mask_edge_indices] = torch.zeros_like(
            edge_feat[mask_edge_indices],
            device=node_feat.device,
            dtype=node_feat.dtype)

        end_atom_type_label = torch.zeros_like(atom_type_label,
                                               device=atom_type_label.device,
                                               dtype=atom_type_label.dtype)
        end_atom_type_label.fill_(-100)
        end_atom_type_label[perm] = atom_type_label[perm]

        graph.ndata['h'] = mask_node_feat
        graph.edata['e'] = mask_edge_feat

        return graph, end_atom_type_label

    def __call__(self, data):
        if len(data[0]) == 6:  # 没有反应类型
            reactant_graphs, reactant_dgraphs, reactant_atom_type_labels, \
                product_graphs, product_dgraphs, product_atom_type_labels = map(list, zip(*data))
        elif len(data[0]) == 7:
            reactant_graphs, reactant_dgraphs, reactant_atom_type_labels, \
                product_graphs, product_dgraphs, product_atom_type_labels, rxn_class_labels = map(list, zip(*data))
        else:
            raise ValueError('dataset class is erro!')

        if self.perform_mask:
            masked_reactant_graphs, end_reactant_atom_type_labels = zip(*[
                self._mask_fn(graph, label) for graph, label in zip(
                    reactant_graphs, reactant_atom_type_labels)
            ])
            masked_product_graphs, end_product_atom_type_labels = zip(*[
                self._mask_fn(graph, label) for graph, label in zip(
                    product_graphs, product_atom_type_labels)
            ])

            reactants_batch_graphs = dgl.batch(masked_reactant_graphs)
            products_batch_graphs = dgl.batch(masked_product_graphs)
            end_reactant_atom_type_labels = torch.cat(
                end_reactant_atom_type_labels, dim=0)
            end_product_atom_type_labels = torch.cat(
                end_product_atom_type_labels, dim=0)
        else:
            reactants_batch_graphs = dgl.batch(reactant_graphs)
            products_batch_graphs = dgl.batch(product_graphs)
            end_reactant_atom_type_labels = torch.cat(
                reactant_atom_type_labels, dim=0)
            end_product_atom_type_labels = torch.cat(product_atom_type_labels,
                                                     dim=0)

        reactants_batch_graphs.set_n_initializer(dgl.init.zero_initializer)
        reactants_batch_graphs.set_e_initializer(dgl.init.zero_initializer)
        products_batch_graphs.set_n_initializer(dgl.init.zero_initializer)
        products_batch_graphs.set_e_initializer(dgl.init.zero_initializer)
        reactants_adm_lists = reactant_dgraphs
        reactants_adms = pad_atom_distance_matrix(reactants_adm_lists)
        products_adm_lists = product_dgraphs
        products_adms = pad_atom_distance_matrix(products_adm_lists)

        if len(data[0]) == 6:
            return reactants_batch_graphs, reactants_adms, products_batch_graphs, products_adms, (
                end_reactant_atom_type_labels, end_product_atom_type_labels)
        else:
            rxn_class_labels = torch.stack(rxn_class_labels)
            return reactants_batch_graphs, reactants_adms, products_batch_graphs, products_adms, (
                end_reactant_atom_type_labels, end_product_atom_type_labels,
                rxn_class_labels)


if __name__ == '__main__':

    dataset = ReactionDataset('pistachio',
                              debug=True,
                              nb_workers=12,
                              use_atom_envs_type=True)
    train_set, val_set, test_set = Subset(dataset, dataset.train_ids), Subset(
        dataset, dataset.val_ids), Subset(dataset, dataset.test_ids)

    collator = MolGraphsCollator(perform_mask=True)

    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=8,
                                  shuffle=True,
                                  collate_fn=collator,
                                  num_workers=12)
    for batch in tqdm(train_dataloader):
        reactants_batch_graphs, reactants_adms, products_batch_graphs, products_adms, labels = batch
        pass
