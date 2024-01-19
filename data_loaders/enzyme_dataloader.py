import os
import warnings
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from tqdm.auto import tqdm
from tqdm import tqdm as top_tqdm
from collections.abc import Mapping, Sequence
from torch.utils import data as torch_data
from torchdrug import data, transforms
from torchdrug import utils
from torchdrug.core import Registry as R
from pandarallel import pandarallel


_dataset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
_structure_path = os.path.abspath(
    os.path.join(_dataset_root, 'ec_site_dataset', 'structures'))


def get_structure_sequence(pdb_file):
    try:
        mol = Chem.MolFromPDBFile(pdb_file)
        protein_sequence = Chem.MolToSequence(mol)
    except:
        protein_sequence = ''
    return protein_sequence


def multiprocess_structure_check(df, nb_workers):
    if nb_workers != 0:

        pandarallel.initialize(nb_workers=nb_workers, progress_bar=True)
        df['aa_sequence_calculated'] = df['pdb_files'].parallel_apply(
            lambda x: get_structure_sequence(x))
    else:
        top_tqdm.pandas(desc='pandas bar')
        df['aa_sequence_calculated'] = df['pdb_files'].progress_apply(
            lambda x: get_structure_sequence(x))

    return df

class MyProtein(data.Protein):
    
    
    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_molecule(cls, mol, atom_feature="default", bond_feature="default", residue_feature="default",
                      mol_feature=None, kekulize=False):
        """
        Create a protein from an RDKit object.

        Parameters:
            mol (rdchem.Mol): molecule
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            residue_feature (str, list of str, optional): residue features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        protein = data.Molecule.from_molecule(mol, atom_feature=atom_feature, bond_feature=bond_feature,
                                         mol_feature=mol_feature, with_hydrogen=False, kekulize=kekulize)
        residue_feature = cls._standarize_option(residue_feature)

        if kekulize:
            Chem.Kekulize(mol)

        residue_type = []
        atom_name = []
        is_hetero_atom = []
        occupancy = []
        b_factor = []
        atom2residue = []
        residue_number = []
        insertion_code = []
        chain_id = []
        _residue_feature = []
        last_residue = None
        atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())] + [cls.dummy_atom]
        for atom in atoms:
            pdbinfo = atom.GetPDBResidueInfo()
            number = pdbinfo.GetResidueNumber()
            code = pdbinfo.GetInsertionCode()
            type = pdbinfo.GetResidueName().strip()
            canonical_residue = (number, code, type)
            if canonical_residue != last_residue:
                last_residue = canonical_residue
                if (type not in cls.residue2id) and (type in ('HIE', 'HID', 'HIP')):
                    warnings.warn('Other forms of histidine: `%s`' % type)
                    type = "HIS"
                elif (type not in cls.residue2id) and (type not in ('HIE', 'HID', 'HIP')):
                    warnings.warn("Unknown residue `%s`. Treat as glycine" % type)
                    type = "GLY"
                residue_type.append(cls.residue2id[type])
                residue_number.append(number)
                if pdbinfo.GetInsertionCode() not in cls.alphabet2id or pdbinfo.GetChainId() not in cls.alphabet2id:
                    return None
                insertion_code.append(cls.alphabet2id[pdbinfo.GetInsertionCode()])
                chain_id.append(cls.alphabet2id[pdbinfo.GetChainId()])
                feature = []
                for name in residue_feature:
                    func = R.get("features.residue.%s" % name)
                    feature += func(pdbinfo)
                _residue_feature.append(feature)
            name = pdbinfo.GetName().strip()
            if name not in cls.atom_name2id:
                name = "UNK"
            atom_name.append(cls.atom_name2id[name])
            is_hetero_atom.append(pdbinfo.GetIsHeteroAtom())
            occupancy.append(pdbinfo.GetOccupancy())
            b_factor.append(pdbinfo.GetTempFactor())
            atom2residue.append(len(residue_type) - 1)
        residue_type = torch.tensor(residue_type)[:-1]
        atom_name = torch.tensor(atom_name)[:-1]
        is_hetero_atom = torch.tensor(is_hetero_atom)[:-1]
        occupancy = torch.tensor(occupancy)[:-1]
        b_factor = torch.tensor(b_factor)[:-1]
        atom2residue = torch.tensor(atom2residue)[:-1]
        residue_number = torch.tensor(residue_number)[:-1]
        insertion_code = torch.tensor(insertion_code)[:-1]
        chain_id = torch.tensor(chain_id)[:-1]
        if len(residue_feature) > 0:
            _residue_feature = torch.tensor(_residue_feature)[:-1]
        else:
            _residue_feature = None

        return cls(protein.edge_list, num_node=protein.num_node, residue_type=residue_type,
                   atom_name=atom_name, atom2residue=atom2residue, residue_feature=_residue_feature,
                   is_hetero_atom=is_hetero_atom, occupancy=occupancy, b_factor=b_factor,
                   residue_number=residue_number, insertion_code=insertion_code, chain_id=chain_id,
                   meta_dict=protein.meta_dict, **protein.data_dict)


class EnzymeDataset(data.ProteinDataset):
    _dataset_root = _dataset_root
    _structure_path = _structure_path

    def __init__(self,
                 path,
                 dataset_root=_dataset_root,
                 structure_path=_structure_path,
                 debug=False,
                 verbose=1,
                 nb_workers=12,
                 protein_max_length=600,
                 save_precessed=True,
                 **kwargs) -> None:

        self.path = os.path.join(os.path.dirname(dataset_root), path)
        self.structure_path = structure_path
        self.verbose = verbose
        self.save_precessed = save_precessed
        self.nb_workers = nb_workers
        self.debug = debug
        self.protein_max_length = protein_max_length

        self._init_enzyme_data_path()
        dataset_df = self.read_and_merge_dataset()

        dataset_df = self.check_structure_sequence(dataset_df)
        dataset_df = dataset_df.loc[
            dataset_df['is_alphafolddb_valid']].reset_index(drop=True)
        self.dataset_df = dataset_df

        self.uniport_sequences = dataset_df['aa_sequence'].tolist()
        self.site_labels = dataset_df['site_labels'].tolist()

        pkl_file = self.alphafolddb_processed_file.replace('.pkl.gz', '_no_rxn_filter.pkl.gz')

        if self.debug:
            pkl_file = pkl_file.replace('.pkl.gz', '_debug.pkl.gz')
        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = [
                os.path.join(self.alphafold_db_path,
                             f'AF-{id}-F1-model_v4.pdb')
                for id in dataset_df['alphafolddb-id'].tolist()
            ]
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)

        splits = dataset_df['dataset_flag'].tolist()
        self.num_samples = [
            splits.count("train"),
            splits.count("valid"),
            splits.count("test")
        ]

        self._setup_protein_transforms()

    def _setup_protein_transforms(self):
        
        if not hasattr(self, 'protein_max_length'):
            self.protein_max_length = 600
        
        transforms_func = transforms.transform.Compose([
            transforms.ProteinView(view='residue', keys="protein_graph"),
            transforms.TruncateProtein(
                max_length=self.protein_max_length, random=False, keys='protein_graph')   # 这里一定得是比self.protein_max_length大的，不然切割之后标签就没意义了, ec site数据集中取600， mcsa中取1000
        ])
        self.transform = transforms_func


    def _init_enzyme_data_path(self):
        
        self.test_remove_aegan_train = getattr(self, 'test_remove_aegan_train', False)
        structure_path = self.structure_path
        # 特别大，最好不要打开
        self.alphafold_db_path = os.path.join(structure_path,
                                              'alphafolddb_download')
        self.pdb_path = os.path.join(structure_path, 'pdb_download')

        # 特别大，最好不要打开
        self.proprecessed_alphafold_db_path = os.path.join(
            structure_path, 'alphafolddb_proprecessed')
        os.makedirs(self.proprecessed_alphafold_db_path, exist_ok=True)

        self.proprecessed_pdb_path = os.path.join(structure_path,
                                                  'pdb_proprecessed')
        os.makedirs(self.proprecessed_pdb_path, exist_ok=True)

        self.alphafolddb_processed_file = os.path.join(
            structure_path,
            f"{os.path.split(self.path)[-1]}_alphafolddb_proprecessed.pkl.gz")
        self.pdb_processed_file = os.path.join(
            structure_path,
            f"{os.path.split(self.path)[-1]}_pdb_proprecessed.pkl.gz")
        if self.test_remove_aegan_train:
            self.alphafolddb_processed_file = self.alphafolddb_processed_file.replace('.pkl.gz', '_remove_aegan_train.pkl.gz')
            self.pdb_processed_file = self.pdb_processed_file.replace('.pkl.gz', '_remove_aegan_train.pkl.gz')

    def read_and_merge_dataset(self):
        train_df = self.get_merged_df('train')
        valid_df = self.get_merged_df('valid')
        test_df = self.get_merged_df('test')

        all_df = pd.concat([train_df, valid_df,
                            test_df]).reset_index(drop=True)
        return all_df

    def check_structure_sequence(self, df, soft_check=False):
        
        self.test_remove_aegan_train = getattr(self, 'test_remove_aegan_train', False)
        
        self.structure_check_file = os.path.join(self.path,
                                                 'structure_check_file.pkl')
        if self.debug:
            self.structure_check_file = self.structure_check_file.replace(
                '.pkl', '_debug.pkl')
        if self.test_remove_aegan_train:
            self.structure_check_file = self.structure_check_file.replace(
                '.pkl', '_remove_aegan_train.pkl')
        if not os.path.exists(self.structure_check_file):
            print('Checking sequence from structure files ...')

            check_df = df[['pdb_files']]
            check_df = check_df.drop_duplicates(
                subset=['pdb_files']).reset_index(drop=True)

            check_df = multiprocess_structure_check(check_df,
                                                    nb_workers=self.nb_workers)

            df = pd.merge(df, check_df, on=['pdb_files'], how='left')
            
            if not soft_check:
                df['is_alphafolddb_valid'] = (
                    df['aa_sequence_calculated'] == df['aa_sequence'])
            else:
                
                def sequence_soft_check(sequence_A, sequence_B):
                    # rdkit会将质子化/去质子化情况下的组氨酸（H）解析为其他（X）
                    # 此比较程序会忽略H->X的转变
                    modified_sequence_A = sequence_A.replace('H', 'X')
                    modified_sequence_B = sequence_B.replace('H', 'X')
                    
                    return modified_sequence_A == modified_sequence_B

                df['is_alphafolddb_valid'] = df.apply(lambda row:sequence_soft_check(row['aa_sequence_calculated'], row['aa_sequence']), axis=1)

            structure_check_df = df[[
                'aa_sequence_calculated', 'is_alphafolddb_valid'
            ]]
            structure_check_df.to_pickle(self.structure_check_file)

        else:
            print('Loading structure check file ...')
            structure_check_df = pd.read_pickle(self.structure_check_file)

            df = pd.concat([df, structure_check_df], axis=1)

        count = len(df)
        # df = df.loc[df['is_alphafolddb_valid']].reset_index(drop=True)
        valid_count = int(df['is_alphafolddb_valid'].sum())

        print(
            '\nAA_sequence from structure file valid {}/{}, {:.2f}%\n'.format(
                valid_count, count, 100 * valid_count / count))

        return df

    def calculate_active_sites(self, site_label, sequence_length):
        site_label = eval(site_label)  # Note: Site label starts from 1
        active_site = torch.zeros((sequence_length, ))
        for one_site in site_label:
            if len(one_site) == 1:
                active_site[one_site[0] - 1] = 1
            elif len(one_site) == 2:
                b, e = one_site
                site_indices = [k - 1 for k in range(b, e+1)]
                # site_indices = [k - 1 for k in range(b, e)]
                active_site[site_indices] = 1
            else:
                raise ValueError(
                    'The label of active site is not standard !!!')
        return active_site

    def get_item(self, index):

        _, pdb_name = os.path.split(self.pdb_files[index])
        pdb_id = pdb_name.split('.')[0]
        pdb_file = os.path.join(self.alphafold_db_path, f'{pdb_id}.pdb')
        proprecessed_pdb_file = os.path.join(
            self.proprecessed_alphafold_db_path, f'{pdb_id}.pkl')

        if getattr(self, "lazy", False):

            if not os.path.exists(proprecessed_pdb_file):
                protein = data.Protein.from_pdb(pdb_file, self.kwargs)
                if self.save_precessed:
                    torch.save(protein, proprecessed_pdb_file)
            else:
                protein = torch.load(proprecessed_pdb_file)
        else:
            protein = self.data[index].clone()

        # if hasattr(protein, "residue_feature"):
        #     sequence = protein.to_sequence()
        #     if sequence != self.uniport_sequences[index]:
        #         raise ValueError()

        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()

        item = {"protein_graph": protein}
        if self.transform:
            item = self.transform(item)
        active_site = self.calculate_active_sites(
            site_label=self.site_labels[index],
            sequence_length=len(self.uniport_sequences[index]))
        item["targets"] = active_site
        return item

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits

    def get_merged_df(self, dataset_flag):

        df = pd.DataFrame()

        self.test_remove_aegan_train = getattr(self, 'test_remove_aegan_train', False)
        
        if (dataset_flag == 'test') and self.test_remove_aegan_train:
            data_path = os.path.join(self.path, 'test_dataset_rm_aegan_train_same')
        else:
             data_path = os.path.join(self.path, f'{dataset_flag}_dataset')
        csv_fnames = os.listdir(data_path)
        if self.verbose:
            csv_fnames = tqdm(csv_fnames, total=len(csv_fnames))
        for fname in csv_fnames:
            read_df = pd.read_csv(os.path.join(data_path, fname))
            df = pd.concat([df, read_df])
            if self.debug:
                if len(df) >= 10:
                    df = df[:10]
                    break
        df['alphafolddb-id'] = df['alphafolddb-id'].apply(
            lambda x: x.split(';')[0])
        df['pdb_files'] = df['alphafolddb-id'].apply(lambda x: os.path.join(
            self.alphafold_db_path, f'AF-{x}-F1-model_v4.pdb'))
        df.loc[:, 'dataset_flag'] = dataset_flag
        return df


def enzyme_dataset_graph_collate(batch):
    """
    Convert any list of same nested container into a container of tensors.

    For instances of :class:`data.Graph <torchdrug.data.Graph>`, they are collated
    by :meth:`data.Graph.pack <torchdrug.data.Graph.pack>`.

    Parameters:
        batch (list): list of samples with the same nested container
    """
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, data.Graph):
        return elem.pack(batch)
    elif isinstance(elem, Mapping):
        return {
            key: enzyme_dataset_graph_collate([d[key] for d in batch])
            for key in elem
        }
    elif isinstance(elem, Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'Each element in list of batch should be of equal size')
        return [
            enzyme_dataset_graph_collate(samples) for samples in zip(*batch)
        ]

    raise TypeError("Can't collate data with type `%s`" % type(elem))


def check_function(batch):
    assert torch.stack([
        batch['graph'][i].num_residue for i in range(len(batch['graph']))
    ]).sum().item() == batch['targets'].size(0)


if __name__ == '__main__':
    # dataset = EnzymeDataset(path='dataset/ec_site_dataset/uniprot_ecreact_merge_dataset_limit_10000', debug=True, verbose=1)

    # train_set, valid_set, test_set = dataset.split()
    # train_set[5]
    # valid_set[5]

    dataset = EnzymeDataset(
        path=
        'dataset/ec_site_dataset/uniprot_ecreact_merge_dataset_limit_10000',
        save_precessed=False,
        debug=True,
        verbose=1,
        lazy=True,
        nb_workers=12)

    train_set, valid_set, test_set = dataset.split()

    for i in tqdm(range(10)):
        train_set[i]
        valid_set[i]
        test_set[i]

    # for i in tqdm(range(len(train_set))):
    #     train_set[i]

    train_loader = torch_data.DataLoader(
        train_set,
        batch_size=16,
        collate_fn=enzyme_dataset_graph_collate,
        num_workers=4)
    for batch_data in tqdm(train_loader):
        check_function(batch_data)

    pass