from copy import deepcopy
from functools import partial
import itertools
import os
import pickle
import random
import warnings
import pandas as pd

import torch
import esm
import dgl
import hashlib
from rdkit import Chem
from tqdm.auto import tqdm
from tqdm import tqdm as top_tqdm
from collections.abc import Mapping, Sequence
from torch.utils import data as torch_data
import logging
from torchdrug import data, utils
from pandarallel import pandarallel
from dgllife.utils import WeaveAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph
import pkg_resources
from rxnfp.tokenization import SmilesTokenizer
import sys



sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from common.utils import cuda
from dataset_preprocess.pdb_preprocess_utils import map_active_site_for_one
from data_loaders.saprot_utils import get_struc_seq


logger = logging.getLogger(__name__)

from data_loaders.enzyme_dataloader import (
    MyProtein,
    _dataset_root,
    _structure_path,
    get_structure_sequence,
    multiprocess_structure_check,
    enzyme_dataset_graph_collate,
    EnzymeDataset,
)

from data_loaders.rxn_dataloader import (
    atom_stereo2label,
    label2atom_stereo,
    bond_stereo2label,
    atom_types,
    pad_atom_distance_matrix,
    get_adm,
    process_reaction,
)


SITE_TYPE_TO_CLASS = {"BINDING": 0, "ACT_SITE": 1, "SITE": 2}

_external_test_structure_path = os.path.abspath(
    os.path.join(_dataset_root, "rxnaamapper_external_test/splited_pdb_files")
)


def get_rxn_smiles(reaction):
    procursors, products = reaction.split(">>")
    reactants, _ = procursors.split("|")
    rxn_smiles = f"{reactants}>>{products}"
    return rxn_smiles


def multiprocess_reaction_check(df, nb_workers):
    if nb_workers != 0:

        pandarallel.initialize(nb_workers=nb_workers, progress_bar=True)
        df["canonicalize_rxn_smiles"] = df["rxn_smiles"].parallel_apply(
            lambda x: process_reaction(x)
        )
    else:
        top_tqdm.pandas(desc="pandas bar")
        df["canonicalize_rxn_smiles"] = df["rxn_smiles"].progress_apply(
            lambda x: process_reaction(x)
        )

    return df


def calculate_rxn_hash(rxn):
    hash_object = hashlib.sha256()
    hash_object.update(rxn.encode("utf-8"))
    return hash_object.hexdigest()


class ReactionFeatures(object):
    def __init__(self, reaction_features) -> None:

        (self.react_fgraph, self.react_dgraph), (self.prod_fgraph, self.prod_dgraph) = (
            reaction_features
        )

        self.react_dgraph = torch.from_numpy(self.react_dgraph)
        self.prod_dgraph = torch.from_numpy(self.prod_dgraph)

    def __repr__(self) -> str:
        repr_str = f"Reaction(reactants(fgraph:{self.react_fgraph},dgraph:{self.react_dgraph}; products(fgraph:{self.prod_fgraph}, dgraph:{self.prod_dgraph}) )"
        return repr_str

    def to(self, device):
        self.react_fgraph = self.react_fgraph.to(device)
        self.react_dgraph = self.react_dgraph.to(device)
        self.prod_fgraph = self.prod_fgraph.to(device)
        self.prod_dgraph = self.prod_dgraph.to(device)


class EnzymeReactionDataset(EnzymeDataset):
    def __init__(
        self,
        path,
        dataset_root=_dataset_root,
        structure_path=_structure_path,
        debug=False,
        verbose=1,
        nb_workers=12,
        protein_max_length=600,
        save_precessed=True,
        **kwargs,
    ) -> None:
        # super().__init__(path, dataset_root, structure_path, debug, verbose, nb_workers, save_precessed, **kwargs)

        self.path = os.path.join(os.path.dirname(dataset_root), path)
        self.structure_path = structure_path
        self.verbose = verbose
        self.save_processed = save_precessed
        self.nb_workers = nb_workers
        self.debug = debug
        self.protein_max_length = protein_max_length

        self.mol_to_graph = partial(mol_to_bigraph, add_self_loop=True)
        self.node_featurizer = WeaveAtomFeaturizer(atom_types=atom_types)
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)

        self._init_enzyme_data_path()
        dataset_df = self.read_and_merge_dataset()
        dataset_df = self.check_data(dataset_df)

        dataset_df = dataset_df.loc[dataset_df["is_valid_data"]].reset_index(drop=True)

        self.dataset_df = dataset_df
        self.uniport_sequences = dataset_df["aa_sequence"].tolist()
        self.site_labels = dataset_df["site_labels"].tolist()
        self.rxn_smiles = dataset_df["canonicalize_rxn_smiles"].tolist()

        afdb_pkl_file = self.alphafolddb_processed_file
        self.rxn_processed_file = "rxn_processed.pkl.gz"
        rxn_pkl_file = os.path.join(self.path, self.rxn_processed_file)
        self.proprecessed_rxn_path = os.path.join(self.path, "rxn_proprecessed")
        os.makedirs(self.proprecessed_rxn_path, exist_ok=True)

        if self.debug:
            afdb_pkl_file = afdb_pkl_file.replace(".pkl.gz", "_debug.pkl.gz")
            rxn_pkl_file = rxn_pkl_file.replace(".pkl.gz", "_debug.pkl.gz")
        self.test_remove_aegan_train = getattr(self, "test_remove_aegan_train", False)
        if self.test_remove_aegan_train:
            afdb_pkl_file = afdb_pkl_file.replace(
                ".pkl.gz", "_remove_aegan_train.pkl.gz"
            )
            rxn_pkl_file = rxn_pkl_file.replace(".pkl.gz", "_remove_aegan_train.pkl.gz")

        if os.path.exists(afdb_pkl_file):
            self.load_pickle(afdb_pkl_file, verbose=verbose, **kwargs)

        else:
            self.use_aug = getattr(self, "use_aug", False)

            if not self.use_aug:

                pdb_files = [
                    os.path.join(self.alphafold_db_path, f"AF-{id}-F1-model_v4.pdb")
                    for id in dataset_df["alphafolddb-id"].tolist()
                ]
            else:
                pdb_files = [
                    os.path.join(self.structure_path, f"{id}_minimized.pdb")
                    for id in dataset_df["alphafolddb-id"].tolist()
                ]

            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(afdb_pkl_file, verbose=verbose)
        assert len(self.pdb_files) == len(self.dataset_df)
        if os.path.exists(rxn_pkl_file):
            self.load_rxn_pickle(rxn_pkl_file, verbose=verbose, **kwargs)
        else:
            rxn_smiles = self.rxn_smiles
            self.load_rxn_smiles(rxn_smiles, verbose=verbose, **kwargs)
            self.save_rxn_pickle(rxn_pkl_file, verbose=verbose)

        assert len(self.rxn_data) == len(self.dataset_df)

        splits = dataset_df["dataset_flag"].tolist()
        self.num_samples = [
            splits.count("train"),
            splits.count("valid"),
            splits.count("test"),
        ]
        self._setup_protein_transforms()
        # pass

    def get_item(self, index):
        _, pdb_name = os.path.split(self.pdb_files[index])
        pdb_id = pdb_name.split(".")[0]
        pdb_file = os.path.join(self.alphafold_db_path, f"{pdb_id}.pdb")
        proprecessed_pdb_file = os.path.join(
            self.proprecessed_alphafold_db_path, f"{pdb_id}.pkl"
        )

        rxn = self.rxn_smiles[index]
        proprecessed_rxn_file = os.path.join(
            self.proprecessed_rxn_path, f"{calculate_rxn_hash(rxn)}.pkl"
        )

        if getattr(self, "lazy", False):

            if not os.path.exists(proprecessed_pdb_file):
                protein = data.Protein.from_pdb(pdb_file, self.kwargs)
                if self.save_processed:
                    torch.save(protein, proprecessed_pdb_file)
            else:
                protein = torch.load(proprecessed_pdb_file)

            if not os.path.exists(proprecessed_rxn_file):
                reaction_features = self._calculate_rxn_features(rxn)
                if self.save_processed:
                    torch.save(reaction_features, proprecessed_rxn_file)
            else:
                reaction_features = torch.load(proprecessed_rxn_file)

        else:
            protein = self.data[index].clone()
            reaction_features = deepcopy(self.rxn_data[index])

        rxn_fclass = ReactionFeatures(reaction_features)

        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()

        item = {"protein_graph": protein, "reaction_graph": rxn_fclass}
        if self.transform:
            item = self.transform(item)
        try:
            active_site = self.calculate_active_sites(
                site_label=self.site_labels[index],
                sequence_length=len(self.uniport_sequences[index]),
            )
        except:
            return {}
        assert protein.num_residue.item() == len(active_site)
        item["targets"] = active_site

        return item

    def load_rxn_pickle(
        self, pkl_file, transform=None, lazy=False, verbose=0, **kwargs
    ):
        with utils.smart_open(pkl_file, "rb") as fin:
            num_sample = pickle.load(fin)

            self.lazy = lazy
            self.kwargs = kwargs
            self.rxn_smiles = []
            self.rxn_data = []

            indexes = range(num_sample)
            if verbose:
                indexes = tqdm(indexes, "Loading %s" % pkl_file)
            for i in indexes:
                rxn, reaction_features = pickle.load(fin)
                self.rxn_smiles.append(rxn)
                self.rxn_data.append(reaction_features)

    def load_rxn_smiles(self, rxn_smiles_list, lazy=False, verbose=0, **kwargs):
        num_sample = len(rxn_smiles_list)
        if num_sample > 1000000:
            warnings.warn(
                "Preprocessing reaction smiles of a large dataset consumes a lot of CPU memory and time. "
                "Use load_rxn_smiles(lazy=True) to construct reactions in the dataloader instead."
            )

        self.lazy = lazy
        self.kwargs = kwargs
        self.rxn_data = []

        if verbose:
            rxn_smiles_list = tqdm(
                rxn_smiles_list, "Constructing reaction features from reaction smiles"
            )

        for i, rxn in enumerate(rxn_smiles_list):
            if not lazy or i == 0:
                reaction_features = self._calculate_rxn_features(rxn)
                if not reaction_features:
                    logger.debug(
                        "Can't construct reaction features from rxn_smiles `%s`. Ignore this sample."
                        % rxn
                    )
                    continue

            else:
                reaction_features = None

            self.rxn_data.append(reaction_features)

    def save_rxn_pickle(self, pkl_file, verbose=0):
        with utils.smart_open(pkl_file, "wb") as fout:
            num_sample = len(self.rxn_data)
            pickle.dump((num_sample), fout)

            indexes = range(num_sample)
            if verbose:
                indexes = tqdm(indexes, "Dumping to %s" % pkl_file)
            for i in indexes:
                pickle.dump((self.rxn_smiles[i], self.rxn_data[i]), fout)

    def _calculate_rxn_features(self, rxn):
        try:
            react, prod = rxn.split(">>")

            react_features_tuple = self._calculate_features(react)
            prod_features_tuple = self._calculate_features(prod)

            return react_features_tuple, prod_features_tuple
        except:
            return None

    def _calculate_features(self, smi):
        mol = Chem.MolFromSmiles(smi)
        fgraph = self.mol_to_graph(
            mol,
            node_featurizer=self.node_featurizer,
            edge_featurizer=self.edge_featurizer,
            canonical_atom_order=False,
        )
        dgraph = get_adm(mol)

        return fgraph, dgraph

    def check_reaction(self, df):

        self.test_remove_aegan_train = getattr(self, "test_remove_aegan_train", False)

        self.reaction_check_file = os.path.join(self.path, "reaction_check_file.pkl")
        if self.debug:
            self.reaction_check_file = self.reaction_check_file.replace(
                ".pkl", "_debug.pkl"
            )
        if self.test_remove_aegan_train:
            self.reaction_check_file = self.reaction_check_file.replace(
                ".pkl", "_remove_aegan_train.pkl"
            )
        if not os.path.exists(self.reaction_check_file):
            print("Checking reaction smiles from csv ...")
            check_df = df[["reaction"]]
            check_df = check_df.drop_duplicates(subset=["reaction"]).reset_index(
                drop=True
            )
            check_df["rxn_smiles"] = check_df["reaction"].apply(
                lambda x: get_rxn_smiles(x)
            )
            df = pd.merge(df, check_df, on=["reaction"], how="left")

            check_df = check_df[["rxn_smiles"]]
            check_df = check_df.drop_duplicates(subset=["rxn_smiles"]).reset_index(
                drop=True
            )

            check_df = multiprocess_reaction_check(check_df, nb_workers=self.nb_workers)

            df = pd.merge(df, check_df, on=["rxn_smiles"], how="left")

            df["is_reaction_valid"] = df["canonicalize_rxn_smiles"] != ""

            reaction_check_df = df[
                ["rxn_smiles", "canonicalize_rxn_smiles", "is_reaction_valid"]
            ]
            reaction_check_df.to_pickle(self.reaction_check_file)

        else:
            print("Loading reaction check file ...")
            reaction_check_df = pd.read_pickle(self.reaction_check_file)

            df = pd.concat([df, reaction_check_df], axis=1)

        count = len(df)
        valid_count = int(df["is_reaction_valid"].sum())

        print(
            "\nReaction from csv file valid {}/{}, {:.2f}%\n".format(
                valid_count, count, 100 * valid_count / count
            )
        )

        return df

    def check_data(self, df):
        df = self.check_structure_sequence(df)
        df = self.check_reaction(df)
        df["is_valid_data"] = df["is_reaction_valid"] & df["is_alphafolddb_valid"]
        return df


class EnzymeReactionSiteTypeDataset(EnzymeReactionDataset):
    def __init__(
        self,
        path,
        dataset_root=_dataset_root,
        structure_path=_structure_path,
        debug=False,
        verbose=1,
        nb_workers=12,
        save_precessed=True,
        protein_max_length=600,
        test_remove_aegan_train=False,
        **kwargs,
    ) -> None:
        self.test_remove_aegan_train = test_remove_aegan_train
        super().__init__(
            path,
            dataset_root,
            structure_path,
            debug,
            verbose,
            nb_workers,
            protein_max_length,
            save_precessed,
            **kwargs,
        )
        self.site_types = self.dataset_df["site_types"].tolist()
        self.num_active_site_type = len(SITE_TYPE_TO_CLASS) + 1  # add negative type

    def get_item(self, index):
        _, pdb_name = os.path.split(self.pdb_files[index])
        pdb_id = pdb_name.split(".")[0]
        pdb_file = os.path.join(self.alphafold_db_path, f"{pdb_id}.pdb")
        proprecessed_pdb_file = os.path.join(
            self.proprecessed_alphafold_db_path, f"{pdb_id}.pkl"
        )

        rxn = self.rxn_smiles[index]
        proprecessed_rxn_file = os.path.join(
            self.proprecessed_rxn_path, f"{calculate_rxn_hash(rxn)}.pkl"
        )

        if getattr(self, "lazy", False):

            if not os.path.exists(proprecessed_pdb_file):
                protein = data.Protein.from_pdb(pdb_file, self.kwargs)
                if self.save_processed:
                    torch.save(protein, proprecessed_pdb_file)
            else:
                protein = torch.load(proprecessed_pdb_file)

            if not os.path.exists(proprecessed_rxn_file):
                reaction_features = self._calculate_rxn_features(rxn)
                if self.save_processed:
                    torch.save(reaction_features, proprecessed_rxn_file)
            else:
                reaction_features = torch.load(proprecessed_rxn_file)

        else:
            protein = self.data[index].clone()
            reaction_features = deepcopy(self.rxn_data[index])

        rxn_fclass = ReactionFeatures(reaction_features)

        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()

        item = {"protein_graph": protein, "reaction_graph": rxn_fclass}
        if self.transform:
            item = self.transform(item)
        active_site_type_labels = self.calculate_active_site_types(
            site_label=self.site_labels[index],
            site_types=self.site_types[index],
            sequence_length=len(self.uniport_sequences[index]),
        )

        assert protein.num_residue.item() == len(active_site_type_labels)
        item["targets"] = active_site_type_labels

        return item

    def calculate_active_site_types(self, site_label, site_types, sequence_length):
        site_label = eval(site_label)  # Note: Site label starts from 1
        site_types = eval(site_types)
        assert len(site_label) == len(site_types)
        active_site_type_labels = torch.zeros((sequence_length,))
        for one_site, tps in zip(
            site_label, site_types
        ):  # Site Type 'BINDING':0, 'ACT_SITE':1 'SITE':2
            if len(one_site) == 1:
                active_site_type_labels[one_site[0] - 1] = (
                    tps + 1
                )  # Convert :# Not Site: 0, Site Type 'BINDING':1, 'ACT_SITE':2 'SITE':3
            elif len(one_site) == 2:
                b, e = one_site
                site_indices = [k - 1 for k in range(b, e + 1)]
                active_site_type_labels[site_indices] = tps + 1
            else:
                raise ValueError("The label of active site is not standard !!!")
        return active_site_type_labels


class EnzymeReactionSiteTypeDatasetForInference(EnzymeReactionSiteTypeDataset):
    def __init__(
        self,
        path,
        dataset_root=_dataset_root,
        structure_path=_structure_path,
        debug=False,
        verbose=1,
        nb_workers=12,
        save_precessed=True,
        test_remove_aegan_train=False,
        **kwargs,
    ) -> None:
        super().__init__(
            path,
            dataset_root,
            structure_path,
            debug,
            verbose,
            nb_workers,
            save_precessed,
            test_remove_aegan_train,
            **kwargs,
        )

    def get_item(self, index):
        item = super().get_item(index)
        _, pdb_name = os.path.split(self.pdb_files[index])
        item["pdb_name"] = pdb_name
        return item


def collate_rxn_features(batch):
    react_fgraphs = [x.react_fgraph for x in batch]
    react_dgraphs = [x.react_dgraph for x in batch]
    prod_fgraphs = [x.prod_fgraph for x in batch]
    prod_dgraphs = [x.prod_dgraph for x in batch]

    rts_bg = dgl.batch(react_fgraphs)
    pds_bg = dgl.batch(prod_fgraphs)
    rts_bg.set_n_initializer(dgl.init.zero_initializer)
    pds_bg.set_n_initializer(dgl.init.zero_initializer)

    rts_adms = pad_atom_distance_matrix(react_dgraphs)
    pds_adms = pad_atom_distance_matrix(prod_dgraphs)

    rts_node_feats, rts_edge_feats = rts_bg.ndata.pop("h"), rts_bg.edata.pop("e")
    pds_node_feats, pds_edge_feats = pds_bg.ndata.pop("h"), pds_bg.edata.pop("e")

    return {
        "rts_bg": rts_bg,
        "rts_adms": rts_adms,
        "rts_node_feats": rts_node_feats,
        "rts_edge_feats": rts_edge_feats,
        "pds_bg": pds_bg,
        "pds_adms": pds_adms,
        "pds_node_feats": pds_node_feats,
        "pds_edge_feats": pds_edge_feats,
    }


def enzyme_rxn_collate(batch):
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
        shapes = [x.shape for x in batch]
        return torch.cat(batch, 0, out=out), torch.tensor(shapes)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, data.Graph):
        return elem.pack(batch)
    elif isinstance(elem, Mapping):
        return {key: enzyme_rxn_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("Each element in list of batch should be of equal size")
        return [enzyme_rxn_collate(samples) for samples in zip(*batch)]

    elif isinstance(elem, ReactionFeatures):
        return collate_rxn_features(batch)

    raise TypeError("Can't collate data with type `%s`" % type(elem))


def enzyme_rxn_collate_extract(batch):
    batch = [x for x in batch if x]
    if not batch:
        return
    batch_data = enzyme_rxn_collate(batch)
    assert isinstance(batch_data, dict)
    if "targets" in batch_data:
        if isinstance(batch_data["targets"], tuple):
            targets, size = batch_data["targets"]
            batch_data["targets"] = targets
            batch_data["protein_len"] = size.view(-1)
    return batch_data


class EnzymeRxnSaprotCollate:
    def __init__(self) -> None:
        foldseek_seq_vocab = "ACDEFGHIKLMNPQRSTVWY#"
        foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"
        # Initialize the alphabet
        tokens = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]
        for seq_token, struc_token in itertools.product(
            foldseek_seq_vocab, foldseek_struc_vocab
        ):
            token = seq_token + struc_token
            tokens.append(token)
        alphabet = esm.data.Alphabet(
            standard_toks=tokens,
            prepend_toks=[],
            append_toks=[],
            prepend_bos=True,
            append_eos=True,
            use_msa=False,
        )
        alphabet.all_toks = alphabet.all_toks[:-2]
        alphabet.unique_no_split_tokens = alphabet.all_toks
        alphabet.tok_to_idx = {tok: i for i, tok in enumerate(alphabet.all_toks)}

        self.alphabet = alphabet
        self.batch_converter = self.alphabet.get_batch_converter()

    def __call__(self, batch):
        batch = [x for x in batch if x]
        if not batch:
            return
        saprot_combined_seq = [x.pop("saprot_combined_seq") for x in batch]
        saprot_data = [(f"batch_{i}", x) for i, x in enumerate(saprot_combined_seq)]
        _, _, batch_tokens = self.batch_converter(saprot_data)

        batch_data = enzyme_rxn_collate(batch)
        batch_data["saprot_batch_tokens"] = batch_tokens
        assert isinstance(batch_data, dict)
        if "targets" in batch_data:
            if isinstance(batch_data["targets"], tuple):
                targets, size = batch_data["targets"]
                batch_data["targets"] = targets
                batch_data["protein_len"] = size.view(-1)
        return batch_data


def check_function(batch):

    protein_len = torch.stack(
        [
            batch["protein_graph"][i].num_residue
            for i in range(len(batch["protein_graph"]))
        ]
    )

    assert protein_len.sum().item() == batch["targets"].size(0)
    assert (protein_len == batch["protein_len"]).all().item()


class EnzymeReactionTestDataset(EnzymeReactionDataset):
    def __init__(
        self,
        path,
        dataset_root=_dataset_root,
        structure_path=_external_test_structure_path,
        structure_source="pdb",
        debug=False,
        verbose=1,
        nb_workers=12,
        save_precessed=True,
        **kwargs,
    ) -> None:

        assert structure_source in ["pdb", "alphafolddb"]

        self.path = os.path.join(os.path.dirname(dataset_root), path)
        self.structure_path = structure_path
        self.verbose = verbose
        self.save_processed = save_precessed
        self.nb_workers = nb_workers
        self.data_source = structure_source
        self.debug = debug

        self.mol_to_graph = partial(mol_to_bigraph, add_self_loop=True)
        self.node_featurizer = WeaveAtomFeaturizer(atom_types=atom_types)
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self._init_enzyme_data_path()
        dataset_df = pd.read_csv(
            os.path.join(self.path, "rxnaamapper_external_test_set_handled.csv")
        )
        dataset_df["rxn_smiles"] = dataset_df["rxn"].apply(lambda x: get_rxn_smiles(x))
        dataset_df["canonicalize_rxn_smiles"] = dataset_df["rxn_smiles"].apply(
            lambda x: process_reaction(x)
        )

        self.dataset_df = dataset_df

        self.uniport_sequences = dataset_df["vaild_aa_sequence"].tolist()
        self.site_labels = dataset_df["vaild_active_site"].tolist()

        self.rxn_smiles = dataset_df["canonicalize_rxn_smiles"].tolist()

        pdb_pkl_file = self.structure_propecessed_file
        self.rxn_processed_file = "rxn_processed.pkl.gz"
        rxn_pkl_file = os.path.join(self.path, self.rxn_processed_file)
        self.proprecessed_rxn_path = os.path.join(self.path, "rxn_proprecessed")

        if self.debug:
            pdb_pkl_file = pdb_pkl_file.replace(".pkl.gz", "_debug.pkl.gz")
            rxn_pkl_file = rxn_pkl_file.replace(".pkl.gz", "_debug.pkl.gz")

        if os.path.exists(pdb_pkl_file):
            self.load_pickle(pdb_pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = [
                os.path.join(self.structure_path, f"{pdb_id}", f"{pdb_file}")
                for pdb_id, pdb_file in zip(
                    dataset_df["pdb-id"].tolist(),
                    dataset_df[f"{self.data_source}_file"].tolist(),
                )
            ]
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pdb_pkl_file, verbose=verbose)

        assert len(self.pdb_files) == len(self.dataset_df)

        if os.path.exists(rxn_pkl_file):
            self.load_rxn_pickle(rxn_pkl_file, verbose=verbose, **kwargs)
        else:
            rxn_smiles = self.rxn_smiles
            self.load_rxn_smiles(rxn_smiles, verbose=verbose, **kwargs)
            self.save_rxn_pickle(rxn_pkl_file, verbose=verbose)

        self._setup_protein_transforms()

    def calculate_active_sites(self, site_label, sequence_length):
        site_label = eval(site_label)  # Note: Site label starts from 0
        active_site = torch.zeros((sequence_length,))
        for one_site in site_label:
            if len(one_site) == 1:
                active_site[one_site[0]] = 1
            elif len(one_site) == 2:
                b, e = one_site
                site_indices = [k for k in range(b, e + 1)]
                active_site[site_indices] = 1
            else:
                raise ValueError("The label of active site is not standard !!!")
        return active_site

    def get_item(self, index):
        pdb_id, pdb_name = os.path.split(self.pdb_files[index])
        pdb_file = os.path.join(self.structure_path, self.pdb_files[index])
        proprecessed_pdb_file = os.path.join(
            self.structure_propecessed_path, f"{pdb_id}.pkl"
        )

        rxn = self.rxn_smiles[index]
        proprecessed_rxn_file = os.path.join(
            self.proprecessed_rxn_path, f"{calculate_rxn_hash(rxn)}.pkl"
        )

        if getattr(self, "lazy", False):

            if not os.path.exists(proprecessed_pdb_file):
                protein = data.Protein.from_pdb(pdb_file, self.kwargs)
                if self.save_processed:
                    torch.save(protein, proprecessed_pdb_file)
            else:
                protein = torch.load(proprecessed_pdb_file)

            if not os.path.exists(proprecessed_rxn_file):
                reaction_features = self._calculate_rxn_features(rxn)
                if self.save_processed:
                    torch.save(reaction_features, proprecessed_rxn_file)
            else:
                reaction_features = torch.load(proprecessed_rxn_file)

        else:
            protein = self.data[index].clone()
            reaction_features = deepcopy(self.rxn_data[index])

        rxn_fclass = ReactionFeatures(reaction_features)

        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()

        item = {"protein_graph": protein, "reaction_graph": rxn_fclass}
        if self.transform:
            item = self.transform(item)

        calculated_protein_sequence = protein.to_sequence().replace(".", "")
        active_site = torch.from_numpy(
            map_active_site_for_one(
                self.uniport_sequences[index],
                calculated_protein_sequence,
                eval(self.site_labels[index]),
            )
        )

        # active_site = self.calculate_active_sites(
        #     site_label=self.site_labels[index],
        #     sequence_length=len(self.uniport_sequences[index]))
        assert protein.num_residue.item() == len(active_site)
        item["targets"] = active_site

        return item

    def _init_enzyme_data_path(self):
        structure_path = self.structure_path

        self.structure_propecessed_path = os.path.join(
            os.path.dirname(structure_path),
            f"{self.data_source}_structure_proprecessed",
        )
        os.makedirs(self.structure_propecessed_path, exist_ok=True)

        self.structure_propecessed_file = os.path.join(
            os.path.dirname(structure_path),
            f"{self.data_source}_structure_proprecessed.pkl.gz",
        )


class AugEnzymeReactionDataset(EnzymeReactionDataset):
    def __init__(
        self,
        path,
        dataset_root=_dataset_root,
        structure_path=_structure_path,
        debug=False,
        verbose=1,
        nb_workers=12,
        protein_max_length=600,
        save_precessed=True,
        use_aug=False,
        soft_check=False,
        **kwargs,
    ) -> None:

        self.use_aug = use_aug
        self.soft_check = soft_check
        super().__init__(
            path,
            dataset_root,
            structure_path,
            debug,
            verbose,
            nb_workers,
            protein_max_length,
            save_precessed,
            **kwargs,
        )

    def get_merged_df(self, dataset_flag):

        df = pd.DataFrame()
        self.test_remove_aegan_train = getattr(self, "test_remove_aegan_train", False)

        if (dataset_flag == "test") and self.test_remove_aegan_train:
            data_path = os.path.join(self.path, "test_dataset_rm_aegan_train_same")
        else:
            self.use_aug = getattr(self, "use_aug", False)
            if self.use_aug:
                data_path = os.path.join(self.path, f"new_{dataset_flag}_dataset")
            else:
                data_path = os.path.join(self.path, f"{dataset_flag}_dataset")
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
        df["alphafolddb-id"] = df["alphafolddb-id"].apply(lambda x: x.split(";")[0])

        df["pdb_files"] = df["alphafolddb-id"].apply(
            lambda x: os.path.join(self.structure_path, f"{x}_minimized.pdb")
        )
        df.loc[:, "dataset_flag"] = dataset_flag
        return df

    def check_data(self, df):
        df = self.check_structure_sequence(df, soft_check=self.soft_check)
        df = self.check_reaction(df)
        df["is_valid_data"] = df["is_reaction_valid"] & df["is_alphafolddb_valid"]
        return df

    def get_item(self, index):
        _, pdb_name = os.path.split(self.pdb_files[index])
        pdb_id = pdb_name.split(".")[0]
        pdb_file = os.path.join(self.structure_path, f"{pdb_id}.pdb")
        proprecessed_pdb_file = os.path.join(
            self.proprecessed_alphafold_db_path, f"{pdb_id}.pkl"
        )

        rxn = self.rxn_smiles[index]
        proprecessed_rxn_file = os.path.join(
            self.proprecessed_rxn_path, f"{calculate_rxn_hash(rxn)}.pkl"
        )

        if getattr(self, "lazy", False):

            if not os.path.exists(proprecessed_pdb_file):
                protein = MyProtein.from_pdb(pdb_file, self.kwargs)
                if self.save_processed:
                    torch.save(protein, proprecessed_pdb_file)
            else:
                protein = torch.load(proprecessed_pdb_file)

            if not os.path.exists(proprecessed_rxn_file):
                reaction_features = self._calculate_rxn_features(rxn)
                if self.save_processed:
                    torch.save(reaction_features, proprecessed_rxn_file)
            else:
                reaction_features = torch.load(proprecessed_rxn_file)

        else:
            protein = self.data[index].clone()
            reaction_features = deepcopy(self.rxn_data[index])

        rxn_fclass = ReactionFeatures(reaction_features)

        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()

        item = {"protein_graph": protein, "reaction_graph": rxn_fclass}
        if self.transform:
            item = self.transform(item)
        try:
            active_site = self.calculate_active_sites(
                site_label=self.site_labels[index],
                sequence_length=len(self.uniport_sequences[index]),
            )
        except:
            return {}
        assert protein.num_residue.item() == len(active_site)
        item["targets"] = active_site

        return item


class EnzymeReactionSaProtDataset(EnzymeReactionDataset):
    def __init__(
        self,
        path,
        dataset_root=_dataset_root,
        structure_path=_structure_path,
        debug=False,
        verbose=1,
        nb_workers=12,
        protein_max_length=600,
        save_precessed=True,
        foldseek_bin_path=None,
        **kwargs,
    ) -> None:
        super().__init__(
            path,
            dataset_root,
            structure_path,
            debug,
            verbose,
            nb_workers,
            protein_max_length,
            save_precessed,
            **kwargs,
        )

        self.foldseek_bin_path = foldseek_bin_path

    def get_item(self, index):
        _, pdb_name = os.path.split(self.pdb_files[index])
        pdb_id = pdb_name.split(".")[0]
        pdb_file = os.path.join(self.alphafold_db_path, f"{pdb_id}.pdb")
        proprecessed_pdb_file = os.path.join(
            self.proprecessed_alphafold_db_path, f"{pdb_id}.pkl"
        )

        rxn = self.rxn_smiles[index]
        proprecessed_rxn_file = os.path.join(
            self.proprecessed_rxn_path, f"{calculate_rxn_hash(rxn)}.pkl"
        )

        if getattr(self, "lazy", False):

            if not os.path.exists(proprecessed_pdb_file):
                protein = data.Protein.from_pdb(pdb_file, self.kwargs)
                if self.save_processed:
                    torch.save(protein, proprecessed_pdb_file)
            else:
                protein = torch.load(proprecessed_pdb_file)

            if not os.path.exists(proprecessed_rxn_file):
                reaction_features = self._calculate_rxn_features(rxn)
                if self.save_processed:
                    torch.save(reaction_features, proprecessed_rxn_file)
            else:
                reaction_features = torch.load(proprecessed_rxn_file)

        else:
            protein = self.data[index].clone()
            reaction_features = deepcopy(self.rxn_data[index])

        rxn_fclass = ReactionFeatures(reaction_features)

        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()

        item = {"protein_graph": protein, "reaction_graph": rxn_fclass}
        if self.transform:
            item = self.transform(item)
        try:
            active_site = self.calculate_active_sites(
                site_label=self.site_labels[index],
                sequence_length=len(self.uniport_sequences[index]),
            )
        except:
            return {}
        assert protein.num_residue.item() == len(active_site), f'{pdb_file}'
        item["targets"] = active_site

        saprot_combined_seq = self.calculate_saprot_parsed_seqs(
            pdb_file, process_id=pdb_id
        )
        item["saprot_combined_seq"] = saprot_combined_seq
        return item

    def calculate_saprot_parsed_seqs(self, pdb_path, process_id):
        # Sample a random rank to avoid file conflict
        rank = random.randint(0, 1000000)
        try:
            parsed_seqs = get_struc_seq(
                self.foldseek_bin_path,
                pdb_path,
                chains=["A"],
                process_id=f"{process_id}_{rank}",
            )[
                "A"
            ]  # 现在只考虑一个chain的情况
        except:
            print(process_id)
            ValueError()
        combined_seq = parsed_seqs[-1]
        return combined_seq


class EnzymeReactionSiteTypeSaProtDataset(EnzymeReactionSiteTypeDataset):
    def __init__(
        self,
        path,
        dataset_root=_dataset_root,
        structure_path=_structure_path,
        debug=False,
        verbose=1,
        nb_workers=12,
        save_precessed=True,
        protein_max_length=600,
        test_remove_aegan_train=False,
        foldseek_bin_path=None,
        **kwargs,
    ) -> None:
        super().__init__(
            path,
            dataset_root,
            structure_path,
            debug,
            verbose,
            nb_workers,
            save_precessed,
            protein_max_length,
            test_remove_aegan_train,
            **kwargs,
        )

        self.foldseek_bin_path = foldseek_bin_path

    def get_item(self, index):
        _, pdb_name = os.path.split(self.pdb_files[index])
        pdb_id = pdb_name.split(".")[0]
        pdb_file = os.path.join(self.alphafold_db_path, f"{pdb_id}.pdb")
        proprecessed_pdb_file = os.path.join(
            self.proprecessed_alphafold_db_path, f"{pdb_id}.pkl"
        )

        rxn = self.rxn_smiles[index]
        proprecessed_rxn_file = os.path.join(
            self.proprecessed_rxn_path, f"{calculate_rxn_hash(rxn)}.pkl"
        )

        if getattr(self, "lazy", False):

            if not os.path.exists(proprecessed_pdb_file):
                protein = data.Protein.from_pdb(pdb_file, self.kwargs)
                if self.save_processed:
                    torch.save(protein, proprecessed_pdb_file)
            else:
                protein = torch.load(proprecessed_pdb_file)

            if not os.path.exists(proprecessed_rxn_file):
                reaction_features = self._calculate_rxn_features(rxn)
                if self.save_processed:
                    torch.save(reaction_features, proprecessed_rxn_file)
            else:
                reaction_features = torch.load(proprecessed_rxn_file)

        else:
            protein = self.data[index].clone()
            reaction_features = deepcopy(self.rxn_data[index])

        rxn_fclass = ReactionFeatures(reaction_features)

        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()

        item = {"protein_graph": protein, "reaction_graph": rxn_fclass}
        if self.transform:
            item = self.transform(item)
        active_site_type_labels = self.calculate_active_site_types(
            site_label=self.site_labels[index],
            site_types=self.site_types[index],
            sequence_length=len(self.uniport_sequences[index]),
        )

        assert protein.num_residue.item() == len(active_site_type_labels)
        item["targets"] = active_site_type_labels

        saprot_combined_seq = self.calculate_saprot_parsed_seqs(
            pdb_file, process_id=pdb_id
        )
        item["saprot_combined_seq"] = saprot_combined_seq

        return item

    def calculate_saprot_parsed_seqs(self, pdb_path, process_id):
        # Sample a random rank to avoid file conflict
        rank = random.randint(0, 1000000)
        try:
            parsed_seqs = get_struc_seq(
                self.foldseek_bin_path,
                pdb_path,
                chains=["A"],
                process_id=f"{process_id}_{rank}",
            )[
                "A"
            ]  # 现在只考虑一个chain的情况
        except:
            print(process_id)
            ValueError()
        combined_seq = parsed_seqs[-1]
        return combined_seq


class EnzymeReactionRXNFPDataset(EnzymeReactionDataset):
    def get_item(self, index):
        _, pdb_name = os.path.split(self.pdb_files[index])
        pdb_id = pdb_name.split(".")[0]
        pdb_file = os.path.join(self.alphafold_db_path, f"{pdb_id}.pdb")
        proprecessed_pdb_file = os.path.join(
            self.proprecessed_alphafold_db_path, f"{pdb_id}.pkl"
        )

        rxn = self.rxn_smiles[index]

        if getattr(self, "lazy", False):

            if not os.path.exists(proprecessed_pdb_file):
                protein = data.Protein.from_pdb(pdb_file, self.kwargs)
                if self.save_processed:
                    torch.save(protein, proprecessed_pdb_file)
            else:
                protein = torch.load(proprecessed_pdb_file)

        else:
            protein = self.data[index].clone()


        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()

        item = {"protein_graph": protein, "rxn_smiles": rxn}
        if self.transform:
            item = self.transform(item)
        try:
            active_site = self.calculate_active_sites(
                site_label=self.site_labels[index],
                sequence_length=len(self.uniport_sequences[index]),
            )
        except:
            return {}
        assert protein.num_residue.item() == len(active_site)
        item["targets"] = active_site

        return item
    
class EnzymeRxnfpCollate:
    def __init__(self, rxnfp_model_name='bert_ft', rxnfp_max_length=512) -> None:

        self.rxnfp_tokenizer_vocab_path = (
        pkg_resources.resource_filename(
                    "rxnfp",
                    f"models/transformers/{rxnfp_model_name}/vocab.txt"
                )
        )
        self.rxnfp_max_length = rxnfp_max_length
        self.tokenizer = SmilesTokenizer(
            self.rxnfp_tokenizer_vocab_path
        )
        pass

    def __call__(self, batch):
        batch = [x for x in batch if x]
        if not batch:
            return
        rxn_smiles = [x.pop("rxn_smiles") for x in batch]

        rxn_bert_inputs = self.tokenizer.batch_encode_plus(rxn_smiles, max_length=self.rxnfp_max_length, padding=True, truncation=True, return_tensors='pt')

        batch_data = enzyme_rxn_collate(batch)
        batch_data["rxn_bert_inputs"] = rxn_bert_inputs
        assert isinstance(batch_data, dict)
        if "targets" in batch_data:
            if isinstance(batch_data["targets"], tuple):
                targets, size = batch_data["targets"]
                batch_data["targets"] = targets
                batch_data["protein_len"] = size.view(-1)
        return batch_data

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset = EnzymeReactionDataset(path='dataset/ec_site_dataset/uniprot_ecreact_merge_dataset_limit_10000', save_precessed=False, debug=False, verbose=1, lazy=True, nb_workers=12)

    # train_set, valid_set, test_set = dataset.split()
    # for i in tqdm(range(10)):
    #     train_set[i]
    #     valid_set[i]
    #     test_set[i]

    # train_loader = torch_data.DataLoader(
    #     train_set,
    #     batch_size=8,
    #     collate_fn=enzyme_rxn_collate_extract,
    #     num_workers=4)
    # for batch_data in tqdm(train_loader, desc='train loader'):
    #     if device.type == "cuda":
    #         batch_data = cuda(batch_data, device=device)
    #     check_function(batch_data)

    # external_test_set = EnzymeReactionTestDataset(path='dataset/rxnaamapper_external_test', save_precessed=False, debug=True, structure_source='pdb', verbose=1, lazy=True, nb_workers=12)

    # external_test_set[0]
    # for i in range(len(external_test_set)):
    #     try:
    #         external_test_set[i]
    #     except:
    #         print(i)

    # erro_idx = [15, 16]
    # for i in erro_idx:
    #     try:
    #         external_test_set[i]
    #     except:
    #         print(i)
    # pass

    # dataset = EnzymeReactionSiteTypeDataset(path='dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100', save_precessed=False, debug=False, verbose=1, lazy=True, nb_workers=12)

    # train_set, valid_set, test_set = dataset.split()
    # for i in tqdm(range(10)):
    #     train_set[i]
    #     valid_set[i]
    #     test_set[i]

    # train_loader = torch_data.DataLoader(
    #     train_set,
    #     batch_size=8,
    #     collate_fn=enzyme_rxn_collate_extract,
    #     num_workers=0)
    # for batch_data in tqdm(train_loader, desc='train loader'):
    #     if device.type == "cuda":
    #         batch_data = cuda(batch_data, device=device)
    #     check_function(batch_data)

    # dataset = EnzymeReactionSiteTypeSaProtDataset(
    #     path="dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100",
    #     save_precessed=False,
    #     debug=False,
    #     verbose=1,
    #     lazy=True,
    #     nb_workers=12,
    #     foldseek_bin_path="../foldseek_bin/foldseek",
    # )
    '''
    SaProt数据集类测试区域
    '''
    # dataset = EnzymeReactionSaProtDataset(
    #     path="dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100",
    #     save_precessed=False,
    #     debug=False,
    #     verbose=1,
    #     lazy=True,
    #     nb_workers=12,
    #     foldseek_bin_path="../foldseek_bin/foldseek",
    # )

    # train_set, valid_set, test_set = dataset.split()
    # for i in tqdm(range(10)):
    #     train_set[i]
    #     valid_set[i]
    #     test_set[i]
    # enzyme_rxn_saprot_collate_extract = EnzymeRxnSaprotCollate()
    # train_loader = torch_data.DataLoader(
    #     train_set,
    #     batch_size=8,
    #     collate_fn=enzyme_rxn_saprot_collate_extract,
    #     # collate_fn=enzyme_rxn_collate_extract,
    #     num_workers=2,
    # )
    # for batch_data in tqdm(train_loader, desc="train loader"):
    #     if device.type == "cuda":
    #         batch_data = cuda(batch_data, device=device)
    #     check_function(batch_data)

    '''
    rxnfp数据集类测试区域
    '''


    dataset = EnzymeReactionRXNFPDataset(
        path="dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100",
        save_precessed=False,
        debug=False,
        verbose=1,
        lazy=True,
        nb_workers=12
    )

    train_set, valid_set, test_set = dataset.split()
    for i in tqdm(range(10)):
        train_set[i]
        valid_set[i]
        test_set[i]
    enzyme_rxnfp_collate_extract = EnzymeRxnfpCollate()
    train_loader = torch_data.DataLoader(
        train_set,
        batch_size=8,
        collate_fn=enzyme_rxnfp_collate_extract,
        # collate_fn=enzyme_rxn_collate_extract,
        num_workers=2,
    )
    for batch_data in tqdm(train_loader, desc="train loader"):
        if device.type == "cuda":
            batch_data = cuda(batch_data, device=device)
        check_function(batch_data)