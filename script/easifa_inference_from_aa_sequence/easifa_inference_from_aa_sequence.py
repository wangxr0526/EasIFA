# %%
import sys
import torch
import os
import time

sys.path.append("../../")
from tqdm.auto import tqdm
from collections import defaultdict
from functools import partial
import py3Dmol
from IPython.display import IFrame, SVG, display, HTML
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
from rdkit import Chem
from pandarallel import pandarallel
from webapp.utils import (
    EasIFAInferenceAPI,
    ECSiteBinInferenceAPI,
    ECSiteSeqBinInferenceAPI,
    UniProtParserMysql,
    get_structure_html_and_active_data,
    cmd,
)
from data_loaders.rxn_dataloader import process_reaction
from data_loaders.enzyme_rxn_dataloader import get_rxn_smiles
from common.utils import calculate_scores_vbin_test


# %%
device = 'cuda:0'
use_esmfold = False
ECSitePred = ECSiteBinInferenceAPI(
    model_checkpoint_path="../../checkpoints/enzyme_site_predition_model/train_in_uniprot_ecreact_cluster_split_merge_dataset_limit_100_at_2024-05-24-02-53-35/global_step_92000/",
    device=device,
    pred_tolist=False,
)
ECSiteSeqPred = ECSiteSeqBinInferenceAPI(
    model_checkpoint_path="../../checkpoints/enzyme_site_no_gearnet_prediction_model/train_in_uniprot_ecreact_cluster_split_merge_dataset_limit_100_at_2024-05-20-05-13-33/global_step_24000",
    device=device,
    pred_tolist=False,
)
# unprot_mysql_parser = UniProtParserMysql(
#     mysql_config_path="../../webapp/mysql_config.json"
# )


# %%
def calculate_active_sites(site_label, sequence_length):
    site_label = eval(site_label)  # Note: Site label starts from 1
    active_site = torch.zeros((sequence_length,))
    for one_site in site_label:
        if len(one_site) == 1:
            active_site[one_site[0] - 1] = 1
        elif len(one_site) == 2:
            b, e = one_site
            site_indices = [k - 1 for k in range(b, e + 1)]
            # site_indices = [k - 1 for k in range(b, e)]
            active_site[site_indices] = 1
        else:
            raise ValueError("The label of active site is not standard !!!")
    return active_site


def inference_and_scoring(test_dataset: pd.DataFrame, esmfold_pdb_path):

    accuracy_list = []
    precision_list = []
    specificity_list = []
    overlap_scores_list = []
    false_positive_rates_list = []
    f1_scores_list = []
    mcc_scores_list = []

    pbar = tqdm(test_dataset.iterrows(), total=len(test_dataset), desc="Testing")

    for idx, row in pbar:
        uniprot_id = row["alphafolddb-id"]
        rxn = row["canonicalize_rxn_smiles"]
        site_label = row["site_labels"]
        aa_sequence = row["aa_sequence"]
        gts = calculate_active_sites(site_label, len(aa_sequence))

        enzyme_structure_path = os.path.join(esmfold_pdb_path, f"{uniprot_id}.pdb")
        try:
            pred_active_labels = ECSitePred.inference(
                rxn, enzyme_structure_path
            )  # 默认输出一个样本的结果
        except:
            print(f'PDB file unavailble for {uniprot_id}, using sequence model instead.')
            pred_active_labels = ECSiteSeqPred.inference(
                rxn, aa_sequence
            )  # 当PDB仍然不可用时，使用基于序列的模型进行预测

        (
            accuracy,
            precision,
            specificity,
            overlap_score,
            fpr,
            f1_scores,
            mcc_scores,
        ) = calculate_scores_vbin_test(
            pred_active_labels, gts=gts, num_residues=[len(aa_sequence)]
        )
        accuracy_list += accuracy
        precision_list += precision
        specificity_list += specificity
        overlap_scores_list += overlap_score
        false_positive_rates_list += fpr
        f1_scores_list += f1_scores
        mcc_scores_list += mcc_scores

        pbar.set_description(
            "Accuracy: {:.4f}, Precision: {:.4f}, Specificity: {:.4f}, Overlap Score: {:.4f}, False Positive Rate: {:.4f}, F1: {:.4f}, MCC: {:.4f}".format(
                sum(accuracy_list) / len(accuracy_list),
                sum(precision_list) / len(precision_list),
                sum(specificity_list) / len(specificity_list),
                sum(overlap_scores_list) / len(overlap_scores_list),
                sum(false_positive_rates_list) / len(false_positive_rates_list),
                sum(f1_scores_list) / len(f1_scores_list),
                sum(mcc_scores_list) / len(mcc_scores_list),
            )
        )
        print(f"Get {len(overlap_scores_list)} results")

    print(
        "Accuracy: {:.4f}, Precision: {:.4f}, Specificity: {:.4f}, Overlap Score: {:.4f}, False Positive Rate: {:.4f}, F1: {:.4f}, MCC: {:.4f}".format(
            sum(accuracy_list) / len(accuracy_list),
            sum(precision_list) / len(precision_list),
            sum(specificity_list) / len(specificity_list),
            sum(overlap_scores_list) / len(overlap_scores_list),
            sum(false_positive_rates_list) / len(false_positive_rates_list),
            sum(f1_scores_list) / len(f1_scores_list),
            sum(mcc_scores_list) / len(mcc_scores_list),
        )
    )


# %%


def get_structure_sequence(pdb_file):
    try:
        mol = Chem.MolFromPDBFile(pdb_file)
        protein_sequence = Chem.MolToSequence(mol)
    except:
        protein_sequence = ""
    return protein_sequence


def multiprocess_structure_check(df, nb_workers, pdb_file_path):

    if nb_workers != 0:

        pandarallel.initialize(nb_workers=nb_workers, progress_bar=True)
        df["pdb_files"] = df["alphafolddb-id"].parallel_apply(
            lambda x: os.path.join(pdb_file_path, f"AF-{x}-F1-model_v4.pdb")
        )
        df["aa_sequence_calculated"] = df["pdb_files"].parallel_apply(
            lambda x: get_structure_sequence(x)
        )
    else:
        tqdm.pandas(desc="pandas bar")
        df["pdb_files"] = df["alphafolddb-id"].progress_apply(
            lambda x: os.path.join(pdb_file_path, f"AF-{x}-F1-model_v4.pdb")
        )
        df["aa_sequence_calculated"] = df["pdb_files"].progress_apply(
            lambda x: get_structure_sequence(x)
        )

    df["is_valid"] = df["aa_sequence_calculated"] == df["aa_sequence"]

    return df


def get_query_database(path, fasta_path, pdb_file_path):
    database_df = pd.read_csv(path)
    database_df = database_df[
        ["alphafolddb-id", "aa_sequence", "site_labels", "site_types", "reaction"]
    ]
    database_df["alphafolddb-id"] = database_df["alphafolddb-id"].apply(
        lambda x: x.replace(";", "")
    )
    database_df["rxn_smiles"] = database_df["reaction"].apply(
        lambda x: get_rxn_smiles(x)
    )
    database_df["canonicalize_rxn_smiles"] = database_df["rxn_smiles"].apply(
        lambda x: process_reaction(x)
    )

    database_df = multiprocess_structure_check(
        database_df, nb_workers=12, pdb_file_path=pdb_file_path
    )

    write_database_df = database_df.drop_duplicates(
        subset=["alphafolddb-id", "aa_sequence", "site_labels", "site_types"]
    ).reset_index(drop=True)

    with open(fasta_path, "w", encoding="utf-8") as f:
        for idx, row in tqdm(
            write_database_df.iterrows(), total=len(write_database_df)
        ):
            f.write(">{}\n".format(row["alphafolddb-id"]))
            f.write("{}\n".format(row["aa_sequence"]))
    return database_df


# %%
dataset_path = "../../dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100"
blast_database_path = "../../dataset/raw_dataset/uniprot/uniprot_sprot.fasta"
test_dataset_fasta_path = os.path.join(dataset_path, "test_dataset.fasta")
esmfold_pdb_path = "./esmfold_pdb"
os.makedirs(esmfold_pdb_path, exist_ok=True)

test_dataset = get_query_database(
    os.path.join(dataset_path, "test_dataset", "uniprot_ecreact_merge.csv"),
    fasta_path=test_dataset_fasta_path,
    pdb_file_path=os.path.join(
        os.path.dirname(dataset_path), "structures", "alphafolddb_download"
    ),
)
test_dataset = test_dataset.loc[test_dataset["is_valid"]].reset_index(drop=True)
test_dataset


# %%
import subprocess

esmfold_script = os.path.abspath("../esmfold_inference.py")
test_dataset_fasta_abspath = os.path.abspath(test_dataset_fasta_path)
esmfold_pdb_abspath = os.path.abspath(esmfold_pdb_path)

pdb_fnames = [x for x in os.listdir(esmfold_pdb_abspath) if x.endswith(".pdb")]


esmfold_cmd = (
    f"python {esmfold_script} -i {test_dataset_fasta_abspath} -o {esmfold_pdb_abspath}"
)

if use_esmfold:
    esmfold_start_time = time.time()
    if len(pdb_fnames) != len(set(test_dataset["alphafolddb-id"])):
        subprocess.run(esmfold_cmd, shell=True)
    esmfold_use_time = time.time() - esmfold_start_time
    print(f"ESMfold inference use: {esmfold_use_time}s")


# %%
inference_and_scoring(test_dataset=test_dataset, esmfold_pdb_path=esmfold_pdb_abspath)
