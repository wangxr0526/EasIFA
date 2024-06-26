# %%

import pandas as pd
import os
import subprocess
from tqdm import tqdm
import shutil
import numpy as np
import time
from utils import calculate_score
import sys

sys.path.append('../../')
from dataset_preprocess.pdb_preprocess_utils import get_active_site_binary

import pymol

# 初始化Pymol
pymol.finish_launching(['pymol', '-c'])

# %%
import pandas as pd
import os
from tqdm.auto import tqdm
from pandarallel import pandarallel
from rdkit import Chem
from tqdm import tqdm as top_tqdm
def get_structure_sequence(pdb_file):
    try:
        mol = Chem.MolFromPDBFile(pdb_file)
        protein_sequence = Chem.MolToSequence(mol)
    except:
        protein_sequence = ''
    return protein_sequence
def multiprocess_structure_check(df, nb_workers, pdb_file_path):
    
    if nb_workers != 0:

        pandarallel.initialize(nb_workers=nb_workers, progress_bar=True)
        df['pdb_files'] = df['alphafolddb-id'].parallel_apply(
            lambda x: os.path.join(pdb_file_path, f'AF-{x}-F1-model_v4.pdb'))
        df['aa_sequence_calculated'] = df['pdb_files'].parallel_apply(
            lambda x: get_structure_sequence(x))
    else:
        top_tqdm.pandas(desc='pandas bar')
        df['pdb_files'] = df['alphafolddb-id'].progress_apply(
            lambda x: os.path.join(pdb_file_path, f'AF-{x}-F1-model_v4.pdb'))
        df['aa_sequence_calculated'] = df['pdb_files'].progress_apply(
            lambda x: get_structure_sequence(x))
    
    df['is_valid'] = (df['aa_sequence_calculated'] == df['aa_sequence'])

    return df
def get_query_database(path, pdb_file_path):
    database_df = pd.read_csv(path)
    database_df = database_df[['alphafolddb-id', 'aa_sequence','site_labels', 'site_types']]
    database_df['alphafolddb-id'] = database_df['alphafolddb-id'].apply(lambda x:x.replace(';',''))
    
    database_df = multiprocess_structure_check(database_df, nb_workers=0, pdb_file_path=pdb_file_path)
    
    write_database_df = database_df.drop_duplicates(subset=['alphafolddb-id', 'aa_sequence','site_labels', 'site_types']).reset_index(drop=True)


    # with open(fasta_path, 'w', encoding='utf-8') as f:
    #     for idx, row in tqdm(write_database_df.iterrows(), total=len(write_database_df)):
    #         f.write('>{}\n'.format(row['alphafolddb-id']))
    #         f.write('{}\n'.format(row['aa_sequence']))
    return database_df

def get_active_site(pdb_name, file_path, distance=2, top_n=1):
    # 只能检测一个样例，多个样例需要每一个样例一个单独文件夹
    pdb_files = [x for x in os.listdir(file_path) if x.endswith('.pdb')]
    # print(pdb_files)
    results_pdb_files = [x for x in pdb_files if x.startswith(pdb_name)]
    # print(results_pdb_files)
    
    pdb_mark = [int(x.split('.')[0].split('-')[-1]) for x in results_pdb_files]
    pdb_mark.sort()
    site_marks = pdb_mark[:-1]
    
    structure_pdb_name = '{}-{}.pdb'.format(pdb_name, pdb_mark[-1])
    site_pdb_names = ['{}-{}.pdb'.format(pdb_name, x) for x in site_marks]
    
    
    pymol.cmd.load(os.path.join(file_path, structure_pdb_name), 'structure')
    select_names = []
    for i, name in enumerate(site_pdb_names):
        pymol.cmd.load(os.path.join(file_path, name), f'site-{i}')
        select_names.append(f'site-{i}')
    
    
    pymol.cmd.select('all_site', ' or '.join(select_names[:top_n]))
    
    pymol.cmd.select('active_res', 'br. all_site around {}'.format(distance))
    
    myspace = {'selected_residues': []}
    pymol.cmd.iterate('active_res', 'selected_residues.append(resi)', space=myspace)
    all_active_site_idx = [int(x) for x in list(set(myspace['selected_residues']))]
    all_active_site_idx.sort()
    # print(all_active_site_idx)
    # print("Selected residues: ", selected_residues)
    pymol.cmd.delete('all')
    return all_active_site_idx
    

def predict_one(pdb_id, sitemap_workspace, pdb_file_path, distance, top_n):
    this_sample_workspace = os.path.join(sitemap_workspace, f'{pdb_id}')
    os.makedirs(this_sample_workspace, exist_ok=True)


    pdb_file_name = f'AF-{pdb_id}-F1-model_v4'
    pdb_file = os.path.abspath(os.path.join(pdb_file_path, f'{pdb_file_name}.pdb'))
    copy_pdb_file = os.path.abspath(os.path.join(this_sample_workspace, f'{pdb_file_name}.pdb'))
    shutil.copy(pdb_file, copy_pdb_file)
    
    abs_path_file_head = copy_pdb_file.split('.')[0]
    
    prepration_command = f'prepwizard -rehtreat -disulfides -fillsidechains -fillloops {pdb_file_name}.pdb {pdb_file_name}_protein_0.maegz'
    if not os.path.exists(f'{abs_path_file_head}_protein_0.maegz'):
        subprocess.run(prepration_command, shell=True, cwd=this_sample_workspace, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while not os.path.exists(f'{abs_path_file_head}_protein_0.maegz'):
            pass
    if not os.path.exists(f'{abs_path_file_head}_protein_0_out.maegz'):
        sitemap_command = f'sitemap -prot {pdb_file_name}_protein_0.maegz -keeplogs'
        subprocess.run(sitemap_command, shell=True, cwd=this_sample_workspace, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while not os.path.exists(f'{abs_path_file_head}_protein_0_out.maegz'):
            pass
    if not os.path.exists(f'{abs_path_file_head}_protein_0_out-1.pdb'):
        structconvert_command = f'structconvert {pdb_file_name}_protein_0_out.maegz {pdb_file_name}_protein_0_out.pdb'
        subprocess.run(structconvert_command, shell=True, cwd=this_sample_workspace, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        while not os.path.exists(f'{abs_path_file_head}_protein_0_out-1.pdb'):
            pass
    
    active_site = get_active_site(f'{pdb_file_name}_protein_0_out', file_path=this_sample_workspace, distance=distance, top_n=top_n)
    return set(active_site)

def predict_activate_site_with_sitemap(test_dataset, sitemap_workspace, pdb_file_path, distance=3, top_n=1, scoring=True, output_results=False):
    predicted_activate_sites = []
    accuracy_list = []
    precision_list = []
    specificity_list = []
    overlap_scores_list = []
    false_positive_rates_list = []
    f1_scores_list = []
    mcc_scores_list = []
    pbar = tqdm(test_dataset.iterrows(), total=len(test_dataset))
    for i, row in pbar:
        sequence_id = row['alphafolddb-id']
        aa_sequence = row['aa_sequence']
        active_site_gt = eval(row['site_labels'])
        active_site_gt_bin = get_active_site_binary(active_site_gt,
                                                    len(aa_sequence),
                                                    begain_zero=False)
        active_site_gt = set(
            np.argwhere(active_site_gt_bin == 1).reshape(-1).tolist())
        
        merge_predicted_results = predict_one(sequence_id, sitemap_workspace=sitemap_workspace, pdb_file_path=pdb_file_path, distance=distance, top_n=top_n)
        predicted_activate_sites.append(merge_predicted_results)
        if scoring:
            acc, prec, spec, overlap_score, fpr, f1, mcc = calculate_score(
                merge_predicted_results, active_site_gt, len(aa_sequence))
            accuracy_list.append(acc)
            precision_list.append(prec)
            specificity_list.append(spec)
            overlap_scores_list.append(overlap_score)
            false_positive_rates_list.append(fpr)
            f1_scores_list.append(f1)
            mcc_scores_list.append(mcc)
            pbar.set_description(
                'Accuracy: {:.4f}, Precision: {:.4f}, Specificity: {:.4f}, Overlap Score: {:.4f}, False Positive Rate: {:.4f}, F1: {:.4f}, MCC: {:.4f}'
                .format(
                    sum(accuracy_list) / len(accuracy_list),
                    sum(precision_list) / len(precision_list),
                    sum(specificity_list) / len(specificity_list),
                    sum(overlap_scores_list) / len(overlap_scores_list),
                    sum(false_positive_rates_list) /
                    len(false_positive_rates_list),
                    sum(f1_scores_list) / len(f1_scores_list),
                    sum(mcc_scores_list) / len(mcc_scores_list),
                    ))
    if scoring:
        print(f'Get {len(overlap_scores_list)} results')
        print(
            'Accuracy: {:.4f}, Precision: {:.4f}, Specificity: {:.4f}, Overlap Score: {:.4f}, False Positive Rate: {:.4f}, F1: {:.4f}, MCC: {:.4f}'
            .format(
                sum(accuracy_list) / len(accuracy_list),
                sum(precision_list) / len(precision_list),
                sum(specificity_list) / len(specificity_list),
                sum(overlap_scores_list) / len(overlap_scores_list),
                sum(false_positive_rates_list) /
                len(false_positive_rates_list),
                sum(f1_scores_list) / len(f1_scores_list),
                sum(mcc_scores_list) / len(mcc_scores_list),
                ))
    if output_results:
        test_dataset['predict_active_label'] = predicted_activate_sites
        test_dataset['accuracy'] = accuracy_list
        test_dataset['precision'] = precision_list
        test_dataset['specificity'] = specificity_list
        test_dataset['overlap_scores'] = overlap_scores_list
        test_dataset['false_positive_rates'] = false_positive_rates_list
        test_dataset['f1_scores'] = f1_scores_list
        test_dataset['mcc_scores'] = mcc_scores_list
        return output_results
    
    return predicted_activate_sites, overlap_scores_list, false_positive_rates_list
        

# %%
sitemap_workspace = './sitemap_workspace'
sitemap_workspace = os.path.abspath(sitemap_workspace)
pdb_file_path = '../../dataset/ec_site_dataset/structures/alphafolddb_download'
pdb_file_path = os.path.abspath(pdb_file_path)
test_dataset_path = '../../dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100/test_dataset/uniprot_ecreact_merge.csv'
dataset_path = '../../dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100'
test_dataset_path = os.path.abspath(test_dataset_path)

# %%
test_dataset = get_query_database(test_dataset_path, pdb_file_path=os.path.join(os.path.dirname(dataset_path), 'structures', 'alphafolddb_download'))
test_dataset

# %%
def merge_similarity_index(dataset_from_loader, dataset_with_similarity_index):
    dataset_with_similarity_index['alphafolddb-id'] = dataset_with_similarity_index['alphafolddb-id'].apply(lambda x:x.split(';')[0])
    dataset_with_similarity_index = dataset_with_similarity_index[[ 'alphafolddb-id', 'similarity_index_level', 'max_tmscore']]
    assert dataset_with_similarity_index['alphafolddb-id'].tolist() == dataset_from_loader['alphafolddb-id'].tolist()
    dataset_from_loader['similarity_index_level'] = dataset_with_similarity_index['similarity_index_level']
    dataset_from_loader['max_tmscore'] = dataset_with_similarity_index['max_tmscore']
    return dataset_from_loader

# %%
test_dataset_with_similarity_index = pd.read_csv(os.path.join(dataset_path, 'test_dataset_with_similarity_idx.csv'))
test_dataset = merge_similarity_index(test_dataset, test_dataset_with_similarity_index)

# %%
# predict_one('Q9F0J6', sitemap_workspace=sitemap_workspace, pdb_file_path=pdb_file_path, distance=3)

test_dataset_with_results = predict_activate_site_with_sitemap(test_dataset, sitemap_workspace, pdb_file_path, distance=3, top_n=5, scoring=True, output_results=True)
os.makedirs('baseline_results', exist_ok=True)
test_dataset_with_results.to_csv(os.path.join('baseline_results', 'schrodinger_sitemap.csv'), index=False)
test_dataset_with_results.to_json(os.path.join('baseline_results', 'schrodinger_sitemap.json'))

# %%



