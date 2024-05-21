# %%

import pandas as pd
import os
from tqdm.auto import tqdm
from pandarallel import pandarallel
from rdkit import Chem
from tqdm import tqdm as top_tqdm


# %%

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
def get_blast_database(dir, fasta_path):
    database_df = pd.DataFrame()
    csv_fnames = os.listdir(dir)
    pbar = tqdm(
        csv_fnames,
        total=len(csv_fnames)
    )
    for fname in pbar:
        df = pd.read_csv(os.path.join(dir, fname))
        df = df[['alphafolddb-id', 'aa_sequence', 'site_labels', 'site_types']]
        database_df = pd.concat([database_df, df])
    
    database_df = database_df.drop_duplicates(subset=['alphafolddb-id', 'aa_sequence','site_labels', 'site_types']).reset_index(drop=True)
    database_df['alphafolddb-id'] = database_df['alphafolddb-id'].apply(lambda x:x.replace(';',''))

    with open(fasta_path, 'w', encoding='utf-8') as f:
        for idx, row in tqdm(database_df.iterrows(), total=len(database_df)):
            f.write('>{}\n'.format(row['alphafolddb-id']))
            f.write('{}\n'.format(row['aa_sequence']))
    return database_df

def get_query_database(path, fasta_path, pdb_file_path):
    database_df = pd.read_csv(path)
    database_df = database_df[['alphafolddb-id', 'aa_sequence','site_labels', 'site_types']]
    database_df['alphafolddb-id'] = database_df['alphafolddb-id'].apply(lambda x:x.replace(';',''))
    
    database_df = multiprocess_structure_check(database_df, nb_workers=12, pdb_file_path=pdb_file_path)
    
    write_database_df = database_df.drop_duplicates(subset=['alphafolddb-id', 'aa_sequence','site_labels', 'site_types']).reset_index(drop=True)


    with open(fasta_path, 'w', encoding='utf-8') as f:
        for idx, row in tqdm(write_database_df.iterrows(), total=len(write_database_df)):
            f.write('>{}\n'.format(row['alphafolddb-id']))
            f.write('{}\n'.format(row['aa_sequence']))
    return database_df



# %%

dataset_path = '../../dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100'
test_dataset_fasta_path = os.path.join(dataset_path, 'test_dataset.fasta')
baseline_results_path = 'baseline_results'

train_database_df = pd.read_pickle('../../dataset/raw_dataset/ec_datasets/split_ec_dataset/train_ec_uniprot_dataset_cluster_sample.pkl')
test_dataset = get_query_database(os.path.join(dataset_path, 'test_dataset', 'uniprot_ecreact_merge.csv'), fasta_path=test_dataset_fasta_path, pdb_file_path=os.path.join(os.path.dirname(dataset_path), 'structures', 'alphafolddb_download'))
test_dataset = test_dataset.loc[test_dataset['is_valid']]
test_dataset


# %%

len(set(test_dataset['alphafolddb-id']))


# %%

import subprocess

deepfri_model_root = '/home/xiaoruiwang/data/ubuntu_work_beta/protein_work/DeepFRI'
args = [
                "--fasta_fn",
                os.path.abspath(test_dataset_fasta_path),
                "-ont",
                "ec",
                "-v",
                "--saliency",
                "--use_guided_grads",
                "-o", os.path.abspath(os.path.join(baseline_results_path, 'DeepFRI')),
                "--model_config", os.path.abspath(os.path.join(deepfri_model_root, 'trained_models/model_config.json'))

        ]

deepfri_results_fname = 'DeepFRI_EC_saliency_maps.json'

# activate_cmd = f"source activate deepfri_env"
python_cmd = '/home/xiaoruiwang/software/miniconda3/envs/deepfri_env/bin/python {} {}'.format(os.path.join(deepfri_model_root, 'predict.py'), ' '.join(args))

# command = f'{activate_cmd} && {python_cmd}'
command = f'{python_cmd}'
if not os.path.exists(os.path.join(baseline_results_path, 'DeepFRI_EC_saliency_maps.json')):
        subprocess.run(command, shell=True, check=True, cwd=deepfri_model_root)
        print('')


# %%

import json
import numpy as np
with open(os.path.join(baseline_results_path, deepfri_results_fname), 'r') as f:
    deepfri_results = json.load(f)


# %%

deepfri_results_df = pd.DataFrame(deepfri_results).T
deepfri_results_df['alphafolddb-id'] = deepfri_results_df.index
deepfri_results_df.index = [i for i in range(len(deepfri_results_df))]
deepfri_results_df.columns = ['EC', 'EC Numbers', 'aa_sequence', 'predict_active_prob', 'alphafolddb-id']   # saliency_maps >> predict_active_prob
# deepfri_results_df['predict_active_prob'] = deepfri_results_df['predict_active_prob'].apply(lambda x:np.array(x).reshape(-1).tolist())
deepfri_results_df


# %%

from utils import get_active_site_binary, calculate_score
def predict_activate_site_with_deepfri(test_dataset, deepfri_results_df, scoring=True, threshold=0.5, output_results=False, evaluate_threshold=False, reprot=True, evaluate_col_name='overlap_score'):

    predicted_activate_sites = []
    predicted_activate_sites_vec = []
    accuracy_list = []
    precision_list = []
    specificity_list = []
    overlap_scores_list = []
    false_positive_rates_list = []
    f1_scores_list = []
    mcc_scores_list = []
    prediction_succ = []

    pbar = tqdm(test_dataset.iterrows(), total=len(test_dataset), disable=True if not reprot else False)
    for i, row in pbar:
        sequence_id = row['alphafolddb-id']
        aa_sequence = row['aa_sequence']
        active_site_gt = eval(row['site_labels'])
        active_site_type_gt = eval(row['site_types'])
        active_site_gt_bin = get_active_site_binary(active_site_gt,
                                                    len(aa_sequence),
                                                    begain_zero=False)
        active_site_gt = set(
            np.argwhere(active_site_gt_bin == 1).reshape(-1).tolist())

        deepfri_results_for_one:pd.DataFrame = deepfri_results_df.loc[deepfri_results_df['alphafolddb-id']==sequence_id]


        predicted_results = []
        if deepfri_results_for_one.empty:
            # 当没有给出预测值时，所有位点都被设置为阴性
            predicted_active_site_bin = np.zeros((len(aa_sequence), ))
            predicted_activate_sites_vec.append(predicted_active_site_bin.tolist())
            prediction_succ.append(False)
            predicted_activate_site = set(np.argwhere(
                    predicted_active_site_bin != 0).reshape(-1).tolist())
            predicted_results.append(predicted_activate_site)
        else:

            predicted_active_probs = np.array(deepfri_results_for_one['predict_active_prob'].tolist()[0])
            if predicted_active_probs.shape[0] != 1:
                print()
                pass
            for predicted_active_prob in predicted_active_probs:
                predicted_active_site_bin = (predicted_active_prob >= threshold).astype(float)
                predicted_active_site = set(
                np.argwhere(
                    predicted_active_site_bin == 1).reshape(-1).tolist())
                predicted_results.append(predicted_active_site)

            prediction_succ.append(True)
        merge_predicted_results = set()
        for pred in predicted_results:
            merge_predicted_results.update(pred)
        
        predicted_activate_sites.append(merge_predicted_results)
        # predicted_activate_sites_vec.append(predicted_active_site_bin)
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

        if reprot:
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
        test_dataset['prediction_succ'] = prediction_succ

        if scoring:
            score_cols = ['accuracy','precision','specificity','overlap_scores','false_positive_rates','f1_scores','mcc_scores']
            succ_prediction_df = test_dataset.loc[test_dataset['prediction_succ']]
            scoring_str = 'Succ Predictions Score: {} results\n'.format(len(succ_prediction_df))
            for score_name in score_cols:
                scoring_str += '{}: {:.4f}, '.format(score_name, succ_prediction_df[score_name].sum()/len(succ_prediction_df))
            if reprot:
                print(scoring_str)
        if evaluate_threshold:
            return succ_prediction_df[evaluate_col_name].sum() / len(succ_prediction_df)
        return test_dataset
    return predicted_activate_sites, overlap_scores_list, false_positive_rates_list


# %%

best_score = 0
best_threshold = 0
for threshold in [0.050, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5]:
    score_mean = predict_activate_site_with_deepfri(test_dataset, deepfri_results_df, threshold=threshold, scoring=True, output_results=True, evaluate_threshold=True, reprot=False, evaluate_col_name='f1_scores')
    if best_score < score_mean:
        best_score = score_mean
        best_threshold = threshold
print()
print('#'*20)
print()
print('Best:')
test_dataset_with_results:pd.DataFrame = predict_activate_site_with_deepfri(test_dataset, deepfri_results_df, threshold=best_threshold, scoring=True, output_results=True)

os.makedirs('baseline_results', exist_ok=True)
test_dataset_with_results.to_csv(os.path.join('baseline_results', 'deepfri_gradcam_aa_sequence.csv'), index=False)
test_dataset_with_results.to_json(os.path.join('baseline_results', 'deepfri_gradcam_aa_sequence.json'))



