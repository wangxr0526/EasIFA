from collections import defaultdict
import pandas as pd
import os
import sys
from sklearn.metrics import matthews_corrcoef, multilabel_confusion_matrix, recall_score
from tqdm.auto import tqdm
import numpy as np

sys.path.append('../../')
from dataset_preprocess.pdb_preprocess_utils import get_active_site_types, map_active_site_for_one, get_active_site_binary, map_active_site_type_for_one


def get_blast_database(dir, fasta_path=None):
    database_df = pd.DataFrame()
    csv_fnames = os.listdir(dir)
    pbar = tqdm(csv_fnames, total=len(csv_fnames))
    for fname in pbar:
        df = pd.read_csv(os.path.join(dir, fname))
        df = df[['alphafolddb-id', 'aa_sequence', 'site_labels', 'site_types']]
        database_df = pd.concat([database_df, df])

    database_df = database_df.drop_duplicates(
        subset=['alphafolddb-id', 'aa_sequence', 'site_labels', 'site_types'
                ]).reset_index(drop=True)
    database_df['alphafolddb-id'] = database_df['alphafolddb-id'].apply(
        lambda x: x.replace(';', ''))
    if fasta_path:
        with open(fasta_path, 'w', encoding='utf-8') as f:
            for idx, row in tqdm(database_df.iterrows(),
                                 total=len(database_df)):
                f.write('>{}\n'.format(row['alphafolddb-id']))
                f.write('{}\n'.format(row['aa_sequence']))
    return database_df


def get_query_database(path, fasta_path=None):
    database_df = pd.read_csv(path)
    database_df = database_df[[
        'alphafolddb-id', 'aa_sequence', 'site_labels', 'site_types'
    ]]
    database_df = database_df.drop_duplicates(
        subset=['alphafolddb-id', 'aa_sequence', 'site_labels', 'site_types'
                ]).reset_index(drop=True)
    database_df['alphafolddb-id'] = database_df['alphafolddb-id'].apply(
        lambda x: x.replace(';', ''))
    if fasta_path:
        with open(fasta_path, 'w', encoding='utf-8') as f:
            for idx, row in tqdm(database_df.iterrows(),
                                 total=len(database_df)):
                f.write('>{}\n'.format(row['alphafolddb-id']))
                f.write('{}\n'.format(row['aa_sequence']))
    return database_df


def read_blast_results(path):
    column_headers = [
        "Query ID",
        "Subject ID",
        "% Identity",
        "Alignment Length",
        "Mismatches",
        "Gap Opens",
        "Query Start",
        "Query End",
        "Subject Start",
        "Subject End",
        "E-value",
        "Bit Score",
    ]
    results_df = pd.read_csv(path, sep='\t', header=None)
    results_df.columns = column_headers
    return results_df


def calculate_score(pred_idx, gt_idx, num_residue):
    amino = [i for i in range(num_residue)]

    TP = len(pred_idx.intersection(gt_idx))
    FP = len(pred_idx.difference(gt_idx))
    TN = len(set(amino).difference((gt_idx).union(pred_idx)))
    FN = len(gt_idx) - TP

    acc = (TP + TN) / (TP + TN + FP + FN)
    prec = TP / (TP + FP) if len(pred_idx) != 0 else 0
    spec = TN / (TN + FP) if (TN + FP) != 0 else 1

    overlap_score = TP / len(gt_idx) if len(gt_idx) != 0 else 0  # recall
    fpr = FP / (FP + TN)
    f1 = 2 * (prec * overlap_score) / (prec + overlap_score) if (
        prec + overlap_score) != 0 else 0
    # Calculate MCC
    mcc_denom = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    mcc = ((TP * TN) - (FP * FN)) / (mcc_denom**0.5) if mcc_denom != 0 else 0
    return acc, prec, spec, overlap_score, fpr, f1, mcc


def get_fpr(cm, class_idx):
    TN, FP, FN, TP = cm[class_idx].ravel()
    fpr = FP / (FP + TN)
    return fpr

def calculate_metrics_multi_class(pred, gt, num_site_types):
    metrics = defaultdict(list)
    recall_list = recall_score(gt, pred, average=None, labels=range(num_site_types), zero_division=0)
    cm = multilabel_confusion_matrix(gt, pred, labels=range(num_site_types))
    metrics['multi-class mcc'].append(matthews_corrcoef(gt, pred))
    for class_idx in range(num_site_types):
        metrics[f'recall_cls_{class_idx}'].append(recall_list[class_idx])
        metrics[f'fpr_cls_{class_idx}'].append(get_fpr(cm, class_idx=class_idx))
    
    return metrics

def predict_activate_site_with_sequence_alignment(test_dataset,
                                                  database,
                                                  blastp_results,
                                                  scoring=True,
                                                  top_n=5,
                                                  output_results=False):
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
        blastp_df = blastp_results.loc[blastp_results['Query ID'] ==
                                       sequence_id]
        blastp_df = blastp_df.sort_values(by=['E-value', '% Identity'],
                                          ascending=[True, False
                                                     ]).reset_index(drop=True)
        blastp_df = blastp_df[:top_n]
        predicted_results = []
        for j, brow in blastp_df.iterrows():
            subject_id = brow['Subject ID']
            obtained_result_df = database.loc[database['alphafolddb-id'] ==
                                              subject_id]
            if len(obtained_result_df) == 0: continue
            obtained_aa_sequence = obtained_result_df['aa_sequence'].tolist(
            )[0]
            if isinstance(obtained_result_df['site_labels'].tolist()[0], list):
                obtained_site_labels = obtained_result_df[
                    'site_labels'].tolist()[0]
            elif isinstance(obtained_result_df['site_labels'].tolist()[0],
                            str):
                obtained_site_labels = eval(
                    obtained_result_df['site_labels'].tolist()[0])
            else:
                continue

            predicted_active_site_bin = map_active_site_for_one(
                obtained_aa_sequence,
                aa_sequence,
                obtained_site_labels,
                begain_zero=False)
            predicted_active_site = set(
                np.argwhere(
                    predicted_active_site_bin == 1).reshape(-1).tolist())
            predicted_results.append(predicted_active_site)
        merge_predicted_results = set()
        for pred in predicted_results:
            merge_predicted_results.update(pred)
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
        return test_dataset
    return predicted_activate_sites, overlap_scores_list, false_positive_rates_list


def predict_activate_site_type_with_sequence_alignment(test_dataset,
                                                  database,
                                                  blastp_results,
                                                  scoring=True,
                                                  top_n=5,
                                                  output_results=False):
    predicted_activate_sites = []
    predicted_activate_sites_vec = []
    accuracy_list = []
    precision_list = []
    specificity_list = []
    overlap_scores_list = []
    false_positive_rates_list = []
    f1_scores_list = []
    mcc_scores_list = []
    
    multicls_metrics_collection = defaultdict(list)
    
    pbar = tqdm(test_dataset.iterrows(), total=len(test_dataset))
    for i, row in pbar:
        sequence_id = row['alphafolddb-id']
        aa_sequence = row['aa_sequence']
        active_site_gt = eval(row['site_labels'])
        active_site_type_gt = eval(row['site_types'])
        active_site_types_vec_gt = get_active_site_types(active_site_gt,
                                                   active_site_type_gt,
                                                    len(aa_sequence),
                                                    begain_zero=False)
        active_site_gt = set(
            np.argwhere(active_site_types_vec_gt != 0).reshape(-1).tolist())
        blastp_df = blastp_results.loc[blastp_results['Query ID'] ==
                                       sequence_id]
        blastp_df = blastp_df.sort_values(by=['E-value', '% Identity'],
                                          ascending=[True, False
                                                     ]).reset_index(drop=True)
        blastp_df = blastp_df[:top_n]
        predicted_results = []
        predicted_active_site_types_vec = []
        for j, brow in blastp_df.iterrows():
            subject_id = brow['Subject ID']
            obtained_result_df = database.loc[database['alphafolddb-id'] ==
                                              subject_id]
            if len(obtained_result_df) == 0: continue
            obtained_aa_sequence = obtained_result_df['aa_sequence'].tolist(
            )[0]
            if isinstance(obtained_result_df['site_labels'].tolist()[0], list):
                obtained_site_labels = obtained_result_df[
                    'site_labels'].tolist()[0]
                obtained_site_types = obtained_result_df[
                    'site_types'].tolist()[0]
            elif isinstance(obtained_result_df['site_labels'].tolist()[0],
                            str):
                obtained_site_labels = eval(
                    obtained_result_df['site_labels'].tolist()[0])
                obtained_site_types = eval(
                    obtained_result_df['site_types'].tolist()[0])
            else:
                continue

            predicted_active_site_types = map_active_site_type_for_one(
                obtained_aa_sequence,
                aa_sequence,
                obtained_site_labels,
                obtained_site_types,
                begain_zero=False)
            predicted_active_site = set(
                np.argwhere(
                    predicted_active_site_types != 0).reshape(-1).tolist())
            predicted_results.append(predicted_active_site)
            predicted_active_site_types_vec.append(predicted_active_site_types)
        merge_predicted_results = set()

        merge_predicted_active_site_types = np.zeros((len(aa_sequence), ))
        for pred, pred_vec in zip(predicted_results, predicted_active_site_types_vec):
            merge_predicted_results.update(pred)
            merge_predicted_active_site_types = np.where(merge_predicted_active_site_types==0, pred_vec, merge_predicted_active_site_types)
        predicted_activate_sites.append(merge_predicted_results)
        predicted_activate_sites_vec.append(merge_predicted_active_site_types)
        if scoring:
            acc, prec, spec, overlap_score, fpr, f1, mcc = calculate_score(
                merge_predicted_results, active_site_gt, len(aa_sequence))
            
            
            multicls_metrics = calculate_metrics_multi_class(merge_predicted_active_site_types, active_site_types_vec_gt, num_site_types=4)
            for key in multicls_metrics:
                multicls_metrics_collection[key] += multicls_metrics[key]
            
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
            
        multicls_cols = ['recall_cls_0', 'recall_cls_1', 'recall_cls_2', 'recall_cls_3', 'fpr_cls_0', 'fpr_cls_1', 'fpr_cls_2', 'fpr_cls_3', 'multi-class mcc']
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
        
        print('Multiclassfication Metrics:')
        multiclass_report_str = ['{}: {:.4f}'.format(key, sum(multicls_metrics_collection[key])/len(multicls_metrics_collection[key])) for key in  multicls_cols]
        print(', '.join(multiclass_report_str))
        
    if output_results:
        test_dataset['predict_active_label'] = predicted_activate_sites
        test_dataset['accuracy'] = accuracy_list
        test_dataset['precision'] = precision_list
        test_dataset['specificity'] = specificity_list
        test_dataset['overlap_scores'] = overlap_scores_list
        test_dataset['false_positive_rates'] = false_positive_rates_list
        test_dataset['f1_scores'] = f1_scores_list
        test_dataset['mcc_scores'] = mcc_scores_list
        for key in multicls_cols:
            test_dataset[key] = multicls_metrics_collection[key]
        return test_dataset
    return predicted_activate_sites, predicted_activate_sites_vec, overlap_scores_list, false_positive_rates_list


if __name__ == '__main__':
    top_n = 5
    dataset_path = '../../dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100'
    database = get_blast_database(os.path.join(dataset_path, 'train_dataset'))

    test_dataset = pd.read_csv(
        os.path.join(dataset_path, 'test_dataset',
                     'uniprot_ecreact_merge.csv'))
    test_dataset['alphafolddb-id'] = test_dataset['alphafolddb-id'].apply(
        lambda x: x.replace(';', ''))

    blast_results_path = os.path.join(dataset_path, 'blast_results.txt')
    blastp_results = read_blast_results(path=blast_results_path)

    # predicted_activate_sites, overlap_scores, false_positive_rates = predict_activate_site_with_sequence_alignment(
    #     test_dataset,
    #     database=database,
    #     blastp_results=blastp_results,
    #     top_n=top_n)
    
    predict_activate_site_type_with_sequence_alignment(
        test_dataset,
        database=database,
        blastp_results=blastp_results,
        top_n=top_n)
