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
blast_database_path = '../../dataset/raw_dataset/uniprot/uniprot_sprot.fasta'
train_database_df = pd.read_pickle('../../dataset/raw_dataset/ec_datasets/split_ec_dataset/train_ec_uniprot_dataset_cluster_sample.pkl')
test_dataset = get_query_database(os.path.join(dataset_path, 'test_dataset', 'uniprot_ecreact_merge.csv'), fasta_path=os.path.join(dataset_path, 'test_dataset.fasta'), pdb_file_path=os.path.join(os.path.dirname(dataset_path), 'structures', 'alphafolddb_download'))



# %%
train_database_df['alphafolddb-id'] = train_database_df['AlphaFoldDB'].apply(lambda x:x.replace(';',''))
train_database_df['aa_sequence'] = train_database_df['Sequence'].apply(lambda x:x)
train_database_df

# %%
test_dataset = test_dataset.loc[test_dataset['is_valid']]
test_dataset

# %%
import subprocess

database_fasta = blast_database_path
database = os.path.join(os.path.dirname(database_fasta), 'uniprot_sprot')
command = f'makeblastdb -in {database_fasta} -dbtype prot -out {database}'
subprocess.run(command, shell=True)

# %%
query_file = os.path.join(dataset_path, 'test_dataset.fasta')
output_file = os.path.join(dataset_path, 'blast_results_use_uniprot_sprot.txt')
command = f'blastp -query {query_file} -db {database} -out {output_file} -evalue 0.001 -outfmt 6 -num_threads 16'
if not os.path.exists(output_file):
    subprocess.run(command, shell=True)
# subprocess.run(command, shell=True)

# %%
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
    results_df['Subject ID'] = results_df['Subject ID'].apply(lambda x:x.split('|')[1])
    results_df = results_df.loc[results_df['Query ID']!=results_df['Subject ID']] # 把测试集删除，防止数据泄露
    return results_df


# %%
blast_p_results = read_blast_results(path=output_file)
blast_p_results

# %%
print(blast_p_results['% Identity'].max())
print(blast_p_results['% Identity'].min())
print(blast_p_results['% Identity'].mean())

# %%
import sys
sys.path.append('../../')
from dataset_preprocess.pdb_preprocess_utils import map_active_site_for_one
from utils import predict_activate_site_with_sequence_alignment, predict_activate_site_type_with_sequence_alignment

# %%
predicted_activate_sites, overlap_scores, false_positive_rates = predict_activate_site_with_sequence_alignment(test_dataset, database=train_database_df, blastp_results=blast_p_results, top_n=5)

# %%
predicted_activate_sites, predicted_activate_sites_vec, overlap_scores_list, false_positive_rates_list = predict_activate_site_type_with_sequence_alignment(test_dataset, database=train_database_df, blastp_results=blast_p_results, top_n=5)


