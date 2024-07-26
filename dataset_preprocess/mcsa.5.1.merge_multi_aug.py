# %%
import os
import pandas as pd
import re
from tqdm import tqdm
import shutil

# %% [markdown]
# 这个代码是用来合并多次生成的增强数据


# %%

def startswith_in_db_id(db_id, uniprot_id):
    if uniprot_id in db_id:
        return True
    else:
        return False
    
def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    
    for item in tqdm(os.listdir(src), desc='Coping PDB files'):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)

# %%
aug_mcsa_dataset_path = '../dataset/mcsa_fine_tune/'

source_aug_data_flag_list = [
    'mcsa_aug_20_mutation_rate_0.2_insertion_rate_0.1_deletion_rate_0.1_max_length_150_seed_123',
    'mcsa_aug_40_mutation_rate_0.35_insertion_rate_0.1_deletion_rate_0.1_max_length_150_seed_123'
]

target_aug_data_flag = 'mcsa_aug_20+40_mutation_rate_0.2+0.35_insertion_rate_0.1_deletion_rate_0.1_max_length_150_seed_123'

# %%
target_aug_data_path = os.path.join(aug_mcsa_dataset_path, target_aug_data_flag)
target_aug_data_structure_path = os.path.join(target_aug_data_path, 'esmfold_generated_aug_structures')


os.makedirs(target_aug_data_path, exist_ok=True)
os.makedirs(target_aug_data_structure_path, exist_ok=True)


for dataset_flag in ['train', 'valid', 'test']:
    
    this_dataset_flag_df = pd.DataFrame(columns=['reaction', 'ec', 'alphafolddb-id', 'aa_sequence', 'site_labels', 'site_types', 'cluster', 'ec_level1', 'dataset_flag'])
    
    this_dataset_flag_path = os.path.join(target_aug_data_path, f'new_{dataset_flag}_dataset')
    os.makedirs(this_dataset_flag_path, exist_ok=True)
    
    
    for aug_data_flag in source_aug_data_flag_list:
        this_aug_data_flag_df = pd.read_csv(os.path.join(
            aug_mcsa_dataset_path, 
            aug_data_flag, 
            f'new_{dataset_flag}_dataset', 
            f'aug_mcsa_{dataset_flag}.csv'
            )
                                   )
        
        this_aug_data_structure_path = os.path.join(aug_mcsa_dataset_path, aug_data_flag, 'esmfold_generated_aug_structures')
        if this_dataset_flag_df.empty:
            this_dataset_flag_df = this_aug_data_flag_df
            
            copytree(this_aug_data_structure_path, target_aug_data_structure_path)
            continue
        for idx, row in tqdm(this_aug_data_flag_df.iterrows(), total=len(this_aug_data_flag_df)):
            uniprot_id = row['alphafolddb-id'].split('-')[0]
            
            already_includ_data_df = this_dataset_flag_df.loc[this_dataset_flag_df['alphafolddb-id'].apply(lambda x: startswith_in_db_id(x, uniprot_id=uniprot_id))]
            
            already_includ_sequences = already_includ_data_df['aa_sequence'].tolist()
            if row['aa_sequence'] not in already_includ_sequences:
                org_db_id = row['alphafolddb-id']
                this_row_db_id = org_db_id + '0'
                while this_row_db_id in already_includ_data_df['alphafolddb-id'].tolist():
                    this_row_db_id += '0'
                row['alphafolddb-id'] = this_row_db_id
                
                src_pdb_path = os.path.join(this_aug_data_structure_path, f'{org_db_id}.pdb')
                tgt_pdb_path = os.path.join(target_aug_data_structure_path, f'{this_row_db_id}.pdb')
                shutil.copy(src_pdb_path, tgt_pdb_path)
                
                this_dataset_flag_df.loc[len(this_dataset_flag_df)] = row
            
        this_dataset_flag_df.to_csv(os.path.join(this_dataset_flag_path, f'aug_mcsa_{dataset_flag}.csv'), index=False)
     


