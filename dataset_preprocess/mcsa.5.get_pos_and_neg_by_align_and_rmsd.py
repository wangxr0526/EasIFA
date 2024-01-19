# %%
import pandas as pd
import os
import shutil
from tqdm import tqdm
import random

# %%
def get_mcsa_normal_dataset(dir, subset=['train', 'valid'], flag='mcsa', read_new=False):
    dataset = pd.DataFrame()
    for dataset_flag in subset:
        sub_df = pd.read_csv(os.path.join(dir, f'{dataset_flag}_dataset' if not read_new else f'new_{dataset_flag}_dataset', f'{flag}_{dataset_flag}.csv'))
        sub_df['dataset_flag'] = [dataset_flag for _ in range(len(sub_df))]
        dataset = pd.concat([dataset, sub_df])
        
    return dataset
def site_labels_to_protected_positions(site_labels):
    protected_positions = []
    for one_site in site_labels:
        if len(one_site) == 1:
            protected_positions.append(one_site[0]-1)          # 这里的活性标签依然是从1开始算起，转为index需要减1
        elif len(one_site) == 2:
            b, e = one_site
            site_indices = [k - 1 for k in range(b, e+1)]
            protected_positions.extend(site_indices)
        else:
            raise ValueError(
                'The label of active site is not standard !!!')
    return protected_positions

def protected_positions_to_site_labels(protected_positions): 
    site_labels = [[x+1] for x in protected_positions]     # 互转加1
    return site_labels

# %%
# 修改生成的pdb为标准名称
# aug_dir_flag = 'mcsa_aug_20_mutation_rate_0.2_insertion_rate_0.1_deletion_rate_0.1_max_length_150_seed_123'
aug_dir_flag = 'mcsa_aug_40_mutation_rate_0.3_insertion_rate_0.1_deletion_rate_0.1_max_length_150_seed_123'

org_structure_dir = '../dataset/mcsa_fine_tune/structures/alphafolddb_download'
org_generated_pdb_dir = f'../dataset/mcsa_fine_tune/{aug_dir_flag}/esmfold_generated_aug_pdb'
fixed_generated_structure_dir = f'../dataset/mcsa_fine_tune/{aug_dir_flag}/esmfold_generated_aug_structures'
os.makedirs(fixed_generated_structure_dir, exist_ok=True)

not_exist_path = []
for fname in tqdm(os.listdir(org_generated_pdb_dir)):
    new_fname = fname.replace(' ', '')
    shutil.copyfile(os.path.join(org_generated_pdb_dir, fname), os.path.join(fixed_generated_structure_dir, new_fname))
    org_fname = 'AF-{}-F1-model_v4.pdb'.format(new_fname.split('_')[0].split('-')[0])
    org_pdb_path = os.path.join(org_structure_dir, org_fname)
    if not os.path.exists(os.path.join(fixed_generated_structure_dir, '{}.pdb'.format(new_fname.split('_')[0].split('-')[0]))):
        try:
            shutil.copyfile(org_pdb_path, os.path.join(fixed_generated_structure_dir, '{}.pdb'.format(new_fname.split('_')[0].split('-')[0])))
        except:
            if org_pdb_path not in not_exist_path:
                print(f'{org_pdb_path} not exist.')
                not_exist_path.append(org_pdb_path)

        
    
    
    

# %%
print(len(not_exist_path))

# %%
mcsa_test_set = get_mcsa_normal_dataset('../dataset/mcsa_fine_tune/normal_mcsa', subset=['test'])
display(mcsa_test_set)

test_not_exist_path = []

for structure_id in tqdm(mcsa_test_set['alphafolddb-id'].tolist()):
    structure_id = structure_id.replace(' ', '')
    org_fname = 'AF-{}-F1-model_v4.pdb'.format(structure_id)
    org_pdb_path = os.path.join(org_structure_dir, org_fname)
    try:
    
        shutil.copyfile(org_pdb_path, os.path.join(fixed_generated_structure_dir, '{}-c0.pdb'.format(structure_id)))
    
    except:
        test_not_exist_path.append(org_pdb_path)
    

    
print(len(test_not_exist_path))


# %%
# 此时已经得到突变之后的3d结构了，但是不一定每个突变都保留活性，按照RFdiffusion的标准，motif骨架和侧链残基rmsd<=1.5A则有活性，否则清空活性位点
mcsa_train_valid_set = get_mcsa_normal_dataset('../dataset/mcsa_fine_tune/normal_mcsa', subset=['train', 'valid'])
mcsa_train_valid_set

# %%
P07342_row = mcsa_train_valid_set.loc[mcsa_train_valid_set['alphafolddb-id'] == 'P07342']

# %%
aug_mcsa_train_valid_set = get_mcsa_normal_dataset(f'../dataset/mcsa_fine_tune/{aug_dir_flag}', subset=['train', 'valid'], flag='aug_mcsa')
aug_mcsa_train_valid_set

# %%
P07342_m1_row = aug_mcsa_train_valid_set.loc[aug_mcsa_train_valid_set['alphafolddb-id'] == 'P07342-c0_m1']
P07342_m1_row

# %%
import MDAnalysis as mda
from MDAnalysis.analysis import align

def calculate_rmsd(ref_pdb_path, target_pdb_path, ref_resid_ids, target_resid_ids, only_backbone=False):
    """
    Calculate the RMSD between specified residues of two protein structures.
    
    Parameters:
    - ref_pdb_path (str): Path to the reference PDB file.
    - target_pdb_path (str): Path to the target PDB file.
    - ref_resid_ids (list): List of residue IDs for the reference structure.
    - target_resid_ids (list): List of residue IDs for the target structure.
    
    Returns:
    float: RMSD value.
    """
    # Load structures
    ref = mda.Universe(ref_pdb_path)
    target = mda.Universe(target_pdb_path)
    
    # Convert resid lists to string
    ref_resid_str = ' '.join(map(str, ref_resid_ids))
    target_resid_str = ' '.join(map(str, target_resid_ids))
    
    # Align and calculate RMSD
    if only_backbone:
        alignment = align.AlignTraj(target, ref, select=(f"resid {target_resid_str} and backbone", f"resid {ref_resid_str}  and backbone"), in_memory=True)
    else:
        alignment = align.AlignTraj(target, ref, select=(f"resid {target_resid_str}", f"resid {ref_resid_str}"), in_memory=True)
    alignment.run()
    
    # Retrieve and return the RMSD
    rmsd = alignment.rmsd.item()
    return rmsd


# %%
# ref_pdb = "../dataset/mcsa_fine_tune/structures/alphafolddb_download/AF-P07342-F1-model_v4.pdb"
# target_pdb = f"../dataset/mcsa_fine_tune/{aug_dir_flag}/esmfold_generated_aug_pdb/P07342_m1.pdb"
# ref_resids = [139, 201, 202, 251, 582]
# target_resids = [11, 69, 70, 113, 408]

# rmsd = calculate_rmsd(ref_pdb, target_pdb, ref_resids, target_resids)

# print(f"RMSD: {rmsd:.2f} Å")


# %%
ref_pdb_dir = '../dataset/mcsa_fine_tune/structures/alphafolddb_download/'
target_pdb_dir = f'../dataset/mcsa_fine_tune/{aug_dir_flag}/esmfold_generated_aug_pdb/'

new_aug_save_path = f'../dataset/mcsa_fine_tune/{aug_dir_flag}/'

# max_pos_number_per_data_point = 6
max_pos_number_per_data_point = 12

all_pos_numbers = []
all_neg_numbers = []

for dataset_flag in ['train', 'valid', 'test']:
    this_flag_new_aug_save_path = os.path.join(new_aug_save_path, f'new_{dataset_flag}_dataset')
    os.makedirs(this_flag_new_aug_save_path, exist_ok=True)
    if dataset_flag != 'test':
        this_flag_mcsa_train_valid_set = mcsa_train_valid_set.loc[mcsa_train_valid_set['dataset_flag']==dataset_flag]
        this_flag_aug_mcsa_train_valid_set = aug_mcsa_train_valid_set.loc[aug_mcsa_train_valid_set['dataset_flag']==dataset_flag]
        
        this_flag_new_aug_mcsa_set = pd.DataFrame(columns=aug_mcsa_train_valid_set.columns)

        
        for uniprot_id, site_label in tqdm(zip(this_flag_mcsa_train_valid_set['alphafolddb-id'].tolist(), this_flag_mcsa_train_valid_set['site_labels'].tolist()), total=len(this_flag_mcsa_train_valid_set)):
            
            uniprot_id = uniprot_id.replace(' ', '')
            ref_pdb_path=os.path.join(ref_pdb_dir, f'AF-{uniprot_id}-F1-model_v4.pdb')
            if not os.path.exists(ref_pdb_path): continue
            
            this_aug_set = this_flag_aug_mcsa_train_valid_set.loc[this_flag_aug_mcsa_train_valid_set['alphafolddb-id'].str.startswith(uniprot_id)]
            site_label_plus1 = [x+1 for x in site_labels_to_protected_positions(eval(site_label))]
            

            
            this_pos_rows = []
            this_neg_rows = []
            
            for idx, row in this_aug_set.iterrows():
                aug_site_label_plus1 = [x+1 for x in site_labels_to_protected_positions(eval(row['site_labels']))]   # MDanalysis选择残基是从1开始的
                structure_id = row['alphafolddb-id']
                structure_id = structure_id.replace(' ', '')
                
                if structure_id == uniprot_id:
                    # this_flag_new_aug_mcsa_set.loc[len(this_flag_new_aug_mcsa_set)] = row.tolist()

                    this_pos_rows.append(row.tolist())
                else:
                    try:
                        rmsd = calculate_rmsd(
                            ref_pdb_path=ref_pdb_path,
                            target_pdb_path=os.path.join(target_pdb_dir, f'{structure_id}.pdb'),
                            ref_resid_ids=site_label_plus1,
                            target_resid_ids=aug_site_label_plus1,
                            )
                    except:
                        continue
                    
                    if rmsd <= 1.5:
                        # 包含活性位点的增强数据

                        this_pos_rows.append(row.tolist())
                    else:
                        # motif构象变化太大，看作没有活性位点，清空活性位点
                        row['site_labels'] = []
                        # this_flag_new_aug_mcsa_set.loc[len(this_flag_new_aug_mcsa_set)] = row.tolist()
                        this_neg_rows.append(row.tolist())
                        
            if len(this_pos_rows) > max_pos_number_per_data_point:
                this_pos_rows_to_df = random.sample(this_pos_rows, max_pos_number_per_data_point)
            else:
                this_pos_rows_to_df = this_pos_rows
            
            this_neg_rows_to_df = random.sample(this_neg_rows, min(2*len(this_pos_rows_to_df), len(this_neg_rows)))
            
            for row in this_pos_rows_to_df+this_neg_rows_to_df:
                this_flag_new_aug_mcsa_set.loc[len(this_flag_new_aug_mcsa_set)] = row
            
            all_pos_numbers.append(len(this_pos_rows_to_df))
            all_neg_numbers.append(len(this_neg_rows_to_df))
            
        this_flag_new_aug_mcsa_set.to_csv(os.path.join(this_flag_new_aug_save_path, f'aug_mcsa_{dataset_flag}.csv'), index=False)
    
    else:
        pd.read_csv(os.path.join(f'../dataset/mcsa_fine_tune/{aug_dir_flag}', 'test_dataset/aug_mcsa_test.csv')).to_csv(os.path.join(this_flag_new_aug_save_path, f'aug_mcsa_{dataset_flag}.csv'), index=False)
    
        

# %%
print(sum(all_pos_numbers)/len(all_pos_numbers))
print(sum(all_neg_numbers)/len(all_neg_numbers))

# %%
import numpy as np
(np.array(all_pos_numbers)==1).sum()


