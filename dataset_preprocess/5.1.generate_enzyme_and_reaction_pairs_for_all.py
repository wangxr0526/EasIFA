# %%
import pandas as pd
import os



# %%
uniprot_raw_path = '../dataset/raw_dataset/ec_datasets/uniprot_raw'
split_ec_dataset_save_path = '../dataset/raw_dataset/ec_datasets/split_ec_dataset'
ecreact_dataset_path =  os.path.join(os.path.dirname(uniprot_raw_path), 'ecreact-1.0.csv')

ecreact_dataset = pd.read_csv(ecreact_dataset_path)
download_uniprot_dataset_path = os.path.join(uniprot_raw_path, 'uniprot-download_sequence_site_ec_clean.pkl')

download_uniprot_dataset = pd.read_pickle(download_uniprot_dataset_path)

train_download_uniprot_dataset = pd.read_pickle(os.path.join(split_ec_dataset_save_path, 'train_ec_uniprot_dataset_cluster_sample.pkl'))

validation_dataset = pd.read_pickle(os.path.join(split_ec_dataset_save_path, 'validation_ec_uniprot_dataset_cluster_sample.pkl'))

uniprot_test_dataset = pd.read_pickle(os.path.join(split_ec_dataset_save_path, 'test_ec_uniprot_dataset_cluster_sample.pkl'))


# %% [markdown]
#  ## 匹配酶催化反应与酶序列，考虑所有酶的结构

# %%
def rm_ec_from_rxn_smiles(rxn):
    precursors, products = rxn.split('>>')
    reactants, ec_number = precursors.split('|')
    return f'{reactants}>>{products}'
ecreact_dataset['rxn_smiles_rm_ec'] = ecreact_dataset['rxn_smiles'].apply(lambda x:rm_ec_from_rxn_smiles(x))

# %%



# %%
all_ec_numbers = set(download_uniprot_dataset['EC number'].tolist()) | set(ecreact_dataset['ec'].tolist())
all_ec_numbers = list(all_ec_numbers)
all_ec_numbers.sort()


# %%
len(all_ec_numbers)


# %%
from tqdm import tqdm
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore")


def split_select_ecreact(df:pd.DataFrame):
    df = df.sample(frac=1).reset_index(drop=True)
    df_cnt = len(df)
    train_df = df.loc[:int(df_cnt*0.8), :]
    valid_df = df.loc[int(df_cnt*0.8):int(df_cnt*0.9), :]
    test_df = df.loc[int(df_cnt*0.9):, :]
    return train_df, valid_df, test_df

def merge_rxn_aa_sequence(rxn, aa_sequence):
    react, prod = rxn.split('>>')
    return f'{react}|{aa_sequence}>>{prod}'

def merge_uniprot_and_ecreact_all(uniprot_df:pd.DataFrame, ecreact_df:pd.DataFrame, max_sample=1, gen_test=False):
    
    # 此脚本不分割反应，与脚本5区分，反应其实可以看作是酶的特征
    
    if len(uniprot_df) == 0: return
    if len(ecreact_df) == 0: return
    uniprot_df_use = uniprot_df[['EC number', 'PDB', 'AlphaFoldDB', 'Sequence', 'site_labels',	'site_types']]
    if not gen_test:
        use_ecreact_number = max_sample
        ecreact_df_use = ecreact_df.sample(n=use_ecreact_number if use_ecreact_number<=len(ecreact_df) else len(ecreact_df), random_state=123)
        uniprot_df_use.loc[:, 'rxn_smiles_rm_ec'] = [ecreact_df_use['rxn_smiles_rm_ec'].tolist()] * len(uniprot_df_use)
        uniprot_df_use = uniprot_df_use.explode('rxn_smiles_rm_ec').reset_index(drop=True)
    else:
        if len(uniprot_df_use) >= len(ecreact_df):
            rxn_smiles_rm_ec_list = (ecreact_df['rxn_smiles_rm_ec'].tolist() * math.ceil(len(uniprot_df_use)/len(ecreact_df)))[:len(uniprot_df_use)]
            uniprot_df_use.loc[:, 'rxn_smiles_rm_ec'] = rxn_smiles_rm_ec_list
        else:
            rxn_smiles_rm_ec_list = ecreact_df['rxn_smiles_rm_ec'].tolist()[:len(uniprot_df_use)]
            uniprot_df_use.loc[:, 'rxn_smiles_rm_ec'] = rxn_smiles_rm_ec_list
        
            
    uniprot_df_use['rxn'] = uniprot_df_use.apply(lambda row:merge_rxn_aa_sequence(row['rxn_smiles_rm_ec'], row['Sequence']), axis=1)
    merge_df = uniprot_df_use[['rxn', 'EC number', 'PDB', 'AlphaFoldDB', 'Sequence', 'site_labels', 'site_types']]
    merge_df.columns = ['reaction', 'ec', 'pdb-id', 'alphafolddb-id', 'aa_sequence', 'site_labels', 'site_types']
    return merge_df

max_sample = 1          # 控制生成数据的上限
uniprot_ecreact_merge_dataset_path = f'../dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_all_limit_{max_sample}'
if not os.path.exists(uniprot_ecreact_merge_dataset_path):
    os.makedirs(uniprot_ecreact_merge_dataset_path)

uniprot_ecreact_merge_dataset_train_path = os.path.join(uniprot_ecreact_merge_dataset_path, 'train_dataset')
if not os.path.exists(uniprot_ecreact_merge_dataset_train_path):
    os.makedirs(uniprot_ecreact_merge_dataset_train_path)
uniprot_ecreact_merge_dataset_valid_path = os.path.join(uniprot_ecreact_merge_dataset_path, 'valid_dataset')
if not os.path.exists(uniprot_ecreact_merge_dataset_valid_path):
    os.makedirs(uniprot_ecreact_merge_dataset_valid_path)
uniprot_ecreact_merge_dataset_test_path = os.path.join(uniprot_ecreact_merge_dataset_path, 'test_dataset')
if not os.path.exists(uniprot_ecreact_merge_dataset_test_path):
    os.makedirs(uniprot_ecreact_merge_dataset_test_path)

all_train_alphafolddb_id = set()
all_merge_test_df = pd.DataFrame()
warning_list = []
train_cnt = 0
valid_cnt = 0
test_cnt = 0
for ec in tqdm(all_ec_numbers):
    select_uniprot_train = train_download_uniprot_dataset.loc[train_download_uniprot_dataset['EC number']==ec]
    select_uniport_valid = validation_dataset.loc[validation_dataset['EC number']==ec]
    select_uniprot_test = uniprot_test_dataset.loc[uniprot_test_dataset['EC number']==ec]
    
    
    select_ecreact_dataset = ecreact_dataset.loc[ecreact_dataset['ec']==ec]
    merge_train_df = merge_uniprot_and_ecreact_all(select_uniprot_train, select_ecreact_dataset, max_sample=max_sample)
    merge_valid_df = merge_uniprot_and_ecreact_all(select_uniport_valid, select_ecreact_dataset, max_sample=max_sample)
    merge_test_df = merge_uniprot_and_ecreact_all(select_uniprot_test, select_ecreact_dataset, max_sample=max_sample, gen_test=True)
    if merge_test_df is not None:
        all_merge_test_df = pd.concat([all_merge_test_df, merge_test_df], axis=0)
    

    save_name = f'uniprot_ecreact_merge_EC={ec}.csv'
    try:
        all_train_alphafolddb_id.update(merge_train_df['alphafolddb-id'])
        merge_train_df.to_csv(os.path.join(uniprot_ecreact_merge_dataset_train_path, save_name), index=False)
        train_cnt += len(merge_train_df)
    except:
        # print(f'Warning: train--{ec} is None!')
        warning_list.append(f'Warning: train--{ec} is None!')
    try:
        merge_valid_df.to_csv(os.path.join(uniprot_ecreact_merge_dataset_valid_path, save_name), index=False)
        valid_cnt += len(merge_valid_df)
    except:
        # print(f'Warning: valid--{ec} is None!')
        warning_list.append(f'Warning: valid--{ec} is None!')
all_merge_test_df.to_csv(os.path.join(uniprot_ecreact_merge_dataset_test_path, 'uniprot_ecreact_merge.csv'), index=False)
test_cnt += len(all_merge_test_df)
print(warning_list)
print('all merge train cnt:', train_cnt)
print('all merge valid cnt:', valid_cnt)
print('all merge test cnt:', test_cnt)

print('all train alphafolddb id cnt:', len(all_train_alphafolddb_id))

with open(os.path.join(uniprot_ecreact_merge_dataset_path, 'dataset_info.txt'), 'w', encoding='utf-8') as f:
    f.write(f'all merge train cnt: {train_cnt}\n')
    f.write(f'all merge valid cnt: {valid_cnt}\n')
    f.write(f'all merge test cnt: {test_cnt}\n')


# %%
merge_train_df

# %%
len(warning_list)


# %%






