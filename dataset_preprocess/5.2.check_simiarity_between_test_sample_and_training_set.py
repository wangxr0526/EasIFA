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



# %%
import subprocess

# 定义CD-HIT命令和参数

def run_subprocess_and_print(args):
    # 开启一个新的进程
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        # 实时读取输出
        while True:
            output = proc.stdout.readline()
            if output == '' and proc.poll() is not None:
                break
            if output:
                print(output.strip())  # 输出每一行
        # 检查错误输出
        err = proc.stderr.read()
        if err:
            print("Error:", err)
    proc.communicate()

threshold = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
command = "/home/xiaoruiwang/software/cdhit/cd-hit"
for thre in threshold:

    if thre < 0.5:
        word = 2
    elif thre < 0.6:
        word = 3
    elif thre < 0.7:
        word = 4
    else:
        word = 5
    args = ["-i", os.path.abspath(os.path.join(uniprot_raw_path, 'all_sequence.fasta')),
            "-o", os.path.abspath(os.path.join(uniprot_raw_path, f'sequence_cutoff_{str(thre)}.fasta')), 
            "-c", str(thre),
            "-n", str(word),
            "-T", str(12)]
    print('Args: {}'.format(' '.join(args)))
    if not os.path.exists(os.path.abspath(os.path.join(uniprot_raw_path, f'sequence_cutoff_{str(thre)}.fasta'))):
        run_subprocess_and_print([command] + args)

# %%
from collections import defaultdict
def get_cluster(cluster_results_path):
    results = defaultdict(list)
    with open(cluster_results_path, 'r', encoding='utf-8') as f:
        data = [x.strip() for x in f.readlines()]
    for line in data:
        if '>Cluster' in line:
            cluster = line.replace('>', '')
            continue
        alphadb_id = line.split('>')[-1].split('.')[0]
        results[cluster].append(alphadb_id)
    return results


def id2cluster(cluster_results:dict):
    alphafold_id2cluster = {}
    for key in cluster_results:
        for id in cluster_results[key]:
            alphafold_id2cluster[id] = key
    return alphafold_id2cluster


# %%
threshold = threshold
alphafold_id2cluster_all_levels = {}
cluster_all_levels2alphafold_id = {}
for thre in threshold:
    cluster_results_path = os.path.abspath(os.path.join(uniprot_raw_path, f'sequence_cutoff_{str(thre)}.fasta.clstr'))
    cluster_results = get_cluster(cluster_results_path)
    print(len(cluster_results))
    alphafold_id2cluster_all_levels[thre] = id2cluster(cluster_results)
    cluster_all_levels2alphafold_id[thre] = cluster_results

# %%
alphafold_id2cluster_all_levels[thre]

# %%
max_sample = 100     
merge_dataset_name_str = 'uniprot_ecreact_cluster_split_merge_dataset_limit_'
ec_site_dataset_path = '../dataset/ec_site_dataset/'

# %%
from tqdm import tqdm
import pandas as pd
import os
from tqdm.auto import tqdm
from pandarallel import pandarallel
from rdkit import Chem
from tqdm import tqdm as top_tqdm
def get_dataset(max_sample, ec_site_dataset_path=ec_site_dataset_path,  
                merge_dataset_name_str='uniprot_ecreact_merge_dataset_limit_', 
                sub_set=['train', 'valid', 'test']):
    
    uniprot_ecreact_merge_dataset_path = os.path.join(ec_site_dataset_path, f'{merge_dataset_name_str}{max_sample}')
    dataset = pd.DataFrame()

    for dataset_flag in sub_set:
        folder_path = os.path.join(uniprot_ecreact_merge_dataset_path, f'{dataset_flag}_dataset')
        csv_fnames = os.listdir(folder_path)
        pbar = tqdm(
            csv_fnames,
            total=len(csv_fnames),
            desc=f'{dataset_flag}'
        )


        for fname in pbar:
            df = pd.read_csv(os.path.join(folder_path, fname))
            # df = df[['alphafolddb-id', 'aa_sequence']]
            dataset = pd.concat([dataset, df])
            

    # info_df = info_df.drop_duplicates(subset=['alphafolddb-id', 'aa_sequence']).reset_index(drop=True)
    return dataset
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
train_dataset = get_dataset(max_sample, ec_site_dataset_path=ec_site_dataset_path, merge_dataset_name_str=merge_dataset_name_str, sub_set=['train'])
# test_dataset = get_dataset(max_sample, ec_site_dataset_path=ec_site_dataset_path, merge_dataset_name_str=merge_dataset_name_str, sub_set=['test'])

dataset_path = os.path.join(ec_site_dataset_path, f'{merge_dataset_name_str}{max_sample}')
test_dataset = get_query_database(os.path.join(dataset_path, 'test_dataset', 'uniprot_ecreact_merge.csv'), fasta_path=os.path.join(dataset_path, 'test_dataset.fasta'), pdb_file_path=os.path.join(os.path.dirname(dataset_path), 'structures', 'alphafolddb_download'))
test_dataset = test_dataset.loc[test_dataset['is_valid']].reset_index(drop=True)
test_dataset

# %%
for thre in threshold:
    train_dataset[f'cluster_ther_{str(thre)}'] = train_dataset['alphafolddb-id'].apply(lambda x:alphafold_id2cluster_all_levels[thre][x.replace(';', '')])
    test_dataset[f'cluster_ther_{str(thre)}'] = test_dataset['alphafolddb-id'].apply(lambda x:alphafold_id2cluster_all_levels[thre][x.replace(';', '')])

# %%
train_dataset.head()

# %%
test_dataset.head()

# %%
similarity_index_levels = ['0~40%', '40~60%', '60~80%']
def get_similarity_index_level(level1_cls, level2_cls, level3_cls, train_dataset):

    if (level3_cls not in train_dataset['cluster_ther_0.4'].tolist()):  # 在0.4阈值下与训练集没有相同的cluster，则代表相似度小于0.4
        return similarity_index_levels[0]  # 0~0.4
    else: # 在0.4阈值下，与训练集有相同的cluster，则相似度大于0.4
        if (level2_cls not in train_dataset['cluster_ther_0.6'].tolist()): # 在0.6阈值下与训练集没有相同的cluster，则代表相似度小于0.6
            return similarity_index_levels[1] # 0.4~0.6
        else:
            return similarity_index_levels[2] # 0.6~0.8

# %%
similarity_index_levels_6 = ['0~30%', '30~40%', '40~50%', '50~60%', '60~70%', '70~80%']
def get_similarity_index_level_6(cluster_80, cluster_70, cluster_60, cluster_50, cluster_40, cluster_30, train_dataset):

    if (cluster_30 not in train_dataset['cluster_ther_0.3'].tolist()):  # 在0.2阈值下与训练集没有相同的cluster，则代表相似度小于0.3
        return similarity_index_levels_6[0]  # 0~0.3
    else: # 在0.2阈值下，与训练集有相同的cluster，则相似度大于0.3
        if (cluster_40 not in train_dataset['cluster_ther_0.4'].tolist()): # 在0.4阈值下与训练集没有相同的cluster，则代表相似度小于0.4
            return similarity_index_levels_6[1] # 0.2~0.4
        else: # 在0.4阈值下，与训练集有相同的cluster，则相似度大于0.4
            if (cluster_50 not in train_dataset['cluster_ther_0.5'].tolist()): # 在0.5阈值下与训练集没有相同的cluster，则代表相似度小于0.5
                return similarity_index_levels_6[2] # 0.4~0.5
            else: # 在0.5阈值下，与训练集有相同的cluster，则相似度大于0.5
                if (cluster_60 not in train_dataset['cluster_ther_0.6'].tolist()): # 在0.6阈值下与训练集没有相同的cluster，则代表相似度小于0.6
                    return similarity_index_levels_6[3] # 0.5~0.6
                else: # 在0.6阈值下，与训练集有相同的cluster，则相似度大于0.6
                    if (cluster_70 not in train_dataset['cluster_ther_0.7'].tolist()): # 在0.7阈值下与训练集没有相同的cluster，则代表相似度小于0.7
                        return similarity_index_levels_6[4] # 0.6~0.7
                    else:
                        return similarity_index_levels_6[5] # 0.7~0.8



# %%
test_dataset['similarity_index_level'] = test_dataset.apply(lambda row: get_similarity_index_level(row['cluster_ther_0.8'], row['cluster_ther_0.6'], row['cluster_ther_0.4'], train_dataset), axis=1)

# %%
test_dataset['similarity_index_level_6'] = test_dataset.apply(lambda row: get_similarity_index_level_6(row['cluster_ther_0.8'], row['cluster_ther_0.7'], row['cluster_ther_0.6'], row['cluster_ther_0.5'], row['cluster_ther_0.4'], row['cluster_ther_0.3'], train_dataset), axis=1)

# %%
import matplotlib.pyplot as plt


thresholds = similarity_index_levels
sample_counts = [(test_dataset['similarity_index_level'] == x).sum() for x in thresholds]



colors = ['#3498db', '#2ecc71', '#e74c3c']  


plt.figure(figsize=(10, 6)) 
plt.bar(thresholds, sample_counts, color=colors)
ax = plt.gca()  
ax.spines['top'].set_visible(False)  
ax.spines['right'].set_visible(False) 

plt.title('Sample Counts by Similarity Index Level')
plt.xlabel('Similarity Index Level')
plt.ylabel('Number of Test Dataset Samples')


for i, count in enumerate(sample_counts):
    plt.text(i, count + 5, str(count), ha='center', va='bottom')


plt.show()


# %%
import matplotlib.pyplot as plt


thresholds = similarity_index_levels_6
sample_counts = [(test_dataset['similarity_index_level_6'] == x).sum() for x in thresholds]



colors = ['#3498db', '#2ecc71', '#e74c3c']  


plt.figure(figsize=(10, 6)) 
plt.bar(thresholds, sample_counts, color=colors)
ax = plt.gca()  
ax.spines['top'].set_visible(False)  
ax.spines['right'].set_visible(False) 

plt.title('Sample Counts by Similarity Index Level')
plt.xlabel('Similarity Index Level')
plt.ylabel('Number of Test Dataset Samples')


for i, count in enumerate(sample_counts):
    plt.text(i, count + 5, str(count), ha='center', va='bottom')


plt.show()


# %%
test_dataset.to_csv(os.path.join(ec_site_dataset_path, merge_dataset_name_str+f'{max_sample}', 'test_dataset_with_similarity_idx.csv'), index=False)

# %%
cluster_all_levels2alphafold_id[0.8]

# %%
import subprocess
tmscore_path='/home/xiaoruiwang/software/TMalign/TMscore'  # 替换为自己下载的TMscore路径
def calculate_tmscore(pdb_file1, pdb_file2, tmscore_path='TMscore'):
    """
    Calculate the TMscore for two PDB files using the TMscore program.
    
    Args:
    pdb_file1 (str): Path to the first PDB file.
    pdb_file2 (str): Path to the second PDB file.    默认的标准化基准
    tmscore_path (str): Path to the TMscore executable.
    
    Returns:
    float: The TMscore of the two PDB structures.
    """
    # 构建命令行命令
    command = [tmscore_path, pdb_file1, pdb_file2]
    
    # 调用 subprocess.run 来执行 TMscore
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    
    # 解析输出，查找TM-score
    tmscore = None
    for line in result.stdout.split('\n'):
        if line.strip().startswith('TM-score'):
            # print(line.strip())
            parts = line.split('=')
            if len(parts) > 1:
                tmscore = float(parts[1].split()[0])
                break
    
    return tmscore

# 调用函数
# pdb1 = "/home/xiaoruiwang/software/TMalign/PDB2.pdb"
# pdb2 = "/home/xiaoruiwang/software/TMalign/PDB1.pdb"
# tmscore = calculate_tmscore(tmscore_path='/home/xiaoruiwang/software/TMalign/TMscore',pdb_file1=pdb1, pdb_file2=pdb2)
# print("Calculated TMscore:", tmscore)


# %%
def get_structure_TMScore(row, train_dataset, pdb_path='../dataset/ec_site_dataset/structures/alphafolddb_download'):
    alphafold_id = row['alphafolddb-id'].replace(';', '')
    most_similar_train_id = []
    for cluster_ther in ['cluster_ther_0.8', 'cluster_ther_0.6', 'cluster_ther_0.4']:
        most_similar_train_id.extend(train_dataset.loc[train_dataset[cluster_ther] == row[cluster_ther]]['alphafolddb-id'].tolist())
        if len(most_similar_train_id) != 0:
            break # 只取序列最相似的去比较结构相似的
    train_pdb_names = [os.path.abspath(os.path.join(pdb_path, 'AF-{}-F1-model_v4.pdb'.format(alphafold_id.replace(';', '')))) for alphafold_id in most_similar_train_id]
    test_pdb_name = os.path.abspath(os.path.join(pdb_path, f'AF-{alphafold_id}-F1-model_v4.pdb'))
    max_tmscore = 0
    for train_pdb_name in train_pdb_names:
        tmscore = calculate_tmscore(train_pdb_name, test_pdb_name, tmscore_path=tmscore_path)
        if max_tmscore < tmscore:
            max_tmscore = tmscore
    # if max_tmscore > 0.5: print(f'TM-score >0.5 in {alphafold_id}')
    return max_tmscore

# %%
tqdm.pandas()
test_dataset['max_tmscore'] = test_dataset.progress_apply(lambda row:get_structure_TMScore(row, train_dataset), axis=1)

# %%
(test_dataset['max_tmscore'] > 0.5).sum()

# %%
(test_dataset['max_tmscore'] < 0.17).sum()

# %%
bins = [0, 0.17, 0.5, 1]
labels = ['0~0.17', '0.17~0.5', '0.5~1']
test_dataset['range'] = pd.cut(test_dataset['max_tmscore'], bins=bins, labels=labels, right=False)
count_series = test_dataset['range'].value_counts().reindex(labels)

# 绘制柱状图
plt.figure(figsize=(10, 6)) 
ax = count_series.plot(kind='bar', color=['#3498db', '#e74c3c', '#f39c12'])
ax.spines['top'].set_visible(False)  
ax.spines['right'].set_visible(False) 
for i, count in enumerate(count_series):
    plt.text(i, count + 5, str(count), ha='center', va='bottom')
plt.title('Sample Distributions by Max TM-Score in Relation to the Training Dataset')
plt.xlabel('Range of max TMscore')
plt.ylabel('Number of Test Dataset Samples')
plt.xticks(rotation=0)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
test_dataset.to_csv(os.path.join(ec_site_dataset_path, merge_dataset_name_str+f'{max_sample}', 'test_dataset_with_similarity_idx.csv'), index=False)
# 注释防止文件被意外覆盖


