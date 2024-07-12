# %%
# %%
import pandas as pd
import os


# %%
# %%
mcsa_row_path = '../dataset/raw_dataset/mcsa/'
dataset_path = '../dataset/mcsa_fine_tune/normal_mcsa/'
structure_path = '../dataset/mcsa_fine_tune/structures/'


# %%
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

threshold = [0.8, 0.6, 0.4]
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
    args = ["-i", os.path.abspath(os.path.join(mcsa_row_path, 'ec_react_100_train_and_mcsa_sequence.fasta')),
            "-o", os.path.abspath(os.path.join(mcsa_row_path, f'ec_react_100_train_and_mcsa_sequence_cutoff_{str(thre)}.fasta')), 
            "-c", str(thre),
            "-n", str(word),
            "-T", str(12)]
    print('Args: {}'.format(' '.join(args)))
    if not os.path.exists(os.path.abspath(os.path.join(mcsa_row_path, f'ec_react_100_train_and_mcsa_sequence_cutoff_{str(thre)}.fasta'))):
        run_subprocess_and_print([command] + args)


# %%
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
# %%
threshold = threshold
alphafold_id2cluster_all_levels = {}
cluster_all_levels2alphafold_id = {}
for thre in threshold:
    cluster_results_path = os.path.abspath(os.path.join(mcsa_row_path, f'ec_react_100_train_and_mcsa_sequence_cutoff_{str(thre)}.fasta.clstr'))
    cluster_results = get_cluster(cluster_results_path)
    print(len(cluster_results))
    alphafold_id2cluster_all_levels[thre] = id2cluster(cluster_results)
    cluster_all_levels2alphafold_id[thre] = cluster_results


# %%
# %%
from tqdm import tqdm
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
import pandas as pd
from Bio import SeqIO

def fasta_to_dataframe(fasta_file):
    # 创建空列表用于存储序列数据
    ids = []
    sequences = []
    
    # 使用BioPython的SeqIO解析fasta文件
    for record in SeqIO.parse(fasta_file, "fasta"):
        # 将序列ID和序列数据分别加入到列表中
        ids.append(record.id)
        sequences.append(str(record.seq))
    
    # 使用pandas创建DataFrame
    df = pd.DataFrame({
        'alphafolddb-id': ids,
        'aa_sequence': sequences
    })
    
    return df

# %%

ec_react_100_train_and_mcsa_sequence_df = fasta_to_dataframe(os.path.join(mcsa_row_path, 'ec_react_100_train_and_mcsa_sequence.fasta'))
ec_react_100_train_and_mcsa_sequence_df

# %%
train_dataset = get_query_database(os.path.join(dataset_path, 'train_dataset', 'mcsa_train.csv'), fasta_path=os.path.join(dataset_path, 'train_dataset_check.fasta'), pdb_file_path=os.path.join(structure_path,'alphafolddb_download'))
train_dataset

# %%
# %%

test_dataset = get_query_database(os.path.join(dataset_path, 'test_dataset', 'mcsa_test.csv'), fasta_path=os.path.join(dataset_path, 'test_dataset_check.fasta'), pdb_file_path=os.path.join(structure_path,'alphafolddb_download'))
test_dataset = test_dataset.loc[test_dataset['is_valid']].reset_index(drop=True)
test_dataset




# %%
def id_handle(id, name_dict):
    if id not in name_dict:
        return name_dict[f' {id}']
    return name_dict[id]


for thre in threshold:
    # ec_react_100_train_and_mcsa_sequence_df[f'cluster_ther_{str(thre)}'] = ec_react_100_train_and_mcsa_sequence_df['alphafolddb-id'].apply(lambda x:alphafold_id2cluster_all_levels[thre][x.replace(';', '')])
    train_dataset[f'cluster_ther_{str(thre)}'] = train_dataset['alphafolddb-id'].apply(lambda x:id_handle(x.replace(';', ''), alphafold_id2cluster_all_levels[thre]))
    # test_dataset[f'cluster_ther_{str(thre)}'] = test_dataset['alphafolddb-id'].apply(lambda x:alphafold_id2cluster_all_levels[thre][x.replace(';', '')])
    test_dataset[f'cluster_ther_{str(thre)}'] = test_dataset['alphafolddb-id'].apply(lambda x:id_handle(x.replace(';', ''), alphafold_id2cluster_all_levels[thre]))

# %%
train_dataset

# %%
test_dataset

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
test_dataset['similarity_index_level'] = test_dataset.apply(lambda row: get_similarity_index_level(row['cluster_ther_0.8'], row['cluster_ther_0.6'], row['cluster_ther_0.4'], train_dataset), axis=1)

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
test_dataset.to_csv(os.path.join(dataset_path, 'test_dataset_with_similarity_idx.csv'), index=False)


