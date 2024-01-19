import gdown
import os
import subprocess
import pandas as pd


url_template = 'https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb'

def cmd(command):
    subp = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8")
    subp.wait()
    if subp.poll() == 0:
        print(subp.communicate()[1])
    else:
        print(f"{command} Failure!")

dirname_path = os.path.abspath(os.path.dirname(__file__))
work_dir = os.path.abspath(dirname_path)
file_list = [
    ('https://drive.google.com/uc?id=1E2zhpsw3GN-Tr-M__JKhs7kqbwvLdmQk', os.path.join(work_dir, 'checkpoints.zip')),
    ('https://drive.google.com/uc?id=15c-KoZ47TpF9_qyQfJiY67gcgVZ8N5WR', os.path.join(work_dir, 'dataset.zip')),
]


def get_alphafolddb_pdb_urls(dataset_path):
    
    all_pdb_ids = set()
    for dataset_flag in ['train', 'valid', 'test']:
        this_path = os.path.join(dataset_path, f'{dataset_flag}_dataset')
        files = os.listdir(this_path)
        for file in files:
            df = pd.read_csv(os.path.join(this_path, file))
            all_pdb_ids.update(set(df['alphafolddb-id'].tolist()))
    
    return [url_template.format(x.strip()) for x in list(all_pdb_ids)]

def download_structures(dataset_path):
    
    print(f'Download structure files for {dataset_path}')
    
    urls = get_alphafolddb_pdb_urls(dataset_path)
    structure_download_path = os.path.abspath(os.path.join(os.path.dirname(dataset_path), 'structures', 'alphafolddb_download'))
    os.makedirs(structure_download_path, exist_ok=True)
    urls_file = os.path.abspath(os.path.join(os.path.dirname(dataset_path), 'structures','urls.txt'))
    with open(urls_file, 'w') as f:
        f.write('\n'.join(urls))
        
    cmd_str = f'aria2c -i {urls_file} -x 16 -d {structure_download_path}'
    cmd(cmd_str)

print('Downloading files...')

for url, save_path in file_list:
    if not os.path.exists(save_path.replace('.zip', '')):
        if not os.path.exists(save_path):
            gdown.download(url, save_path, quiet=False)
            assert os.path.exists(save_path)
            
        else:
            print(f"{save_path} exists, skip downloading")
            
        if save_path.endswith('.zip'):
            cmd('unzip -o {} -d {}'.format(save_path, os.path.dirname(save_path)))
    else:
        print(f"{save_path.replace('.zip', '')} exists, skip downloading")


download_structures(os.path.join(work_dir, 'dataset', 'ec_site_dataset', 'uniprot_ecreact_cluster_split_merge_dataset_limit_100'))
download_structures(os.path.join(work_dir, 'dataset', 'mcsa_fine_tune', 'normal_mcsa'))
 
