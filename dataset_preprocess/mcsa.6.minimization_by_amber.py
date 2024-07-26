# %%
import pandas as pd
import os
import subprocess
from tqdm import tqdm
import re


# %%
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


# %%
def extract_leap_info(log_text):
    # 使用正则表达式匹配错误、警告和注意的数量
    match = re.search(r"Exiting LEaP: Errors = (\d+); Warnings = (\d+); Notes = (\d+).", log_text)
    if match:
        errors, warnings, notes = map(int, match.groups())
        return errors, warnings, notes
    else:
        return None
    
def minimize_use_amber(input_pdb_file, output_path, AMBERHOME='/home/xiaoruiwang/software/amber20'):
    is_success = False
    abs_input_pdb_path = os.path.abspath(input_pdb_file)
    abs_output_path = os.path.abspath(output_path)
    pdb_name = os.path.split(abs_input_pdb_path)[-1].split('.')[0]
    minimize_command = f'./minimize_script_folder/minimize_protein.sh {abs_input_pdb_path} {abs_output_path} {AMBERHOME}'
    if not os.path.exists(os.path.join(output_path, f'{pdb_name}_minimized.pdb')):
        completed_process = subprocess.run(minimize_command, shell=True)
        if completed_process.returncode != 0:
            print("Error: The command did not run successfully!")
    
    # 抓错误

    with open(os.path.join(output_path, f'{pdb_name}_minimized_tleap.log'), 'r') as f:
        for line in f.readlines():
            line = line.strip()
            info = extract_leap_info(line)
            if info:
                errors, warnings, notes = info
                if errors == 0:
                    is_success = True

        
    return abs_output_path, is_success

def get_mcsa_normal_dataset(dir, subset=['train', 'valid'], flag='mcsa', read_new=False):
    dataset = pd.DataFrame()
    for dataset_flag in subset:
        sub_df = pd.read_csv(os.path.join(dir, f'{dataset_flag}_dataset' if not read_new else f'new_{dataset_flag}_dataset', f'{flag}_{dataset_flag}.csv'))
        sub_df['dataset_flag'] = [dataset_flag for _ in range(len(sub_df))]
        dataset = pd.concat([dataset, sub_df])
    return dataset
        
    

# %%
# aug_dir_flag ='mcsa_aug_20_mutation_rate_0.2_insertion_rate_0.1_deletion_rate_0.1_max_length_150_seed_123'
# aug_dir_flag = 'mcsa_aug_40_mutation_rate_0.3_insertion_rate_0.1_deletion_rate_0.1_max_length_150_seed_123'
aug_dir_flag = 'mcsa_aug_20+40_mutation_rate_0.2+0.35_insertion_rate_0.1_deletion_rate_0.1_max_length_150_seed_123'
new_aug_save_path = f'../dataset/mcsa_fine_tune/{aug_dir_flag}/'

fixed_generated_structure_dir = f'../dataset/mcsa_fine_tune/{aug_dir_flag}/esmfold_generated_aug_structures'
end_aug_dataset = get_mcsa_normal_dataset(new_aug_save_path, subset=['train', 'valid'], flag='aug_mcsa', read_new=True)
end_aug_dataset

# %%
end_aug_dataset['site_labels'].apply(lambda x:x=='[]').sum()/len(end_aug_dataset)

# %%
minimize_failure_list = []

for pdb_id in tqdm(end_aug_dataset['alphafolddb-id'].tolist()):
    pdb_id = pdb_id.replace(' ', '')
    abs_input_path = os.path.abspath(os.path.join(fixed_generated_structure_dir, f'{pdb_id}.pdb'))
    # print(abs_input_path)
    # break
    _, is_success = minimize_use_amber(abs_input_path, os.path.abspath(fixed_generated_structure_dir), AMBERHOME='/home/ipmgpu2022a/Software/amber20')
    if not is_success:
        minimize_failure_list.append(abs_input_path)

# %%
minimize_failure_list

# %%
end_aug_test_dataset = get_mcsa_normal_dataset(new_aug_save_path, subset=['test'], flag='aug_mcsa', read_new=True)
end_aug_test_dataset



# %%
import shutil

minimize_failure_list_test = []
org_structure_dir = '../dataset/mcsa_fine_tune/structures/alphafolddb_download'
for pdb_id in tqdm(end_aug_test_dataset['alphafolddb-id'].tolist()):
    pdb_id = pdb_id.replace(' ', '')
    org_fname = 'AF-{}-F1-model_v4.pdb'.format(pdb_id.split('_')[0].split('-')[0])
    org_pdb_path = os.path.join(org_structure_dir, org_fname)
    abs_input_path = os.path.abspath(os.path.join(fixed_generated_structure_dir, f'{pdb_id}.pdb'))
    try:
        shutil.copyfile(org_pdb_path, os.path.join(fixed_generated_structure_dir, '{}.pdb'.format(pdb_id.split('_')[0].split('-')[0])))
    except:
        minimize_failure_list_test.append(abs_input_path)
        continue
    
    # print(abs_input_path)
    # break
    _, is_success = minimize_use_amber(abs_input_path, os.path.abspath(fixed_generated_structure_dir), AMBERHOME='/home/ipmgpu2022a/Software/amber20')
    if not is_success:
        minimize_failure_list_test.append(abs_input_path)

# %%
print(len(minimize_failure_list_test))


