import os
import pandas as pd


def check_ec_in(x, ec_list):
    one_ec_ls = [ec.strip() for ec in x.split(';')]
    for ec in one_ec_ls:
        if ec in ec_list:
            return True
    return False

if __name__ == '__main__':
    ecreact_dataset = pd.read_csv('../dataset/raw_dataset/ec_datasets/ecreact-1.0.csv')
    ec_numbers = list(set(ecreact_dataset['ec'].tolist()))
    
    ec_numbers.sort()
    
    with open('../dataset/raw_dataset/ec_datasets/ec_numbers.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(ec_numbers))

    uniprot_ec_pdb_alphafolddb_data = pd.read_csv('../dataset/raw_dataset/ec_datasets/uniprot_raw/uniprot-download_sequence_site.tsv', sep='\t')
    
    # uniprot_ec_pdb_alphafolddb_data = uniprot_ec_pdb_alphafolddb_data.loc[uniprot_ec_pdb_alphafolddb_data['Organism'] == 'Homo sapiens (Human)'].reset_index(drop=True)
    uniprot_ec_pdb_alphafolddb_data = uniprot_ec_pdb_alphafolddb_data.loc[~pd.isna(uniprot_ec_pdb_alphafolddb_data['EC number'])].reset_index(drop=True)

    
    uniprot_ec_pdb_alphafolddb_data['in_ec_react'] = uniprot_ec_pdb_alphafolddb_data['EC number'].apply(lambda x:check_ec_in(x, ec_numbers))
    uniprot_ec_pdb_alphafolddb_data = uniprot_ec_pdb_alphafolddb_data.loc[uniprot_ec_pdb_alphafolddb_data['in_ec_react']].reset_index(drop=True)
    
    print('ec data #', len(uniprot_ec_pdb_alphafolddb_data))
    
    uniprot_ec_pdb_alphafolddb_data.to_csv('../dataset/raw_dataset/ec_datasets/uniprot_raw/uniprot-download_sequence_site_ec.tsv', sep='\t', index=False)
    

    
    uniprot_ec_pdb_alphafolddb_data_nositelabel = uniprot_ec_pdb_alphafolddb_data.loc[(pd.isna(uniprot_ec_pdb_alphafolddb_data['Site']) & pd.isna(uniprot_ec_pdb_alphafolddb_data['Active site']) & pd.isna(uniprot_ec_pdb_alphafolddb_data['Binding site']))].reset_index(drop=True)
    
    uniprot_ec_pdb_alphafolddb_data_nositelabel.to_csv('../dataset/raw_dataset/ec_datasets/uniprot_raw/uniprot-download_sequence_site_ec_nositelabel.tsv', sep='\t', index=False)
    
    print('ec none site label data #', len(uniprot_ec_pdb_alphafolddb_data_nositelabel))
    
    uniprot_ec_pdb_alphafolddb_data_sitelabel = uniprot_ec_pdb_alphafolddb_data.loc[~(pd.isna(uniprot_ec_pdb_alphafolddb_data['Site']) & pd.isna(uniprot_ec_pdb_alphafolddb_data['Active site']) & pd.isna(uniprot_ec_pdb_alphafolddb_data['Binding site']))].reset_index(drop=True)
    
    print('ec site label data #', len(uniprot_ec_pdb_alphafolddb_data_sitelabel))
    
    uniprot_ec_pdb_alphafolddb_data_sitelabel.to_csv('../dataset/raw_dataset/ec_datasets/uniprot_raw/uniprot-download_sequence_site_ec_sitelabel.tsv', sep='\t', index=False)
    
    # uniprot_entrys = list(set(uniprot_ec_pdb_alphafolddb_data['Entry'].tolist()))
    # uniprot_entrys.sort()
    
    # with open('../dataset/raw_dataset/ec_datasets/uniprot_raw/uniprot_entrys.txt', 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(uniprot_entrys))
    
    # all_alphafolddb_ids = []
    # for db_ids in uniprot_ec_pdb_alphafolddb_data['AlphaFoldDB'].tolist():
    #     if pd.isna(db_ids) :continue
    #     all_alphafolddb_ids.extend([x.strip() for x in db_ids.split(';')])
    # all_alphafolddb_ids = [f'https://alphafold.ebi.ac.uk/files/AF-{x}-F1-model_v4.pdb' for x in all_alphafolddb_ids if x]
    # all_alphafolddb_ids.sort()
    # print('AF PDB #', len(all_alphafolddb_ids))
    # with open('../dataset/raw_dataset/ec_datasets/uniprot_raw/all_alphafolddb_number.txt', 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(all_alphafolddb_ids))
        
    # all_pdb_ids = []
    # for pdb_ids in uniprot_ec_pdb_alphafolddb_data['PDB'].tolist():
    #     if pd.isna(pdb_ids) :continue
    #     all_pdb_ids.extend([x.strip() for x in pdb_ids.split(';') if isinstance(x, str)])
    # all_pdb_ids = ['https://files.rcsb.org/download/{}.pdb'.format(x.upper()) for x in all_pdb_ids if x]
    # all_pdb_ids.sort()
    # print('PDB #', len(all_pdb_ids))
    # with open('../dataset/raw_dataset/ec_datasets/uniprot_raw/all_pdb_ids.txt', 'w', encoding='utf-8') as f:
    #     f.write('\n'.join(all_pdb_ids))