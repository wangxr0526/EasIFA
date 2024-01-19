import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from pandarallel import pandarallel

def canonicalize_smies(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return ''

def canonicalize_rxn(rxn_smiles):
    precursor, product = rxn_smiles.split('>>')
    precursor = canonicalize_smies(precursor)
    product = canonicalize_smies(product)
    if '' in [precursor, product]:
        return ''
    return f'{precursor}>>{product}'

if __name__ == '__main__':
    pandarallel.initialize(nb_workers=12, progress_bar=True)
    ec_site_root = '../dataset/organic_rxn_dataset/'
    raw_pistachio_path = '../dataset/raw_dataset/pistachio'
    pistachio_dataset_df = pd.read_csv(os.path.join(raw_pistachio_path, 'canonical_pistachio.csv'))
    pistachio_dataset_df = pistachio_dataset_df.drop_duplicates(subset=['rxn', 'rxn_class', 'rxn_class_name', 'rxn_category', 'rxn_superclass', 'label'])
    pistachio_dataset_df = pistachio_dataset_df.sample(frac=1).reset_index()
    
    pistachio_dataset_df['rxn'] = pistachio_dataset_df['rxn'].parallel_apply(lambda x:canonicalize_rxn(x))
    pistachio_dataset_df = pistachio_dataset_df.loc[pistachio_dataset_df['rxn']!=''].reset_index(drop=True)
    
    train_split_point = 0.9 * len(pistachio_dataset_df)
    valid_split_point = 0.95 * len(pistachio_dataset_df)
    
    pistachio_dataset_df.loc[:train_split_point, 'dataset'] = 'train'
    pistachio_dataset_df.loc[train_split_point:valid_split_point, 'dataset'] = 'valid'
    pistachio_dataset_df.loc[valid_split_point:, 'dataset'] = 'test'
    
    
    
    for data_flag in ['train', 'valid', 'test']:
        save_df = pistachio_dataset_df.loc[pistachio_dataset_df['dataset']==data_flag]
        save_df = save_df[['rxn', 'rxn_category', 'label']]
        
        save_df['ec'] = ['organic'] * save_df.shape[0]
        save_df['source'] = ['pistachio'] * save_df.shape[0]

        save_df.columns = ['rxn_smiles', 'rxn_category', 'label', 'ec', 'source']
        save_df = save_df[['rxn_smiles', 'ec', 'source', 'rxn_category', 'label']]
        save_path_ec_site = os.path.join(ec_site_root, 'pistachio', data_flag)
        if not os.path.exists(save_path_ec_site):
            os.makedirs(save_path_ec_site)
        save_df.to_csv(os.path.join(save_path_ec_site, 'rxn-smiles.csv'), index=False)

        