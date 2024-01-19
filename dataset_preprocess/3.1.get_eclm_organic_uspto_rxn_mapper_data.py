import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from pandarallel import pandarallel

def canonicalize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return ''

def merge_react_and_reagent(rxn_smiles):
    react, reagent, product = rxn_smiles.split('>')
    if reagent:
        precursor = canonicalize_smiles(f'{react}.{reagent}')
    else:
        precursor = canonicalize_smiles(react)
    product = canonicalize_smiles(product)
    if '' in [precursor, product]:
        return ''
    return f'{precursor}>>{product}'

if __name__ == '__main__':
    pandarallel.initialize(nb_workers=12, progress_bar=True)
    rxnmapper_dataset_path = '../dataset/raw_dataset/uspto_rxnmapper/Training'
    ec_site_root = '../dataset/organic_rxn_dataset/'
    
    rxnmapper_uspto_train_set = pd.read_csv(os.path.join(rxnmapper_dataset_path, 'uspto_all_reactions_training.txt'))
    rxnmapper_uspto_train_set.columns = ['rxn_smiles']
    rxnmapper_uspto_train_set = rxnmapper_uspto_train_set.drop_duplicates(subset=['rxn_smiles']).reset_index(drop=True)
    
    test_set = rxnmapper_uspto_train_set.sample(n=40000, random_state=42)
    rxnmapper_uspto_train_set = rxnmapper_uspto_train_set.drop(test_set.index)
    val_set = rxnmapper_uspto_train_set.sample(n=40000, random_state=42)
    train_set = rxnmapper_uspto_train_set.drop(val_set.index)
    
    test_set = test_set.reset_index(drop=True)
    val_set = val_set.reset_index(drop=True)
    train_set = train_set.reset_index(drop=True)
    
    dataset_dict = {
        'train': train_set,
        'valid': val_set,
        'test': test_set
    }
    
    for data_flag in ['train', 'valid', 'test']:
        
        save_df = dataset_dict[data_flag]
        save_df['ec'] = ['organic'] * save_df.shape[0]
        save_df['source'] = ['USPTO'] * save_df.shape[0]
        save_df.columns = ['rxn_smiles', 'ec', 'source']
        save_df['rxn_smiles'] = save_df['rxn_smiles'].parallel_apply(lambda x:merge_react_and_reagent(x))
        save_df = save_df.loc[save_df['rxn_smiles']!=''].reset_index(drop=True)
        save_path_ec_site = os.path.join(ec_site_root, 'uspto_rxnmapper', data_flag)
        print(data_flag, len(save_df))
        if not os.path.exists(save_path_ec_site):
            os.makedirs(save_path_ec_site)
        save_df.to_csv(os.path.join(save_path_ec_site, 'rxn-smiles.csv'), index=False)

        