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
    precursor = canonicalize_smiles(f'{react}.{reagent}')
    product = canonicalize_smiles(product)
    if '' in [precursor, product]:
        return ''
    return f'{precursor}>>{product}'

if __name__ == '__main__':
    pandarallel.initialize(nb_workers=12, progress_bar=True)
    rxnaamapper_root = '../../rxnaamapper/'
    ec_site_root = '../dataset/organic_rxn_dataset/'
    
    for data_flag in ['train', 'valid', 'test']:
        data_df = pd.read_csv('../dataset/raw_dataset/uspto/US_patents_1976-Sep2016_1product_reactions_{}.csv'.format(data_flag), sep='\t')
        
        save_df = pd.DataFrame(data_df['CanonicalizedReaction'])
        save_df['ec'] = ['organic'] * save_df.shape[0]
        save_df['source'] = ['USPTO'] * save_df.shape[0]
        # save_path_rxnaamapper = os.path.join(rxnaamapper_root, 'data', 'uspto', data_flag)
        # if not os.path.exists(save_path_rxnaamapper):
        #     os.makedirs(save_path_rxnaamapper)
        # save_df.to_csv(os.path.join(save_path_rxnaamapper, 'rxn-smiles.csv'), index=False, header=None)
        save_df.columns = ['rxn_smiles', 'ec', 'source']
        save_df['rxn_smiles'] = save_df['rxn_smiles'].parallel_apply(lambda x:merge_react_and_reagent(x))
        save_df = save_df.loc[save_df['rxn_smiles']!=''].reset_index(drop=True)
        save_path_ec_site = os.path.join(ec_site_root, 'uspto', data_flag)
        if not os.path.exists(save_path_ec_site):
            os.makedirs(save_path_ec_site)
        save_df.to_csv(os.path.join(save_path_ec_site, 'rxn-smiles.csv'), index=False)

        