# %%
import os
import pandas as pd
from rxnmapper import RXNMapper
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG, display
from rdchiral.template_extractor import extract_from_reaction
from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants
from tqdm import tqdm
tqdm.pandas()
import torch


rxn_mapper = RXNMapper()


# %%
def get_mcsa_normal_dataset(dir, subset=['train', 'valid'], flag='mcsa', read_new=False):
    dataset = pd.DataFrame()
    for dataset_flag in subset:
        sub_df = pd.read_csv(os.path.join(dir, f'{dataset_flag}_dataset' if not read_new else f'new_{dataset_flag}_dataset', f'{flag}_{dataset_flag}.csv'))
        sub_df['dataset_flag'] = [dataset_flag for _ in range(len(sub_df))]
        dataset = pd.concat([dataset, sub_df])
    dataset = dataset.reset_index(drop=True)
    return dataset

def get_reaction_smiles(reaction):
    # reaction 格式为：底物SMILES|酶序列>>产物SMILES
    precursors, products = reaction.split('>>')
    substrates, aa_sequence = precursors.split('|')
    return f'{substrates}>>{products}'


def mapping_reaction(rxn_smiles):
    try:
        results = rxn_mapper.get_attention_guided_atom_maps([rxn_smiles])[0]
        mapped_rxn = results['mapped_rxn']
    except Exception as e:
        print(e)
        print(f'Erro in {rxn_smiles}')
        mapped_rxn = ''
    return mapped_rxn
    
def canonicalize_smiles(smiles, remove_map=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        if remove_map:
            [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms() if a.HasProp('molAtomMapNumber')]
        return Chem.MolToSmiles(mol)
    else:
        return ''
    
def get_fixed_prec(rxn_smiles):
    
    reactants, products = rxn_smiles.split('>>')
    reactants_list = reactants.split('.')
    products_list = products.split('.')
    
    fixed_reactants_list = []
    for react in reactants_list:
        if (react not in products_list) and (':' in react):
            fixed_reactants_list.append(react)
    
    return canonicalize_smiles('.'.join(fixed_reactants_list))

    
def get_templates(rxn_smi,  
                  add_brackets=True):
    """
    Adapt from https://github.com/hesther/templatecorr/blob/main/templatecorr/extract_templates.py#L22

    Extracts a template at a specified level of specificity for a reaction smiles.

    :param rxn_smi: Reaction smiles string
    :param no_special_groups: Boolean whether to omit special groups in template extraction
    :param radius: Integer at which radius to extract templates
    :param add_brackets: Whether to add brackets to make template pseudo-unimolecular

    :return: Template
    """    
    #Extract:
    try:
        rxn_split = rxn_smi.split(">")
        reaction={"_id":0,"reactants":rxn_split[0],"spectator":rxn_split[1],"products":rxn_split[2]}
        template = extract_from_reaction(reaction)["reaction_smarts"]
        if add_brackets:
            template = "(" + template.replace(">>", ")>>")
    except:
        template = ''  
    #Validate:
    if template != '':
        
        prec = get_fixed_prec(rxn_smi)
        
        rct = rdchiralReactants(rxn_smi.split(">")[-1])
        try:
            rxn = rdchiralReaction(template)
            outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False)
        except:
            outcomes =[]
        if not prec in outcomes:

            print('template:')
            print(template)
            print('rxn_smiles:')
            print(rxn_smi+'\n')
            template=''
            
    return template
    
def assign_mapper(rxn_smiles, mapper_list):
    reactants, products = rxn_smiles.split('>>')
    mol_reactants, mol_products = Chem.MolFromSmiles(reactants), Chem.MolFromSmiles(products)
    for prod_idx in range(mol_products.GetNumAtoms()):
        mol_products.GetAtomWithIdx(prod_idx).SetProp('molAtomMapNumber', str(prod_idx+1))
        react_idx = mapper_list[0][prod_idx]
        mol_reactants.GetAtomWithIdx(react_idx).SetProp('molAtomMapNumber', str(prod_idx+1))
    mapped_reactants = Chem.MolToSmiles(mol_reactants)
    mapped_products  = Chem.MolToSmiles(mol_products)
    return f'{mapped_reactants}>>{mapped_products}'


# https://gist.github.com/greglandrum/61c1e751b453c623838759609dc41ef1
def draw_chemical_reaction(smiles, highlightByReactant=False, font_scale=1.5):
    rxn = rdChemReactions.ReactionFromSmarts(smiles,useSmiles=True)
    trxn = rdChemReactions.ChemicalReaction(rxn)
    # move atom maps to be annotations:
    for m in trxn.GetReactants():
        moveAtomMapsToNotes(m)
    for m in trxn.GetProducts():
        moveAtomMapsToNotes(m)
    d2d = rdMolDraw2D.MolDraw2DSVG(800,300)
    d2d.drawOptions().annotationFontScale=font_scale
    d2d.DrawReaction(trxn,highlightByReactant=highlightByReactant)

    d2d.FinishDrawing()

    return d2d.GetDrawingText()

def moveAtomMapsToNotes(m):
    for at in m.GetAtoms():
        if at.GetAtomMapNum():
            at.SetProp("atomNote",str(at.GetAtomMapNum()))
    

# %%
mcsa_dataset_df = get_mcsa_normal_dataset('../dataset/mcsa_fine_tune/normal_mcsa', subset=['train', 'valid', 'test'])
mcsa_dataset_df['rxn_smiles'] = mcsa_dataset_df['reaction'].apply(lambda x: get_reaction_smiles(x))
reaction_length = mcsa_dataset_df['rxn_smiles'].apply(lambda x:len(x))

# %%
mcsa_dataset_df

# %%
reaction_length.describe()

# %%
(reaction_length >= 512).sum()

# %%


mcsa_dataset_df['mapped_rxn_smiles'] = mcsa_dataset_df['rxn_smiles'].progress_apply(lambda x: mapping_reaction(x))
mcsa_dataset_df = mcsa_dataset_df.loc[mcsa_dataset_df['mapped_rxn_smiles']!='']
mcsa_dataset_df

# %%
mcsa_dataset_df[['rxn_smiles', 'mapped_rxn_smiles']].loc[0]

# %%
print(mcsa_dataset_df[['alphafolddb-id']].loc[90])
for rxn in mcsa_dataset_df[['rxn_smiles', 'mapped_rxn_smiles']].loc[90].tolist():
    display(SVG(draw_chemical_reaction(rxn)))

# %%


mcsa_dataset_df['retro_templates'] = mcsa_dataset_df.progress_apply(lambda row: get_templates(row['mapped_rxn_smiles'], add_brackets=True), axis=1)

# %%
mcsa_dataset_df['retro_templates']

# %%
(mcsa_dataset_df['retro_templates'] == '').sum()


