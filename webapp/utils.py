
import os
import sys
import subprocess
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), '..')))
from  pandarallel import pandarallel
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import torch
import re
import json
from tqdm.auto import tqdm
from collections import defaultdict, OrderedDict
from functools import partial
import py3Dmol
from torch.utils import data as torch_data
from common.utils import convert_fn, cuda, read_model_state
from torchdrug import data
from dgllife.utils import WeaveAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph
from data_loaders.enzyme_dataloader import MyProtein
from data_loaders.enzyme_rxn_dataloader import enzyme_rxn_collate_extract, atom_types, get_adm, ReactionFeatures
from rdkit import Chem
import mysql.connector
from mysql.connector import Error
from biotite.structure.io.pdb import PDBFile, get_structure
from model_structure.enzyme_site_model import EnzymeActiveSiteClsModel, EnzymeActiveSiteModel

file_cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static', 'pdb_cache'))
rxn_fig_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static', 'rxn_fig'))
os.makedirs(rxn_fig_path, exist_ok=True)
os.makedirs(file_cache_path, exist_ok=True)

default_ec_site_model_state_path = '../checkpoints/enzyme_site_type_predition_model/train_in_uniprot_ecreact_cluster_split_merge_dataset_limit_100_at_2023-06-14-11-04-55/global_step_70000'
full_swissprot_checkpoint_path = '../checkpoints/enzyme_site_type_predition_model/train_in_uniprot_ecreact_cluster_split_merge_dataset_all_limit_3_at_2023-12-19-16-06-42/global_step_284000'

mol_to_graph = partial(mol_to_bigraph, add_self_loop=True)
node_featurizer = WeaveAtomFeaturizer(atom_types=atom_types)
edge_featurizer = CanonicalBondFeaturizer(self_loop=True)


label2active_type = {
    0: None,
    1: 'Binding Site',
    2: 'Catalytic Site',    # Active Site in UniProt
    3: 'Other Site'
}


def cmd(command):
    subp = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8")
    subp.wait()
    if subp.poll() == 0:
        print(subp.communicate()[1])
    else:
        print(f"{command} Failure!")  


def svg2file(fname, svg_text):
    with open(fname, 'w') as f:
        f.write(svg_text)
        
def reaction2svg(reaction, path):
    # smi = ''.join(smi.split(' '))
    # mol = Chem.MolFromSmiles(smi)
    d = Draw.MolDraw2DSVG(1500, 500)
    # opts = d.drawOptions()
    # opts.padding = 0  # 增加边缘的填充空间
    # opts.bondLength = -5  # 如果需要，可以调整键的长度
    # opts.atomLabelFontSize = 5  # 调整原子标签字体大小
    # opts.additionalAtomLabelPadding = 0  # 增加原子标签的额外填充空间
    
    
    d.DrawReaction(reaction)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    # svg2 = svg.replace('svg:', '').replace(';fill:#FFFFFF;stroke:none', '').replace('y=\'0.0\'>', 'y=\'0.0\' fill=\'rgb(255,255,255,0)\'>')  # 使用replace将原始白色的svg背景变透明
    # svg2 = svg.replace('svg:', '').replace(';fill:#FFFFFF;stroke:none', 'rgb(255,255,255,0)')
    svg2 = svg.replace('svg:', '')
    svg2file(path, svg2)
    return '\n'.join(svg2.split('\n')[8:-1])

def white_pdb(pdb_lines):
    save_path = os.path.join(file_cache_path, 'input_pdb.pdb')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(pdb_lines))
        
def get_3dmolviewer_id(html_line1):
    matches = re.search(r'id="([^"]+)"', html_line1)
    if matches:
        return matches.group(1)
    else:
        raise ValueError()
 
 
    

        
        
def get_structure_html_and_active_data(
                   enzyme_structure_path,
                   site_labels=None, 
                   view_size=(900, 900), 
                   res_colors={
                    0: '#73B1FF',   # 非活性位点
                    1: '#FF0000',     # Binding Site
                    2: '#00B050',     # Active Site
                    3: '#FFFF00',     # Other Site
                    },
                   show_active=True
):
    with open(enzyme_structure_path) as ifile:
        system = ''.join([x for x in ifile])
    
    view = py3Dmol.view(width=view_size[0], height=view_size[1])
    view.addModelsAsFrames(system)
    
    active_data = []
    
    if show_active and (site_labels is not None):
        i = 0
        res_idx = None
        for line in system.split("\n"):
            split = line.split()
            if len(split) == 0 or split[0] != "ATOM":
                continue
            if res_idx is None:
                first_res_idx = int(line[22:26].strip())
            res_idx = int(line[22:26].strip()) - first_res_idx
            color = res_colors[site_labels[res_idx]]
            view.setStyle({'model': -1, 'serial': i+1}, {"cartoon": {'color': color}})
            atom_name = line[12:16].strip()
            if (atom_name == 'CA') and (site_labels[res_idx] !=0) :
                residue_name = line[17:20].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                view.addLabel(f'{residue_name} {res_idx + 1}', {"fontSize": 15, "position": {"x": x, "y": y, "z": z}, "fontColor": color, "fontOpacity":1.0 ,"backgroundColor": 'white', "bold":True,"backgroundOpacity": 0.2})
                active_data.append((res_idx + 1, residue_name, color, label2active_type[site_labels[res_idx]])) # 设置label从1开始#
            
            i += 1
    else:
        view.setStyle({'model': -1}, {"cartoon": {'color': res_colors[0]}})
    # view.addSurface(py3Dmol.SAS, {'opacity': 0.5})
    view.zoomTo()
    # view.show()
    view.zoom(2.5, 600)
    return view.write_html(), active_data

class UniProtParser:
    def __init__(self, chebi_path, json_folder, rxn_folder, alphafolddb_folder):
        
        
        self.json_folder = json_folder
        os.makedirs(self.json_folder, exist_ok=True)
        self.rxn_folder = rxn_folder
        os.makedirs(self.rxn_folder, exist_ok=True)
        self.alphafolddb_folder = alphafolddb_folder
        os.makedirs(self.alphafolddb_folder, exist_ok=True)
       
        
        self.chebi_df = pd.read_csv(chebi_path)
        # self.query_uniprotkb_template = "curl  -o {} -H \"Accept: text/plain; format=tsv\" \"https://rest.uniprot.org/uniprotkb/search?query=accession:{}&fields=accession,ec,sequence,cc_catalytic_activity,xref_alphafolddb,ft_binding,ft_act_site,ft_site\""
        self.query_uniprotkb_template = "/usr/bin/curl  -o {} -H \"Accept: application/json\" \"https://rest.uniprot.org/uniprotkb/search?query=accession:{}&fields=accession,ec,sequence,cc_catalytic_activity,xref_alphafolddb,ft_binding,ft_act_site,ft_site\""
        
        self.rhea_rxn_url_template = '/usr/bin/curl -o {} https://ftp.expasy.org/databases/rhea/ctfiles/rxn/{}.rxn'
        self.download_alphafolddb_url_template = '/usr/bin/curl -o {} https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v4.pdb'
        
    def _qurey_smiles_from_chebi(self, chebi_id):
        this_id_data = self.chebi_df.loc[self.chebi_df['COMPOUND_ID']==int(chebi_id)]
        this_id_data = this_id_data.loc[this_id_data['TYPE']=='SMILES']
        smiles = this_id_data['STRUCTURE'].tolist()[0]
        return smiles
    
    def _canonicalize_smiles(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        else:
            return ''
    
    def _get_reactions(self, query_data):
        reaction_comments = [comment['reaction'] for comment in query_data['comments'] if comment['commentType'] == 'CATALYTIC ACTIVITY']
        
        rxn_smiles_list = []
        ec_number_list = []
        for rxn_comment in reaction_comments:    # (ˉ▽ˉ；)... 这个chebi_id的顺序并不一定按照name的顺序来记录，需要从交叉引用中直接找到反应信息，坑死
            if 'reactionCrossReferences' in rxn_comment:              
                enzyme_lib_info = [ref['id'].split(':') for ref in rxn_comment['reactionCrossReferences'] if ref['database'] != 'ChEBI']
                for enzyme_lib_name,  enzyme_lib_id in enzyme_lib_info:
                    if enzyme_lib_name == 'RHEA':
                        rxn_fpath = os.path.join(self.rxn_folder, f'{str(int(enzyme_lib_id)+1)}.rxn')
                        if not os.path.exists(rxn_fpath):
                            cmd(self.rhea_rxn_url_template.format(os.path.abspath(rxn_fpath), str(int(enzyme_lib_id)+1)))
                        try:
                            rxn = AllChem.ReactionFromRxnFile(rxn_fpath)
                            reaction_smiles = AllChem.ReactionToSmiles(rxn)
                            reactants_smiles, products_smiles = reaction_smiles.split('>>')
                            reactants_smiles = self._canonicalize_smiles(reactants_smiles)
                            products_smiles = self._canonicalize_smiles(products_smiles)
                            if '' not in [reactants_smiles, products_smiles]:
                                rxn_smiles = f'{reactants_smiles}>>{products_smiles}'
                                rxn_smiles_list.append(rxn_smiles)
                                if 'ecNumber'  in rxn_comment:
                                    ec_number_list.append(rxn_comment['ecNumber'])
                                else:
                                    ec_number_list.append('UNK')
                        except:
                            continue
        
        return [x for x in zip(ec_number_list, rxn_smiles_list)]
    
    def parse_from_uniprotkb_query(self, uniprot_id):
        
        uniprot_data_fpath = os.path.join(self.json_folder, f'{uniprot_id}.json')
        query_uniprotkb_cmd = self.query_uniprotkb_template.format(os.path.abspath(uniprot_data_fpath), uniprot_id)
        if not os.path.exists(uniprot_data_fpath):
            cmd(query_uniprotkb_cmd)
        with open(uniprot_data_fpath, 'r') as f:
            query_data = json.load(f)['results'][0]
        # query_data = pd.read_csv(f'test/{query_id}.tsv', sep='\t')
        
        try:
            ecNumbers = query_data['proteinDescription']['recommendedName']['ecNumbers']
        except:
            ecNumbers = []
            # return None, 'Not Enzyme'
        try: 
            alphafolddb_id = query_data['uniProtKBCrossReferences'][0]['id']
        except:
            return None, 'No Alphafolddb Structure'
        aa_length = query_data['sequence']['length']
        pdb_fpath = os.path.join(self.alphafolddb_folder, f'AF-{alphafolddb_id}-F1-model_v4.pdb')
        if not os.path.exists(pdb_fpath):
            cmd(self.download_alphafolddb_url_template.format(os.path.abspath(pdb_fpath), alphafolddb_id))
        
        ec2rxn_smiles = self._get_reactions(query_data)
        
        if (len(ecNumbers) == 0) and (len(ec2rxn_smiles) == 0):
            return None, 'Not Enzyme'
        elif (len(ec2rxn_smiles) == 0) and (len(ecNumbers) != 0):
            return None, 'No recorded reaction catalyzed found'


        
        df = pd.DataFrame(ec2rxn_smiles, columns=['ec', 'rxn_smiles'])
        df['pdb_fpath'] = [pdb_fpath for _ in range(len(df))]
        df['aa_length'] = [aa_length for _ in range(len(df))]
        
        return df, 'Good'



class EasIFAInferenceAPI:
    def __init__(self, device='cpu', model_checkpoint_path=default_ec_site_model_state_path, max_enzyme_aa_length=600) -> None:
        self.max_enzyme_aa_length = max_enzyme_aa_length
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if model_checkpoint_path in [default_ec_site_model_state_path, full_swissprot_checkpoint_path]:
            self.convert_fn = lambda x: convert_fn(x)
        else:
            self.convert_fn = lambda x: x.tolist()
        model = EnzymeActiveSiteClsModel(
            rxn_model_path='../checkpoints/reaction_attn_net/model-ReactionMGMTurnNet_train_in_uspto_at_2023-04-05-23-46-25', 
            num_active_site_type=4, 
            from_scratch=True)
        
        model_state, _ = read_model_state(model_save_path=model_checkpoint_path)
        model.load_state_dict(model_state)
        print('Loaded checkpoint from {}'.format(model_checkpoint_path))
        model.to(self.device)
        model.eval()
        self.model = model
        

    
    def _calculate_features(self, smi):
        mol = Chem.MolFromSmiles(smi)
        fgraph = mol_to_graph(mol,
                                    node_featurizer=node_featurizer,
                                    edge_featurizer=edge_featurizer,
                                    canonical_atom_order=False)
        dgraph = get_adm(mol)

        return fgraph, dgraph
    
    def _calculate_rxn_features(self, rxn):
        try:
            react, prod = rxn.split('>>')

            react_features_tuple = self._calculate_features(react)
            prod_features_tuple = self._calculate_features(prod)

            return react_features_tuple, prod_features_tuple
        except:
            return None
    
    def _preprocess_one(self, rxn_smiles, enzyme_structure_path):
    
        protein = MyProtein.from_pdb(enzyme_structure_path)
        # protein = data.Protein.from_pdb(enzyme_structure_path)
        reaction_features = self._calculate_rxn_features(rxn_smiles)
        rxn_fclass = ReactionFeatures(reaction_features)
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {
            'protein_graph': protein, 
            'reaction_graph': rxn_fclass, 
            'protein_sequence':protein.to_sequence()}
        return item
    
    def _calculate_one_data(self, rxn, enzyme_structure_path):
        data_package = self._preprocess_one(rxn_smiles=rxn, enzyme_structure_path=enzyme_structure_path)
        self.caculated_sequence = data_package['protein_sequence']
        if len(self.caculated_sequence) > self.max_enzyme_aa_length:
            return None
        batch_one_data = enzyme_rxn_collate_extract([data_package])
        return batch_one_data
    
        
    @torch.no_grad()
    def inference(self, rxn, enzyme_structure_path):
        batch_one_data = self._calculate_one_data(rxn,enzyme_structure_path)
        if batch_one_data is None:
            return 
        
        if self.device.type == "cuda":
            batch_one_data = cuda(batch_one_data, device=self.device)
        try:
            protein_node_logic, _ = self.model(batch_one_data)
        except:
            print(f'erro in this data')
            return 
        pred = torch.argmax(protein_node_logic.softmax(-1), dim=-1)
        pred = self.convert_fn(pred)
        return pred
    
    
class ECSiteBinInferenceAPI(EasIFAInferenceAPI):
    def __init__(self, device='cpu', model_checkpoint_path=default_ec_site_model_state_path) -> None:
        model_state, model_args = read_model_state(model_save_path=model_checkpoint_path)
        need_convert = model_args.get('need_convert', False)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if (model_checkpoint_path == default_ec_site_model_state_path) or need_convert:
            self.convert_fn = lambda x: convert_fn(x)
        else:
            self.convert_fn = lambda x: x.tolist()
        model = EnzymeActiveSiteModel(
            rxn_model_path='../checkpoints/reaction_attn_net/model-ReactionMGMTurnNet_train_in_uspto_at_2023-04-05-23-46-25')
        

        model.load_state_dict(model_state)
        print('Loaded checkpoint from {}'.format(model_checkpoint_path))
        model.to(self.device)
        model.eval()
        self.model = model
        

class UniProtParserMysql:
    def __init__(self, mysql_config_path) -> None:
        self.mysql_config_path = mysql_config_path
        
        self.unprot_parser = UniProtParser(chebi_path='../dataset/raw_dataset/chebi/structures.csv.gz', json_folder='test/uniprot_json', rxn_folder='test/rxn_folder', alphafolddb_folder='test/alphafolddb_folder')
        
        self.query_data_template = "SELECT * FROM qurey_data WHERE uniprot_id = %s"
        self.query_results_template = "SELECT * FROM predicted_results WHERE uniprot_id = %s"
        self.insert_data_template = """
        INSERT INTO qurey_data (uniprot_id, qurey_dataframe, message, calculated_sequence)
        VALUES (%s, %s, %s, %s)
        """
        self.insert_results_template = """
        INSERT INTO predicted_results (uniprot_id, pred_active_site_labels)
        VALUES (%s, %s)
        """
        
        self._connect_to_mysql(mysql_config_path)
        
    def _connect_to_mysql(self, mysql_config_path):
        try:
            with open(mysql_config_path, 'r') as f:
                self.mysql_config = json.load(f)
            self.mysql_conn = mysql.connector.connect(**self.mysql_config)
            self.use_mysql = True
        except Exception as e:
            self.use_mysql = False
            self.mysql_conn = None
            print(e)
            print('Warning: MySQL connect fail!')
    
    def _reconnect_mysql(self):
        try:
            self.mysql_conn = mysql.connector.connect(**self.mysql_config)
            self.use_mysql = True
        except Exception as e:
            self.use_mysql = False
            print(e)
            print('Warning: MySQL reconnect fail!')
    
    def _check_and_reconnect_mysql(self):
        
        # if not hasattr(self, 'mysql_conn'):
        #     self._connect_to_mysql(self.mysql_config_path)
        if self.mysql_conn:
            if not self.mysql_conn.is_connected():
                self._reconnect_mysql()
        else:
            self._connect_to_mysql(self.mysql_config_path)

                
    def _insert_data(self, data, predicted_results):
        try:
            cursor = self.mysql_conn.cursor()
            (uniprot_id, query_dataframe, message, calculated_sequence) = data
            if query_dataframe is None:
                query_dataframe = pd.DataFrame()

            
            cursor.execute(self.insert_data_template, (uniprot_id, query_dataframe.to_json(), message, calculated_sequence))
            cursor.execute(self.insert_results_template, (uniprot_id, json.dumps(predicted_results)))
            self.mysql_conn.commit()
        except Error as e:
            print("Error while inserting data into MySQL", e)
            # self._reconnect_mysql()
            if self.use_mysql:
                self._insert_data(data, predicted_results)  # 重试插入操作
        finally:
            if cursor:
                cursor.close()
   
    def _query_data(self, uniprot_id):
        try:
            cursor = self.mysql_conn.cursor()
            cursor.execute(self.query_data_template, (uniprot_id,))
            query_data = cursor.fetchone()
            # cursor.close()
            
            # cursor = self.mysql_conn.cursor()
            cursor.execute(self.query_results_template, (uniprot_id,))
            predicted_results = cursor.fetchone()
            return query_data, predicted_results
        except Error as e:
            print("Error while querying data from MySQL", e)
            # self._reconnect_mysql()
            if self.use_mysql:
                return self._query_data(uniprot_id)  # 重试查询操作
            else:
                return None, None
        finally:
            if cursor:
                cursor.close()


        
    def query_from_uniprot(self, uniprot_id):
        self._check_and_reconnect_mysql()
        is_new_data = False
        
        if self.use_mysql:
            try:
                stored_query_data, stored_predicted_results = self._query_data(uniprot_id)
            except:
                stored_query_data, stored_predicted_results = (), ()
        else:
            stored_query_data, stored_predicted_results = (), ()
        if stored_predicted_results and stored_query_data and self.use_mysql:
            
            uniprot_id, query_dataframe_json, message, calculated_sequence = stored_query_data
            query_dataframe = pd.read_json(query_dataframe_json)
            if query_dataframe.empty:
                query_dataframe = None
            
            results_uniprot_id, predicted_results = stored_predicted_results
            assert results_uniprot_id == uniprot_id
            predicted_results = json.loads(predicted_results)
            
            return (uniprot_id, query_dataframe, message, calculated_sequence), predicted_results, is_new_data
        
        else:
            is_new_data = True
            query_dataframe, message = self.unprot_parser.parse_from_uniprotkb_query(uniprot_id)
            
            query_data =  (uniprot_id, query_dataframe, message, None)
            return query_data, None, is_new_data
    
    def insert_to_local_database(self, uniprot_id, query_dataframe, message, calculated_sequence, predicted_results):
        self._check_and_reconnect_mysql()
        if self.use_mysql:
            insert_data = (uniprot_id, query_dataframe, message, calculated_sequence)
            self._insert_data(insert_data, predicted_results)
        else:
            print('Warning: insert failure!')

        

        

if __name__ == '__main__':
    
    # unprot_parser = UniProtParser(chebi_path='../dataset/raw_dataset/chebi/structures.csv.gz', json_folder='test/uniprot_json', rxn_folder='test/rxn_folder', alphafolddb_folder='test/alphafolddb_folder')
    # # query_results_df = unprot_parser.parse_from_uniprotkb_query('F6KCZ5')
    # query_results_df, msg = unprot_parser.parse_from_uniprotkb_query('Q05BL1')
    
    unprot_mysql_parser = UniProtParserMysql(mysql_config_path='./mysql_config.json')
    query_data, predicted_results, is_new_data = unprot_mysql_parser.query_from_uniprot('Q05BL1')
    
    uniprot_id, query_dataframe, message, calculated_sequence = query_data
    if message != 'Good':
        predicted_results = []
        if is_new_data:
            unprot_mysql_parser.insert_to_local_database(uniprot_id=uniprot_id, query_dataframe=query_dataframe, message=message, calculated_sequence=calculated_sequence, predicted_results=predicted_results)
    elif message == 'Good':
        # prediction
        pass
    pass
    