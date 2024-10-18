import sys
import os
import argparse
import torch
import pandas as pd
sys.path.append('../')
from tqdm import tqdm
from webapp.utils import UniProtParserMysql, EasIFAInferenceAPI, retrain_ec_site_model_state_path, cmd



def inference_from_uniprot_ids(uniprot_id_list: list, ECSitePred: EasIFAInferenceAPI, unprot_mysql_parser: UniProtParserMysql):
    all_results_df = pd.DataFrame()

    for uniprot_id in tqdm(uniprot_id_list):
        query_data, _, _ = unprot_mysql_parser.query_from_uniprot(uniprot_id)
        uniprot_id, query_results_df, msg, _ = query_data
        query_results_df['UniProt ID'] = [uniprot_id for _ in range(len(query_results_df))]
        if query_results_df is None:
            msg = uniprot_id + ': ' + msg 
            print(msg)
            continue

        enzyme_aa_length = query_results_df["aa_length"].tolist()[0]
        if ECSitePred.max_enzyme_aa_length < enzyme_aa_length:
            query_results_df['predicted_results'] = [None for _ in range(len(query_results_df))]
            print(uniprot_id + ' sequence is too long! pass! (☼Д☼)')
            continue

        predicted_results = []

        for idx, row in enumerate(query_results_df.itertuples()):
            rxn = row[2]
            enzyme_structure_path = row[3]
            if not os.path.exists(enzyme_structure_path):
                enzyme_structure_path = os.path.join(
                    unprot_mysql_parser.unprot_parser.alphafolddb_folder,
                    f"AF-{uniprot_id}-F1-model_v4.pdb",
                )
                cmd(
                    unprot_mysql_parser.unprot_parser.download_alphafolddb_url_template.format(
                        enzyme_structure_path, uniprot_id
                    )
                )

            pred_active_site_labels = ECSitePred.inference(
                rxn=rxn, enzyme_structure_path=enzyme_structure_path
            )
            del ECSitePred.caculated_sequence
            predicted_results.append(pred_active_site_labels)
        query_results_df['predicted_results'] = predicted_results
        all_results_df = pd.concat([all_results_df, query_results_df], axis=0)
    all_results_df = all_results_df[['UniProt ID', 'ec','rxn_smiles','pdb_fpath','aa_length','predicted_results']]
    return all_results_df

def read_uniprot_ids_from_file(file_path: str):
    with open(file_path, 'r') as f:
        uniprot_ids = [line.strip() for line in f.readlines()]
    return uniprot_ids

def main():
    # 使用argparse获取输入和输出文件路径
    parser = argparse.ArgumentParser(description='Run prediction from UniProt IDs and output to CSV.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input txt file containing UniProt IDs.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output CSV file to save the prediction results.')
    
    args = parser.parse_args()
    
    # 读取UniProt ID列表
    uniprot_id_list = read_uniprot_ids_from_file(args.input)
    # 初始化设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 初始化预测模型
    ECSitePred = EasIFAInferenceAPI(
        model_checkpoint_path=retrain_ec_site_model_state_path, device=device
    )

    # 初始化 UniProt MySQL 解析器
    unprot_mysql_parser = UniProtParserMysql(
        mysql_config_path="./mysql_config.json", no_warning=True,
    )  
    # 运行预测
    all_results_df = inference_from_uniprot_ids(uniprot_id_list, ECSitePred, unprot_mysql_parser)
    
    # 将结果保存为CSV文件
    all_results_df.to_csv(args.output, index=False)
    print(f"Prediction results saved to {args.output}")

if __name__ == '__main__':
    main()
