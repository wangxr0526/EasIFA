import argparse
import os
from tqdm.auto import tqdm
from collections import defaultdict
import torch
from torch.utils import data as torch_data
from common.utils import calculate_scores_vbin_test, calculate_scores_vmulti_test, convert_fn, cuda, read_model_state
from data_loaders.enzyme_rxn_dataloader import EnzymeReactionDataset, EnzymeReactionSiteTypeDataset,  enzyme_rxn_collate_extract
from model_structure.enzyme_site_model import EnzymeActiveSiteClsModel, EnzymeActiveSiteESMModel, EnzymeActiveSiteModel
from main_train import is_valid_outputs



def main(args):
    device = torch.device(f"cuda:{args.gpu}") if (
        args.gpu >= 0) and torch.cuda.is_available() else torch.device('cpu')

    if args.task_type == 'active-site-categorie-prediction':
        dataset = EnzymeReactionSiteTypeDataset(path=args.dataset_path,
                                    save_precessed=False,
                                    debug=False,
                                    verbose=1,
                                    test_remove_aegan_train=args.test_remove_aegan_train,
                                    lazy=True,
                                    nb_workers=12)
    else:
        dataset = EnzymeReactionDataset(path=args.dataset_path,
                                        save_precessed=False,
                                        debug=False,
                                        verbose=1,
                                        lazy=True,
                                        nb_workers=12)
    _, _, test_dataset = dataset.split()

    test_dataloader = torch_data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=enzyme_rxn_collate_extract,
        shuffle=False,
        num_workers=4)
    
    
    if args.task_type == 'active-site-position-prediction':
        model = EnzymeActiveSiteModel(
            rxn_model_path=args.pretrained_rxn_attn_model_path
            )
    elif args.task_type == 'active-site-categorie-prediction':
        model = EnzymeActiveSiteClsModel(
            rxn_model_path=args.pretrained_rxn_attn_model_path, num_active_site_type=dataset.num_active_site_type
            )
    elif args.task_type == 'ablation-experiment-1':
        model = EnzymeActiveSiteESMModel(
            bridge_hidden_dim=args.bridge_hidden_dim
        )
    
    elif args.task_type == 'ablation-experiment-2':
        model = EnzymeActiveSiteModel(
            rxn_model_path=args.pretrained_rxn_attn_model_path, 
            from_scratch=True,
            )    # 这里传递预训练的模型只是为了初始化模型的形状，但是模型state并不继承
    else:
        raise ValueError('Task erro')


    model_state, model_args = read_model_state(model_save_path=args.checkpoint)
    need_convert = model_args.get('need_convert', False)  
    model.load_state_dict(model_state)
    print('Loaded checkpoint from {}'.format(os.path.abspath(args.checkpoint)))
    model.to(device)
    model.eval()

    node_correct_cnt = 0
    node_cnt = 0
    
    
    accuracy_list = []
    precision_list = []
    specificity_list = []
    overlap_scores_list = []
    false_positive_rates_list = []
    f1_scores_list = []
    
    if args.task_type == 'active-site-categorie-prediction':
        metrics_collection = defaultdict(list)
    
        multicls_cols = ['recall_cls_0', 'recall_cls_1', 'recall_cls_2', 'recall_cls_3', 'fpr_cls_0', 'fpr_cls_1', 'fpr_cls_2', 'fpr_cls_3']
    


    pbar = tqdm(test_dataloader, total=len(test_dataloader), desc='Testing')
    with torch.no_grad():
        for batch_id, batch in enumerate(pbar):
            if device.type == "cuda":
                batch = cuda(batch, device=device)
            try:
                if args.task_type == 'ablation-experiment-1':
                    protein_node_logic = model(batch)
                else:
                    protein_node_logic, _ = model(batch)
            except:
                
                print(f'erro in batch: {batch_id}')
                continue

            targets = batch['targets'].long()
            if not is_valid_outputs(protein_node_logic, targets):
                print(f'erro in batch: {batch_id}')
                continue

            pred = torch.argmax(protein_node_logic.softmax(-1), dim=-1)
            if need_convert:
                pred = convert_fn(pred, to_list=False)
            correct = pred == targets
            node_correct_cnt += correct.sum().item()
            node_cnt += targets.size(0)

            if args.task_type == 'active-site-categorie-prediction':
                metrics = calculate_scores_vmulti_test(
                    pred, targets, batch['protein_graph'].num_residues.tolist(), num_site_types=dataset.num_active_site_type)
                accuracy_list += metrics['accuracy']
                precision_list += metrics['precision']
                specificity_list += metrics['specificity']
                overlap_scores_list += metrics['overlap_scores']
                false_positive_rates_list += metrics['false_positive_rates']
                f1_scores_list += metrics['f1_scores']
            

                for key in metrics:
                    metrics_collection[key] += metrics[key]
            else:
                accuracy, precision, specificity, overlap_score, fpr, f1_scores = calculate_scores_vbin_test(
                    pred, targets, batch['protein_graph'].num_residues.tolist()
                )
                accuracy_list += accuracy
                precision_list += precision
                specificity_list += specificity
                overlap_scores_list += overlap_score
                false_positive_rates_list += fpr
                f1_scores_list += f1_scores

            pbar.set_description(
                'Accuracy: {:.4f}, Precision: {:.4f}, Specificity: {:.4f}, Overlap Score: {:.4f}, False Positive Rate: {:.4f}, F1: {:.4f}'.format(
                    sum(accuracy_list) / len(accuracy_list),
                    sum(precision_list) / len(precision_list),
                    sum(specificity_list) / len(specificity_list),
                    sum(overlap_scores_list) / len(overlap_scores_list),
                    sum(false_positive_rates_list) / len(false_positive_rates_list),
                    sum(f1_scores_list) / len(f1_scores_list)
                    )
                )

    print(f'Get {len(overlap_scores_list)} results')
    print('Accuracy: {:.4f}, Precision: {:.4f}, Specificity: {:.4f}, Overlap Score: {:.4f}, False Positive Rate: {:.4f}, F1: {:.4f}'.format(
                    sum(accuracy_list) / len(accuracy_list),
                    sum(precision_list) / len(precision_list),
                    sum(specificity_list) / len(specificity_list),
                    sum(overlap_scores_list) / len(overlap_scores_list),
                    sum(false_positive_rates_list) / len(false_positive_rates_list),
                    sum(f1_scores_list) / len(f1_scores_list)
                    ))
    
    if args.task_type == 'active-site-categorie-prediction':
        print('Multiclassfication Metrics:')
        multiclass_report_str = ['{}: {:.4f}'.format(key, sum(metrics_collection[key])/len(metrics_collection[key])) for key in  multicls_cols]
        print(', '.join(multiclass_report_str))
        
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Test arguements')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='CUDA devices id to use')
    
    parser.add_argument('--task_type', 
                        choices=[
                            'active-site-position-prediction', 
                            'active-site-categorie-prediction', 
                            'ablation-experiment-1',      #消融实验1： 研究反应分支及酶-反应相互作用网络的作用
                            'ablation-experiment-2',      #消融实验2： 研究反应分支预训练的作用
                            ],
                        default='active-site-position-prediction',
                        help='Choose a task')
    
    parser.add_argument('--dataset_path',
                        type=str,
                        default='dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100',
                        help='Test dataset path')
    parser.add_argument('--pretrained_rxn_attn_model_path',
                        type=str,
                        default=
                        'checkpoints/reaction_attn_net/model-ReactionMGMTurnNet_train_in_uspto_at_2023-04-05-23-46-25',
                        help='Pretrained reaction representation branch')
    
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,  # 推荐batch size=1
                        help='Batch size of dataloader')
    
    parser.add_argument('--bridge_hidden_dim',
                        type=int,
                        default=128,
                        help='Bridge layer hidden size')   # 搭配消融实验1

    parser.add_argument('--checkpoint',
                        type=str,
                        default=
                        'checkpoints/enzyme_site_predition_model/train_in_uniprot_ecreact_merge_dataset_limit_100_at_2023-05-25-20-39-05/global_step_44000',
                        help='Pretrained reaction attention model path')
    
    parser.add_argument('--test_remove_aegan_train',
                        type=bool, 
                        default=False)

    args = parser.parse_args()
    main(args)