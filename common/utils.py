from collections import OrderedDict, defaultdict
import os
import random
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, recall_score
import torch
from torch import nn as nn
import yaml
import dgl
import pandas as pd
from torchdrug.utils import comm
import shutil
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
from matplotlib import pyplot as plt

def save_model(model_state, args, eval_results, model_save_path):
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    model_state_fname = os.path.join(model_save_path, 'model.pth')
    args_fname = os.path.join(model_save_path, 'args.yml')
    eval_results_fname = os.path.join(model_save_path, 'eval_results.csv')

    torch.save(model_state, model_state_fname)
    with open(args_fname, 'w', encoding='utf-8') as f:
        yaml.dump(args, f)
    eval_results.to_csv(eval_results_fname, index=False)

def delete_ckpt(path):
    try:
        shutil.rmtree(path)
        print(f"delete {path}")
    except OSError as e:
        print(f"delete {path} failure: {e}")


def weight_loss(class_num,
                negative_weight,
                device,
                negative_class_index=None,
                reduction='mean'):
    weights = torch.ones(class_num)
    if negative_class_index is not None:
        weights[negative_class_index] = negative_weight
    return nn.CrossEntropyLoss(weight=weights.to(device), reduction=reduction)


def check_args(now_args: dict, loaded_args: dict):
    check_items = [
        'node_out_feats', 'edge_hidden_feats', 'num_step_message_passing',
        'attention_heads', 'attention_layers'
    ]
    for item in check_items:
        if now_args[item] != loaded_args[item]:
            return False
    return True


def read_model_state(model_save_path):
    model_state_fname = os.path.join(model_save_path, 'model.pth')
    args_fname = os.path.join(model_save_path, 'args.yml')
    # eval_results_fname = os.path.join(model_save_path, 'eval_results.csv')

    model_state = torch.load(model_state_fname,
                             map_location=torch.device('cpu'))
    keys = list(model_state.keys())
    if 'module.' in keys[0]:
        model_state = {k.replace('module.', ''): v for k,v in model_state.items()}
    args = yaml.load(open(args_fname, "r"), Loader=yaml.FullLoader)

    return model_state, args

def load_pretrain_model_state(model, pretrained_state, load_active_net=True):
    model_state = model.state_dict()
    pretrained_state_filter = {}
    extra_layers = []
    different_shape_layers = []
    need_train_layers = []
    for name, parameter in pretrained_state.items():
        if name in model_state and parameter.size() == model_state[name].size():
            pretrained_state_filter[name] = parameter
        elif name not in model_state:
            extra_layers.append(name)
        elif parameter.size() != model_state[name].size():
            different_shape_layers.append(name)
        else:
            pass
    if not load_active_net:
        for name, parameter in model_state.items():
            if 'active_net' in name:
                del pretrained_state_filter[name]
    
    for name, parameter in model_state.items():
        if name not in pretrained_state_filter:
            need_train_layers.append(name)

    model_state.update(pretrained_state_filter)
    model.load_state_dict(model_state)
    
    print('Extra layers:', extra_layers)
    print('Different shape layers:', different_shape_layers)
    print('Need to train layers:', need_train_layers)
    return model


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def cuda(obj, *args, **kwargs):
    """
    Transfer any nested container of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, (str, bytes)):
        return obj
    elif isinstance(obj, dict):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)
    
    elif isinstance(obj, dgl.DGLGraph):
        return obj.to(*args, **kwargs)

    raise TypeError("Can't transfer object type `%s`" % type(obj))


class Recorder:
    def __init__(self, path, keep_num_ckpt=10, validation_recorder_columns=['global step', 'evaluation loss', 'overlap score', 'false positive rate']) -> None:
        self.path = path
        self.keep_num_ckpt = keep_num_ckpt
        os.makedirs(path, exist_ok=True)
        self.train_recorder = pd.DataFrame(columns=['global step', 'learning rate', 'training loss'])
        self.train_recorder.to_csv(os.path.join(self.path, 'train_info.csv'), index=False)

        self.validation_recorder = pd.DataFrame(columns=validation_recorder_columns)
        self.validation_recorder.to_csv(os.path.join(self.path, 'valid_info.csv'), index=False)
        self.ckpt_dict = {}
        self.updated = False


    
    def record(self, data, mode='train', criteria='evaluation loss'):
        if mode =='train':
            train_recorder = pd.DataFrame(data, columns=self.train_recorder.columns)
            self.train_recorder = self.train_recorder.append(train_recorder, ignore_index=True)
            train_recorder.to_csv(os.path.join(self.path, 'train_info.csv'), mode='a', index=False, header=None)
        elif mode == 'valid':
            assert criteria in ['evaluation loss', 'overlap score', 'false positive rate', 'negative accuracy']
            
            if (set(self.validation_recorder.columns) != set(data.keys())) and not self.updated:
                columns = list(data.keys())
                columns.sort()
                self.validation_recorder = pd.DataFrame(columns=columns)
                self.validation_recorder.to_csv(os.path.join(self.path, 'valid_info.csv'), index=False)
                self.updated = True
            validation_recorder = pd.DataFrame(data, columns=self.validation_recorder.columns)
            self.validation_recorder = self.validation_recorder.append(validation_recorder, ignore_index=True)
            validation_recorder.to_csv(os.path.join(self.path, 'valid_info.csv'), mode='a', index=False, header=None)
            self.ckpt_dict[data['global step'][0]] = data[criteria][0]
            self.criteria = criteria
        else:
            raise ValueError()
    
    def get_del_ckpt(self):
        if len(self.ckpt_dict) <= self.keep_num_ckpt:
            return []
        else:
            items = list(self.ckpt_dict.items())
            if self.criteria == 'overlap score':
                items.sort(key=lambda x:x[1], reverse=True)  #overlap score 越大越好, 排序最后的要删除
            elif self.criteria == 'negative accuracy':
                items.sort(key=lambda x:x[1], reverse=True)  #negative accuracy 越大越好, 排序最后的要删除
            else:
                items.sort(key=lambda x:x[1])   #loss 越小越好, 排序最后的要删除
            del_num = len(self.ckpt_dict) - self.keep_num_ckpt
            del_steps = [x[0] for x in items[-del_num:]]
            for step in del_steps:
                self.ckpt_dict.pop(step)
            return del_steps
        
        
def visualize_atom_attention(smiles, weights, cmap='RdBu', scale=-1, alpha=0, size=(150, 150), add_dummy_atom=False):
    
    if smiles != '':
        mol = Chem.MolFromSmiles(smiles)
        if add_dummy_atom and (mol.GetNumAtoms() == 1):
            mol = Chem.RWMol(mol)
            mol.AddAtom(Chem.Atom('*'))
            if weights[0] == 0:
                weights.append(1)
            else:
                weights.append(0)
    else:
        mol = Chem.MolFromSmiles('*-*')
        weights.append(0)
        weights.append(0)

        
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, colorMap=plt.get_cmap(cmap), scale=scale, alpha=alpha, size=size)
    return fig


def convert_fn(labels: torch.tensor, to_list=True):

    
    spase_labels_dict = OrderedDict()
    for site_type in range(4):
        spase_labels = torch.argwhere(labels==site_type).reshape(-1).tolist()
        spase_labels.sort()
        spase_labels_dict[site_type] = spase_labels
        
    def fix_continuous(labels:list):
        continuous_labels_lists = []
        remaining_labels = []   
        i = 0
        while i < len(labels):
            current_sequence = [labels[i]]
            j = i + 1
            while j < len(labels) and labels[j] == labels[j - 1] + 1:
                current_sequence.append(labels[j])
                j += 1
            if len(current_sequence) >= 2:
                continuous_labels_lists.append(current_sequence)
            else:
                remaining_labels.extend(current_sequence)
            i = j

        fixed_continuous_labels = []
        for continuous_labels in continuous_labels_lists:
            fixed_continuous_labels.extend(continuous_labels)
            fixed_continuous_labels.append(fixed_continuous_labels[-1] + 1)
        
        fixed_labels = fixed_continuous_labels + remaining_labels
        fixed_labels.sort()
        
        return fixed_labels
    fixed_spase_labels_dict = OrderedDict()
    for k in spase_labels_dict:
        if k == 0:
            fixed_spase_labels_dict[k] = spase_labels_dict[k]
        else:
            fixed_spase_labels_dict[k] = fix_continuous(spase_labels_dict[k])
    
    fix_labels = torch.zeros_like(labels)
    for k in fixed_spase_labels_dict:
        fix_labels[fixed_spase_labels_dict[k]] = k
    if to_list:
        return fix_labels.tolist()
    else:
        return fix_labels
    
    
def is_valid_outputs(protein_node_logic, targets):
    if protein_node_logic.size(0) != targets.size(0):
        return False
    return True



def calculate_score_vbin_train(pred_idx, gt_idx, num_residue):
    amino = [i for i in range(num_residue)]

    TP = len(pred_idx.intersection(gt_idx))
    FP = len(pred_idx.difference(gt_idx))
    TN = len(set(amino).difference((gt_idx).union(pred_idx)))

    overlap_score = TP / len(gt_idx) if len(gt_idx) != 0 else 0
    fpr = FP / (FP + TN)
    return overlap_score, fpr

def get_fpr(cm, class_idx):
    TN, FP, FN, TP = cm[class_idx].ravel()
    fpr = FP / (FP + TN)
    return fpr

def calculate_score_vbin_test(pred_idx, gt_idx, num_residue):
    amino = [i for i in range(num_residue)]

    TP = len(pred_idx.intersection(gt_idx))
    FP = len(pred_idx.difference(gt_idx))
    TN = len(set(amino).difference((gt_idx).union(pred_idx)))
    FN = len(gt_idx) - TP

    acc = (TP + TN) / (TP + TN + FP + FN)
    prec = TP / (TP + FP) if len(pred_idx) !=0 else 0
    spec = TN / (TN + FP) if (TN + FP) != 0 else 1

    overlap_score = TP / len(gt_idx) if len(gt_idx) != 0 else 0  # recall
    fpr = FP / (FP + TN)
    f1 = 2 * (prec * overlap_score) / (prec + overlap_score) if  (prec + overlap_score) !=0 else 0
    return acc, prec, spec, overlap_score, fpr, f1


def calculate_scores_vbin_test(preds, gts, num_residues):
    accuracy = []
    precision = []
    overlap_scores = []
    fpr_list = []
    f1_scores = []
    specificity = []
    preds = torch.split(preds, num_residues)
    gts = torch.split(gts, num_residues)

    for pred, gt, num_residue in zip(preds, gts, num_residues):
        pred_idx = set(torch.argwhere(pred == 1).view(-1).tolist())
        gt_idx = set(torch.argwhere(gt == 1).view(-1).tolist())
        acc, prec, spec, overlap_score, fpr, f1 = calculate_score_vbin_test(pred_idx, gt_idx, num_residue)
        accuracy.append(acc)
        precision.append(prec)
        overlap_scores.append(overlap_score)
        fpr_list.append(fpr)
        specificity.append(spec)
        f1_scores.append(f1)

    return accuracy, precision, specificity, overlap_scores, fpr_list, f1_scores

def get_fpr(cm, class_idx):
    TN, FP, FN, TP = cm[class_idx].ravel()
    fpr = FP / (FP + TN)
    return fpr



def calculate_metrics_multi_class(pred, gt, num_site_types):
    metrics = defaultdict(list)
    recall_list = recall_score(gt, pred, average=None, labels=range(num_site_types), zero_division=0)
    cm = multilabel_confusion_matrix(gt, pred, labels=range(num_site_types))
    
    for class_idx in range(num_site_types):
        metrics[f'recall_cls_{class_idx}'].append(recall_list[class_idx])
        metrics[f'fpr_cls_{class_idx}'].append(get_fpr(cm, class_idx=class_idx))
    
    return metrics


def calculate_scores_vmulti_test(preds, gts, num_residues, num_site_types):
    metrics = defaultdict(list)
    
    bin_preds = (preds != 0).long()
    bin_gts = (gts != 0).long()
    preds = torch.split(preds, num_residues)
    gts = torch.split(gts, num_residues)

    for pred, gt, num_residue in zip(preds, gts, num_residues):
        pred_idx = set(torch.argwhere(bin_preds == 1).view(-1).tolist())
        gt_idx = set(torch.argwhere(bin_gts == 1).view(-1).tolist())
        acc, prec, spec, overlap_score, fpr, f1 = calculate_score_vbin_test(pred_idx, gt_idx, num_residue)
        
        metrics['overlap_scores'].append(overlap_score)
        metrics['false_positive_rates'].append(fpr)
        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['specificity'].append(spec)
        metrics['f1_scores'].append(f1)
        
        metrics_multi_class = calculate_metrics_multi_class(pred.tolist(), gt.tolist(), num_site_types=num_site_types)
        for key in metrics_multi_class:
            metrics[key] += metrics_multi_class[key]
        

    return metrics

def calculate_scores_vbin_train(preds, gts, num_residues):
    overlap_scores = []
    fpr_list = []
    preds = torch.split(preds, num_residues)
    gts = torch.split(gts, num_residues)

    for pred, gt, num_residue in zip(preds, gts, num_residues):
        pred_idx = set(torch.argwhere(pred == 1).view(-1).tolist())
        gt_idx = set(torch.argwhere(gt == 1).view(-1).tolist())
        overlap_score, fpr = calculate_score_vbin_train(
            pred_idx, gt_idx, num_residue)
        overlap_scores.append(overlap_score)
        fpr_list.append(fpr)

    return overlap_scores, fpr_list

   


def calculate_scores_vmulti_train(preds, gts, num_residues, num_site_types):
    metrics = defaultdict(list)
    bin_preds = (preds != 0).long()
    bin_gts = (gts != 0).long()
    
    
    preds = torch.split(preds, num_residues)
    gts = torch.split(gts, num_residues)
    bin_preds = torch.split(bin_preds, num_residues)
    bin_gts = torch.split(bin_gts, num_residues)

    for pred, gt, b_pred, b_gt, num_residue in zip(preds, gts, bin_preds, bin_gts, num_residues):
        pred_idx = set(torch.argwhere(b_pred == 1).view(-1).tolist())
        gt_idx = set(torch.argwhere(b_gt == 1).view(-1).tolist())
        overlap_score, fpr = calculate_score_vbin_train(
            pred_idx, gt_idx, num_residue)
        metrics['overlap_scores'].append(overlap_score)
        metrics['false_positive_rates'].append(fpr)
        
        metrics_multi_class = calculate_metrics_multi_class(pred.tolist(), gt.tolist(), num_site_types=num_site_types)
        for key in metrics_multi_class:
            metrics[key] += metrics_multi_class[key]

    return metrics




def calculate_overlap_fpr_for_pos(pred_idx, gt_idx, num_residue):
    amino = [i for i in range(num_residue)]

    TP = len(pred_idx.intersection(gt_idx))
    FP = len(pred_idx.difference(gt_idx))
    TN = len(set(amino).difference((gt_idx).union(pred_idx)))

    overlap_score = TP / len(gt_idx) if len(gt_idx) != 0 else 0
    fpr = FP / (FP + TN)
    return overlap_score, fpr

def calculate_accuracy_for_neg(pred_idx):
    if len(pred_idx) == 0:
        return 1
    else:
        return 0

def calculate_scores_ood(preds, gts, num_residues):
    overlap_scores = []
    fpr_list = []
    preds = torch.split(preds, num_residues)
    gts = torch.split(gts, num_residues)
    
    neg_accuracy = []

    for pred, gt, num_residue in zip(preds, gts, num_residues):
        pred_idx = set(torch.argwhere(pred == 1).view(-1).tolist())
        gt_idx = set(torch.argwhere(gt == 1).view(-1).tolist())
        if len(gt_idx) != 0:
            overlap_score, fpr = calculate_overlap_fpr_for_pos(
                pred_idx, gt_idx, num_residue)
            overlap_scores.append(overlap_score)
            fpr_list.append(fpr)
        else:
            neg_accuracy.append(calculate_accuracy_for_neg(pred_idx))

    assert len(overlap_scores) + len(neg_accuracy) == len(preds)
    return overlap_scores, fpr_list, neg_accuracy


