import math
import os
import argparse
import time
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn as nn
from torch.utils import data as torch_data
from data_loaders.enzyme_rxn_dataloader import AugEnzymeReactionDataset, EnzymeReactionDataset, enzyme_rxn_collate_extract
from torch.optim import AdamW
from torch.utils.data import Subset
from torchdrug.utils import comm
import yaml

from model_structure.enzyme_site_model import EnzymeActiveSiteModel

from common.utils import calculate_scores_ood, check_args, cuda, Recorder, delete_ckpt, is_valid_outputs, load_pretrain_model_state, read_model_state, save_model
from transformers.optimization import get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings("ignore")






def evaluate(
        args,
        model,
        valid_set,
        device,
        loss_fn,
        evaluate_num_samples,

):

    model.eval()

    node_correct_cnt = 0
    node_cnt = 0

    overlap_scores = []
    false_positive_rates = []
    negative_correct = []
    all_loss = 0.0

    sampled_valid_set = Subset(valid_set, torch.randperm(
        len(valid_set))[:evaluate_num_samples])
    valid_dataloader = torch_data.DataLoader(
        sampled_valid_set,
        batch_size=args.batch_size,
        collate_fn=enzyme_rxn_collate_extract,
        num_workers=args.num_workers)

    pbar = tqdm(
        valid_dataloader,
        total=len(valid_dataloader),
        desc='Evaluating'
    )

    with torch.no_grad():

        for batch_id, batch in enumerate(pbar):
            if device.type == "cuda":
                batch = cuda(batch, device=device)
            protein_node_logic, protein_mask = model(batch)

            targets = batch['targets'].long()
            if not is_valid_outputs(protein_node_logic, targets):
                continue

            loss = loss_fn(protein_node_logic, targets)
            all_loss += loss.item()

            pred = torch.argmax(protein_node_logic.softmax(-1), dim=-1)
            correct = pred == targets
            node_correct_cnt += correct.sum().item()
            node_cnt += targets.size(0)

            os, fpr, neg_acc = calculate_scores_ood(
                pred, targets, batch['protein_graph'].num_residues.tolist())
            overlap_scores += os
            false_positive_rates += fpr
            negative_correct += neg_acc
    model.train()
    
    if len(negative_correct) == 0:
        negative_accuracies = float('nan')
    else:
        negative_accuracies = sum(negative_correct) / len(negative_correct)
    
    if len(overlap_scores) != 0:
        return all_loss / len(valid_dataloader), sum(overlap_scores) / len(overlap_scores), sum(false_positive_rates) / len(false_positive_rates), negative_accuracies
    else:
        return all_loss / len(valid_dataloader), 0.0, 0.0, negative_accuracies


def train_one_epoch(
        args,
        model,
        train_dataloader,
        valid_set,
        device,
        loss_fn,
        optimizer,
        scheduler,
        epoch,
        global_step,
        world_size,
        recorder: Recorder,
        model_save_path,
):

    model.train()
    pbar = tqdm(
        train_dataloader,
        total=len(train_dataloader),
        desc='Training',
    )

    # batch_per_epoch = len(train_dataloader)
    # gradient_interval = min(batch_per_epoch - global_step, gradient_interval)

    all_loss = 0.0

    for batch_id, batch in enumerate(pbar):
        # for batch_id, batch in enumerate(train_dataloader):
        if not batch: continue
        if device.type == "cuda":
            batch = cuda(batch, device=device)
        # print(batch_id)

        protein_node_logic, protein_mask = model(batch)

        targets = batch['targets']
        if not is_valid_outputs(protein_node_logic, targets):
            print('Warning! this batch is not valid, protein residues number {}, targets length: {}'.format(
                batch['protein_graph'].num_residues.sum(), targets.size(0)))
            continue

        loss = loss_fn(protein_node_logic, targets.long())
        loss = loss / args.gradient_interval
        all_loss += loss.item()

        loss.backward()

        if (global_step + 1) % args.gradient_interval == 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_clip)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        report_global_step = world_size * global_step
        if (report_global_step % 10 == 0) and (comm.get_rank() == 0):
            lr = scheduler.get_last_lr()[0]

            pbar.set_description(
                '\rEpoch %d/%d, batch %d/%d, global step: %d, lr: %.8f, loss %.4f'
                % (epoch + 1, args.num_epochs, batch_id + 1,
                   len(train_dataloader), report_global_step, lr, loss.item()))
            recorder.record(
                data={
                    'global step': [report_global_step],
                    'learning rate': [lr],
                    'training loss': [loss.item()]
                }, mode='train'
            )

        scheduler.step()

        # if (report_global_step % args.evaluate_per_step == 0) and (comm.get_rank() == 0):
        if (report_global_step % args.evaluate_per_step == 0):
            evaluate_loss, overlap_scores, false_positive_rates, negative_accuracies = evaluate(
                args,
                model=model,
                valid_set=valid_set,
                device=device,
                loss_fn=loss_fn,
                evaluate_num_samples=args.evaluate_num_samples)
            print('\n\rEvaluation: Epoch %d/%d, batch %d/%d, global step: %d, evaluation loss %.4f, Postive Scores: (overlap score  %.4f, false positive rate  %.4f), Negative Score:(negative accuracy %.4f)\n'
                  % (epoch + 1, args.num_epochs, batch_id + 1,
                     len(train_dataloader), report_global_step, evaluate_loss, overlap_scores, false_positive_rates, negative_accuracies))

            eval_results = {
                'global step': [report_global_step],
                'evaluation loss': [evaluate_loss],
                'overlap score': [overlap_scores],
                'false positive rate': [false_positive_rates],
                'negative accuracy': [negative_accuracies]
            }
            if comm.get_rank() == 0:

                recorder.record(
                    data=eval_results, mode='valid',
                    criteria=args.criteria
                )

                eval_results = pd.DataFrame(
                    eval_results
                )
                eval_model_save_path = os.path.join(
                    model_save_path, f'global_step_{report_global_step}')

                save_model(model.state_dict(), args.__dict__,
                           eval_results, eval_model_save_path)

                del_model_ckpt_list = [os.path.join(
                    model_save_path, f'global_step_{step}') for step in recorder.get_del_ckpt()]
                for ckpt_path in del_model_ckpt_list:
                    delete_ckpt(ckpt_path)

    print('\nEpoch %d/%d, loss %.4f' %
          (epoch + 1, args.num_epochs, all_loss / len(train_dataloader)))

    return global_step


def main(args):

    print(yaml.dump(args.__dict__))

    world_size = comm.get_world_size()
    gpus = args.gpus
    
    if not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")
    local_rank = comm.get_rank()
    device_id = gpus[local_rank]
    device = torch.device("cuda", device_id)

    debug = args.debug
    
    dataset = AugEnzymeReactionDataset(
        path=args.dataset_path,
        structure_path=args.structure_path,
        save_precessed=False, 
        debug=debug, 
        verbose=1, 
        protein_max_length=1000, 
        lazy=True,
        use_aug=args.use_aug,
        soft_check=args.soft_check,
        nb_workers=args.propcess_core)

    train_set, valid_set, _ = dataset.split()

    train_sampler = torch_data.DistributedSampler(train_set)

    train_dataloader = torch_data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=enzyme_rxn_collate_extract,
        num_workers=args.num_workers)

    model = EnzymeActiveSiteModel(
        rxn_model_path=args.pretrained_rxn_attn_model_path)

    if args.train_from_checkpoint != '':
        model_state, _ = read_model_state(
            model_save_path=args.train_from_checkpoint)
        if not args.not_load_active_net:
            try:
                model.load_state_dict(model_state)
                print('Loaded checkpoint from {}'.format(
                    args.train_from_checkpoint))
            except:
                print('Load the checkpoint failed, train from scratch')
        else:
            model = load_pretrain_model_state(model, model_state, load_active_net=False)

    model.to(device)

    dataset_name = os.path.split(args.dataset_path)[-1]
    model_save_path = os.path.join(
        args.checkpoint_path,
        'train_in_{}_at_{}'.format(
            dataset_name, time.strftime(
                "%Y-%m-%d-%H-%M-%S",
                time.localtime())
        ))
    if debug:
        model_save_path = os.path.join(
            args.checkpoint_path, 'debug'
        )

    model = DDP(model, find_unused_parameters=False, device_ids=[device_id])

    pretrained_model_lr_ratio = {
        'enzyme_attn_model.model.sequence_model': args.protein_sequnece_model_lr_ratio,
        'rxn_attn_model': args.reaction_attention_model_lr_ratio
    }

    params_groups = []

    for i, named_param in enumerate(model.named_parameters()):
        param_dict = {}
        name, param = named_param
        param_dict['params'] = param
        param_dict['lr'] = args.lr
        for pretrain_name in pretrained_model_lr_ratio:
            if pretrain_name in name:
                param_dict['lr'] = args.lr * \
                    pretrained_model_lr_ratio[pretrain_name]
        param_dict['params_name'] = name
        params_groups.append(param_dict)

    optimizer = AdamW(params_groups, lr=args.lr,
                      weight_decay=args.weight_decay)
    total_step = (len(train_dataloader) * args.num_epochs)
    warmup_steps = math.ceil(total_step * args.warmup_raio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_step,
    )

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([args.weight_negative, 1]).to(device))

    recoder = Recorder(model_save_path, keep_num_ckpt=args.keep_num_ckpt, validation_recorder_columns=['global step', 'evaluation loss', 'overlap score', 'false positive rate', 'negative accuracy'])

    global_step = 0
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)

        global_step = train_one_epoch(
            args,
            model=model,
            train_dataloader=train_dataloader,
            valid_set=valid_set,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            epoch=epoch,
            global_step=global_step,
            world_size=world_size,
            recorder=recoder,
            model_save_path=model_save_path
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Finetune arguements')
    parser.add_argument('--gpus',
                        type=int,
                        nargs='+',
                        default=[0],
                        help='CUDA devices id to use')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help='Learning rate of optimizer')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-6,
                        help='Weight Decay')
    parser.add_argument('--batch_size',
                        type=int,
                        default=2,
                        help='Batch size of dataloader')
    parser.add_argument('--keep_num_ckpt',
                        type=int,
                        default=10,
                        help='Keep checkpoint numbers')
    parser.add_argument('--warmup_raio',
                        type=float,
                        default=0.01,
                        help='Warmup Raio')
    parser.add_argument('--weight_negative',
                        type=float,
                        default=0.5,
                        help='Negative class weight')
    parser.add_argument('--dataset_path',
                        type=str,
                        default='dataset/mcsa_fine_tune/normal_mcsa',
                        help='dataset path')
    parser.add_argument('--structure_path',
                        type=str,
                        default='dataset/mcsa_fine_tune/structures/alphafolddb_download',)
    parser.add_argument('--use_aug',
                        action='store_true',)
    parser.add_argument('--soft_check',
                        action='store_true',)
    parser.add_argument('--pretrained_rxn_attn_model_path',
                        type=str,
                        default='checkpoints/reaction_attn_net/model-ReactionMGMTurnNet_train_in_uspto_at_2023-04-05-23-46-25',
                        help='Pretrained reaction attention model path')
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default='checkpoints/enzyme_site_predition_model/',
                        help='Pretrained reaction attention model path')
    parser.add_argument('--train_from_checkpoint',
                        type=str,
                        default='',
                        help='Train from checkpoint')
    parser.add_argument('--not_load_active_net',
                        action='store_true',
                        help='Train from checkpoint and not load active net')
    parser.add_argument('--num_workers',
                        type=int,
                        default=2,
                        help='Number of processes for data loader')
    parser.add_argument('--propcess_core',
                        type=int,
                        default=16,
                        help='Number of processes for data preprocess')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=10,
                        help='Maximum number of epochs for training')
    parser.add_argument('--max_clip', type=int, default=20)
    parser.add_argument('--gradient_interval', type=int, default=1)
    parser.add_argument('--evaluate_per_step',
                        type=int,
                        default=10000,
                        help='evaluate pre step at training')
    parser.add_argument('--evaluate_num_samples',
                        type=int,
                        default=1000,
                        )
    parser.add_argument('--protein_sequnece_model_lr_ratio',
                        type=float,
                        default=0.1)
    parser.add_argument('--reaction_attention_model_lr_ratio',
                        type=float,
                        default=0.1)
    parser.add_argument('--criteria',
                        type=str,
                        default='evaluation loss',
                        )


    args = parser.parse_args()

    args.debug = False

    main(args)
