from argparse import ArgumentParser
import math
import os
import time
import pandas as pd
import numpy as np
import torch
from torch.optim import Adam, lr_scheduler
from torch import nn as nn
from tqdm.auto import tqdm, trange
import yaml
from common.utils import check_args, read_model_state, save_model, set_seed, weight_loss
from transformers.optimization import get_linear_schedule_with_warmup
from model_structure.rxn_attn_model import ReactionMGMNet, ReactionMGMTurnNet

from data_loaders.rxn_dataloader import ReactionDataset, get_batch_data, MolGraphsCollator
from dgl.data.utils import Subset
from torch.utils.data import DataLoader


def train_one_epoch(args,
                    model,
                    train_dataloader,
                    val_dataloader,
                    device,
                    loss_fn,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    evaluate_in_train=False,
                    model_save_path=None,
                    best_eval_loss=np.inf):

    all_loss = 0.0

    model.train()

    pbar = tqdm(enumerate(train_dataloader),
                desc='Training',
                total=len(train_dataloader))
    for batch_id, batch_data in pbar:
        rts_bg, rts_adms, rts_node_feats, rts_edge_feats, pds_bg, pds_adms, pds_node_feats, pds_edge_feats, labels = get_batch_data(
            batch_data, device)
        

        optimizer.zero_grad()
        
        atom_type_logic, attentions = model(rts_bg, rts_adms, rts_node_feats,
                                            rts_edge_feats, pds_bg, pds_adms,
                                            pds_node_feats, pds_edge_feats)
        atom_type_labels = torch.cat(labels[:2], dim=0)

        loss = loss_fn(atom_type_logic, atom_type_labels)
        all_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_clip)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if batch_id % 10 == 0:

            lr = scheduler.get_last_lr()[0]
            pbar.set_description(
                '\rEpoch %d/%d, batch %d/%d, global_step: %d, lr: %.8f, loss %.4f'
                % (epoch + 1, args.num_epochs, batch_id + 1,
                   len(train_dataloader), global_step, lr, loss))
        if evaluate_in_train:
            if global_step % args.evaluate_pre_step == 0:
                eval_loss, accuracy = eval_one_epoch(model,
                                                     val_dataloader,
                                                     device=device,
                                                     loss_fn=loss_fn)
                model.train()
                eval_results = pd.DataFrame({
                    'eval_loss': [eval_loss],
                    'eval_accuracy': [accuracy],
                })
                print(
                    'Epoch: {}/{}, global_step: {}, evaluation loss: {:.4f}, evluaction accuracy: {:.4f}'
                    .format(epoch + 1, args.num_epochs, global_step, eval_loss,
                            accuracy))
                checkpoint_path = os.path.join(
                    model_save_path,
                    'checkpoint_epoch_{}-global_step_{}'.format(
                        epoch + 1, global_step + 1))
                save_model(model.state_dict(),
                               args=args.__dict__,
                               eval_results=eval_results,
                               model_save_path=checkpoint_path)
                if best_eval_loss > eval_loss:
                    best_eval_loss = eval_loss
                    save_model(model.state_dict(),
                               args=args.__dict__,
                               eval_results=eval_results,
                               model_save_path=model_save_path)

    print('\nEpoch %d/%d, loss %.4f' %
          (epoch + 1, args.num_epochs, all_loss / len(train_dataloader)))
    return global_step, best_eval_loss


def eval_one_epoch(model, data_loader, device, loss_fn):
    model.eval()
    all_loss = 0.0
    all_acc = []
    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader),
                    desc='Evaluating',
                    total=len(data_loader))
        for batch_id, batch_data in pbar:
            rts_bg, rts_adms, rts_node_feats, rts_edge_feats, pds_bg, pds_adms, pds_node_feats, pds_edge_feats, labels = get_batch_data(
                batch_data, device)
            atom_type_logic, attentions = model(rts_bg, rts_adms,
                                                rts_node_feats, rts_edge_feats,
                                                pds_bg, pds_adms,
                                                pds_node_feats, pds_edge_feats)
            atom_type_labels = torch.cat(labels[:2], dim=0)

            loss = loss_fn(atom_type_logic, atom_type_labels)
            all_loss += loss.item()

            pred_atom_types = torch.argmax(
                atom_type_logic[atom_type_labels != -100].softmax(dim=-1),
                dim=-1)
            atom_type_labels_unmask = atom_type_labels[
                atom_type_labels != -100]
            acc = (pred_atom_types == atom_type_labels_unmask).to(
                device=torch.device('cpu'))
            all_acc.append(acc)
    all_acc = torch.cat(all_acc)
    accuracy = (all_acc.sum() / len(all_acc)).item()

    return all_loss / len(data_loader), accuracy


def main(args):
    debug = False

    set_seed(123)
    model_save_path = 'checkpoints/reaction_attn_net/model-{}_train_in_{}_at_{}'.format(
       args.model_name, args.dataset_name, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    ) if not debug else 'checkpoints/reaction_attn_net/model_debug_in_{}'.format(
        args.dataset_name)

    print('########################## Args ##########################')
    print(yaml.dump(args.__dict__))
    print('##########################################################')
    print()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available(
    ) else 'cpu') if args.gpu != -1 else torch.device('cpu')

    dataset = ReactionDataset(args.dataset_name,
                              debug=debug,
                              nb_workers=args.num_workers,
                              use_atom_envs_type=args.use_atom_envs_type)
    print('Atomic environment vocabulary size: {}'.format(
        dataset.num_atom_types))
    print()
    train_set, val_set, _ = Subset(dataset, dataset.train_ids), Subset(
        dataset, dataset.val_ids), Subset(dataset, dataset.test_ids)

    collator = MolGraphsCollator(perform_mask=args.perform_mask,
                                 mask_percent=args.mask_percent)

    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(dataset=val_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=collator,
                                num_workers=args.num_workers)
    args.node_in_feats = dataset.node_dim
    args.edge_in_feats = dataset.edge_dim
    args.num_atom_types = dataset.num_atom_types

    # assert args.model_name in ['ReactionMGMTurnNet', 'ReactionMGMNet']

    model_structure_same = False
    if args.train_from_checkpoint != '':
        model_state, loaded_args = read_model_state(
            model_save_path=args.train_from_checkpoint)
        model_structure_same = check_args(args.__dict__, loaded_args)

    if args.model_name == 'ReactionMGMNet':

        model = ReactionMGMNet(
            node_in_feats=dataset.node_dim,
            node_out_feats=args.node_out_feats,
            edge_in_feats=dataset.edge_dim,
            edge_hidden_feats=args.edge_hidden_feats,
            num_step_message_passing=args.num_step_message_passing,
            attention_heads=args.attention_heads,
            attention_layers=args.attention_layers,
            dropout=args.dropout,
            num_atom_types=dataset.num_atom_types)
    elif args.model_name == 'ReactionMGMTurnNet':
        model = ReactionMGMTurnNet(
            node_in_feats=dataset.node_dim,
            node_out_feats=args.node_out_feats,
            edge_in_feats=dataset.edge_dim,
            edge_hidden_feats=args.edge_hidden_feats,
            num_step_message_passing=args.num_step_message_passing,
            attention_heads=args.attention_heads,
            attention_layers=args.attention_layers,
            dropout=args.dropout,
            cross_attn_h_rate=args.cross_attn_h_rate,
            num_atom_types=dataset.num_atom_types)
    else:
        raise ValueError

    if model_structure_same:
        model.load_state_dict(model_state)
        print('load checkpoint from {}'.format(args.train_from_checkpoint))

    model = model.to(device)

    loss_fn = weight_loss(class_num=dataset.num_atom_types,
                          negative_class_index=dataset.atom_types.index('C')
                          if not args.use_atom_envs_type else None,
                          negative_weight=args.negative_weight,
                          device=device)  # 设置C的权重为0.8, 其余为1
    total_step = (len(train_dataloader) * args.num_epochs)

    optimizer = Adam(model.parameters(),
                     lr=args.learning_rate,
                     weight_decay=args.weight_decay)
    warmup_steps = math.ceil(total_step * args.warmup_raio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_step,
    )

    epochs = args.num_epochs
    best_eval_loss = np.inf
    global_step = 0
    for epoch in range(epochs):
        global_step, best_eval_loss = train_one_epoch(
            args,
            model,
            train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            evaluate_in_train=args.evaluate_in_train,
            model_save_path=model_save_path,
            best_eval_loss=best_eval_loss)
        eval_loss, accuracy = eval_one_epoch(model,
                                             val_dataloader,
                                             device=device,
                                             loss_fn=loss_fn)
        print(
            'Epoch: {}/{}, evaluation loss: {:.4f}, evluaction accuracy: {:.4f}'
            .format(epoch + 1, epochs, eval_loss, accuracy))
        eval_results = pd.DataFrame({
            'eval_loss': [eval_loss],
            'eval_accuracy': [accuracy],
        })
        checkpoint_path = os.path.join(model_save_path,
                                       'checkpoint_epoch_{}'.format(epoch + 1))
        save_model(model.state_dict(),
                   args=args.__dict__,
                   eval_results=eval_results,
                   model_save_path=checkpoint_path)
        if best_eval_loss > eval_loss:
            best_eval_loss = eval_loss
            save_model(model.state_dict(),
                       args=args.__dict__,
                       eval_results=eval_results,
                       model_save_path=model_save_path)
    print('Done!')


if __name__ == '__main__':

    parser = ArgumentParser('Training arguements')
    parser.add_argument('--gpu', default=0, type=int, help='GPU device to use')
    parser.add_argument('--model_name', type=str, default='ReactionMGMTurnNet')
    parser.add_argument('--dataset_name', type=str, default='pistachio')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size of dataloader')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Maximum number of epochs for training')
    parser.add_argument('--negative_weight',
                        type=float,
                        default=0.5,
                        help='Loss weight for negative labels')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-4,
                        help='Learning rate of optimizer')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-6,
                        help='Weight Decay')
    parser.add_argument('--warmup_raio',
                        type=float,
                        default=0.05,
                        help='Warmup Raio')
    parser.add_argument('--num_workers',
                        type=int,
                        default=14,
                        help='Number of processes for data loading')
    parser.add_argument('--node_out_feats', type=int, default=32)
    parser.add_argument('--edge_hidden_feats', type=int, default=32)
    parser.add_argument('--num_step_message_passing', type=int, default=3)
    parser.add_argument('--attention_heads', type=int, default=4)
    parser.add_argument('--attention_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--max_clip', type=int, default=20)
    parser.add_argument('--train_from_checkpoint', type=str, default='')
    parser.add_argument('--perform_mask', action="store_true")
    parser.add_argument('--use_atom_envs_type', action="store_true")
    parser.add_argument('--evaluate_in_train', action="store_true")
    parser.add_argument('--evaluate_pre_step',
                        type=int,
                        default=1000,
                        help='evaluate pre step at training')
    parser.add_argument('--mask_percent',
                        type=float,
                        default=0.15,
                        help='Atom Mask Percent')
    
    parser.add_argument('--cross_attn_h_rate',
                        type=float,
                        default=0.1,
                        help='h rate in Cross Attention')

    args = parser.parse_args()
    main(args)