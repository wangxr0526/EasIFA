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
from data_loaders.enzyme_rxn_dataloader import (
    EnzymeReactionSaProtDataset,
    EnzymeReactionSiteTypeSaProtDataset,
    enzyme_rxn_collate_extract,
    EnzymeRxnSaprotCollate,
)
from torch.optim import AdamW
from torch.utils.data import Subset
from torchdrug.utils import comm
import yaml
from collections import defaultdict
from model_structure.enzyme_site_model import (
    EnzymeActiveSiteClsModel,
    EnzymeActiveSiteESMGearNetModel,
    EnzymeActiveSiteModel,
)
from common.utils import (
    calculate_scores_vbin_train,
    calculate_scores_vmulti_train,
    check_args,
    cuda,
    Recorder,
    delete_ckpt,
    is_valid_outputs,
    load_pretrain_model_state,
    read_model_state,
    save_model,
)
from transformers.optimization import get_linear_schedule_with_warmup

import warnings

warnings.filterwarnings("ignore")

enzyme_rxn_saprot_collate_extract = EnzymeRxnSaprotCollate()
foldseek_bin_path = "foldseek_bin/foldseek"
assert os.path.exists(
    foldseek_bin_path
), f"Need foldseek binary file, download it: https://drive.google.com/file/d/1B_9t3n_nlj8Y3Kpc_mMjtMdY0OPYa7Re/view, and put it at {foldseek_bin_path}"


def evaluate_vbin(
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
    all_loss = 0.0

    sampled_valid_set = Subset(
        valid_set, torch.randperm(len(valid_set))[:evaluate_num_samples]
    )
    valid_dataloader = torch_data.DataLoader(
        sampled_valid_set,
        batch_size=args.batch_size,
        collate_fn=enzyme_rxn_saprot_collate_extract,
        num_workers=args.num_workers,
    )

    pbar = tqdm(valid_dataloader, total=len(valid_dataloader), desc="Evaluating")

    with torch.no_grad():

        for batch_id, batch in enumerate(pbar):
            if device.type == "cuda":
                batch = cuda(batch, device=device)

            if args.task_type == "ablation-experiment-1":
                protein_node_logic = model(batch)
            else:
                protein_node_logic, _ = model(batch)

            targets = batch["targets"].long()
            if not is_valid_outputs(protein_node_logic, targets):
                continue

            loss = loss_fn(protein_node_logic, targets)
            all_loss += loss.item()

            pred = torch.argmax(protein_node_logic.softmax(-1), dim=-1)
            correct = pred == targets
            node_correct_cnt += correct.sum().item()
            node_cnt += targets.size(0)

            os, fpr = calculate_scores_vbin_train(
                pred, targets, batch["protein_graph"].num_residues.tolist()
            )
            overlap_scores += os
            false_positive_rates += fpr
    model.train()
    if len(overlap_scores) != 0:
        return (
            all_loss / len(valid_dataloader),
            sum(overlap_scores) / len(overlap_scores),
            sum(false_positive_rates) / len(false_positive_rates),
        )
    else:
        return all_loss / len(valid_dataloader), 0.0, 0.0


def evaluate_vmulti(
    args,
    model,
    valid_set,
    device,
    loss_fn,
    evaluate_num_samples,
    num_site_types,
):

    model.eval()

    node_correct_cnt = 0
    node_cnt = 0

    # overlap_scores = []
    # false_positive_rates = []
    all_loss = 0.0

    all_metrics = defaultdict(list)

    sampled_valid_set = Subset(
        valid_set, torch.randperm(len(valid_set))[:evaluate_num_samples]
    )
    valid_dataloader = torch_data.DataLoader(
        sampled_valid_set,
        batch_size=args.batch_size,
        collate_fn=enzyme_rxn_saprot_collate_extract,
        num_workers=args.num_workers,
    )

    pbar = tqdm(valid_dataloader, total=len(valid_dataloader), desc="Evaluating")

    with torch.no_grad():

        for batch_id, batch in enumerate(pbar):
            if device.type == "cuda":
                batch = cuda(batch, device=device)

            protein_node_logic, _ = model(batch)

            targets = batch["targets"].long()
            # targets_bin = (targets != 0).long()
            if not is_valid_outputs(protein_node_logic, targets):
                continue

            loss = loss_fn(protein_node_logic, targets)
            all_loss += loss.item()

            pred = torch.argmax(protein_node_logic.softmax(-1), dim=-1)
            # pred_bin = (pred_multi != 0).long()
            correct = pred == targets
            node_correct_cnt += correct.sum().item()
            node_cnt += targets.size(0)

            metrics = calculate_scores_vmulti_train(
                pred,
                targets,
                batch["protein_graph"].num_residues.tolist(),
                num_site_types=num_site_types,
            )
            for key in metrics:
                all_metrics[key] += metrics[key]

    model.train()
    overlap_scores = all_metrics.pop("overlap_scores")
    false_positive_rates = all_metrics.pop("false_positive_rates")

    mean_metrics = defaultdict(list)
    for key in metrics:
        if len(metrics[key]) != 0:
            mean_metrics[key].append(sum(all_metrics[key]) / len(all_metrics[key]))
        else:
            mean_metrics[key].append(0)

    if len(overlap_scores) != 0:
        return (
            all_loss / len(valid_dataloader),
            sum(overlap_scores) / len(overlap_scores),
            sum(false_positive_rates) / len(false_positive_rates),
            mean_metrics,
        )
    else:
        return all_loss / len(valid_dataloader), 0.0, 0.0, mean_metrics


def train_one_epoch_vbin(
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
        desc="Training",
    )

    all_loss = 0.0
    for batch_id, batch in enumerate(pbar):
        if device.type == "cuda":
            batch = cuda(batch, device=device)

        if args.task_type == "ablation-experiment-1":
            protein_node_logic = model(batch)
        else:
            protein_node_logic, _ = model(batch)

        targets = batch["targets"]
        if not is_valid_outputs(protein_node_logic, targets):
            print(
                "Warning! this batch is not valid, protein residues number {}, targets length: {}".format(
                    batch["protein_graph"].num_residues.sum(), targets.size(0)
                )
            )
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
                "\rEpoch %d/%d, batch %d/%d, global step: %d, lr: %.8f, loss %.4f"
                % (
                    epoch + 1,
                    args.num_epochs,
                    batch_id + 1,
                    len(train_dataloader),
                    report_global_step,
                    lr,
                    loss.item(),
                )
            )
            recorder.record(
                data={
                    "global step": [report_global_step],
                    "learning rate": [lr],
                    "training loss": [loss.item()],
                },
                mode="train",
            )

        scheduler.step()

        # if (report_global_step % args.evaluate_per_step == 0) and (comm.get_rank() == 0):
        if report_global_step % args.evaluate_per_step == 0:
            evaluate_loss, overlap_scores, false_positive_rates = evaluate_vbin(
                args,
                model=model,
                valid_set=valid_set,
                device=device,
                loss_fn=loss_fn,
                evaluate_num_samples=args.evaluate_num_samples,
            )
            print(
                "\n\rEvaluation: Epoch %d/%d, batch %d/%d, global step: %d, evaluation loss %.4f, overlap score  %.4f, false positive rate  %.4f\n"
                % (
                    epoch + 1,
                    args.num_epochs,
                    batch_id + 1,
                    len(train_dataloader),
                    report_global_step,
                    evaluate_loss,
                    overlap_scores,
                    false_positive_rates,
                )
            )

            eval_results = {
                "global step": [report_global_step],
                "evaluation loss": [evaluate_loss],
                "overlap score": [overlap_scores],
                "false positive rate": [false_positive_rates],
            }
            if comm.get_rank() == 0:

                recorder.record(data=eval_results, mode="valid", criteria=args.criteria)

                eval_results = pd.DataFrame(eval_results)
                eval_model_save_path = os.path.join(
                    model_save_path, f"global_step_{report_global_step}"
                )

                save_model(
                    model.state_dict(),
                    args.__dict__,
                    eval_results,
                    eval_model_save_path,
                )

                del_model_ckpt_list = [
                    os.path.join(model_save_path, f"global_step_{step}")
                    for step in recorder.get_del_ckpt()
                ]
                for ckpt_path in del_model_ckpt_list:
                    delete_ckpt(ckpt_path)

    print(
        "\nEpoch %d/%d, loss %.4f"
        % (epoch + 1, args.num_epochs, all_loss / len(train_dataloader))
    )

    return global_step


def train_one_epoch_vmulti(
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
    num_site_types,
):

    model.train()
    pbar = tqdm(
        train_dataloader,
        total=len(train_dataloader),
        desc="Training",
    )

    all_loss = 0.0

    for batch_id, batch in enumerate(pbar):

        if device.type == "cuda":
            batch = cuda(batch, device=device)

        protein_node_logic, _ = model(batch)

        targets = batch["targets"]
        if not is_valid_outputs(protein_node_logic, targets):
            print(
                "Warning! this batch is not valid, protein residues number {}, targets length: {}".format(
                    batch["protein_graph"].num_residues.sum(), targets.size(0)
                )
            )
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
                "\rEpoch %d/%d, batch %d/%d, global step: %d, lr: %.8f, loss %.4f"
                % (
                    epoch + 1,
                    args.num_epochs,
                    batch_id + 1,
                    len(train_dataloader),
                    report_global_step,
                    lr,
                    loss.item(),
                )
            )
            print(
                "\rEpoch %d/%d, batch %d/%d, global step: %d, lr: %.8f, loss %.4f"
                % (
                    epoch + 1,
                    args.num_epochs,
                    batch_id + 1,
                    len(train_dataloader),
                    report_global_step,
                    lr,
                    loss.item(),
                )
            )
            recorder.record(
                data={
                    "global step": [report_global_step],
                    "learning rate": [lr],
                    "training loss": [loss.item()],
                },
                mode="train",
            )

        scheduler.step()

        # if (report_global_step % args.evaluate_per_step == 0) and (comm.get_rank() == 0):
        if report_global_step % args.evaluate_per_step == 0:
            evaluate_loss, overlap_scores, false_positive_rates, metrics = (
                evaluate_vmulti(
                    args,
                    model=model,
                    valid_set=valid_set,
                    device=device,
                    loss_fn=loss_fn,
                    evaluate_num_samples=args.evaluate_num_samples,
                    num_site_types=num_site_types,
                )
            )
            print(
                "\n\rEvaluation: Epoch %d/%d, batch %d/%d, global step: %d, evaluation loss %.4f, overlap score  %.4f, false positive rate  %.4f\n"
                % (
                    epoch + 1,
                    args.num_epochs,
                    batch_id + 1,
                    len(train_dataloader),
                    report_global_step,
                    evaluate_loss,
                    overlap_scores,
                    false_positive_rates,
                )
            )

            eval_results = {
                "global step": [report_global_step],
                "evaluation loss": [evaluate_loss],
                "overlap score": [overlap_scores],
                "false positive rate": [false_positive_rates],
            }
            eval_results.update(metrics)

            if comm.get_rank() == 0:

                recorder.record(data=eval_results, mode="valid", criteria=args.criteria)

                eval_results = pd.DataFrame(eval_results)
                eval_model_save_path = os.path.join(
                    model_save_path, f"global_step_{report_global_step}"
                )
                args_dict = args.__dict__
                args_dict["num_site_types"] = num_site_types
                save_model(
                    model.state_dict(), args_dict, eval_results, eval_model_save_path
                )

                del_model_ckpt_list = [
                    os.path.join(model_save_path, f"global_step_{step}")
                    for step in recorder.get_del_ckpt()
                ]
                for ckpt_path in del_model_ckpt_list:
                    delete_ckpt(ckpt_path)

    print(
        "\nEpoch %d/%d, loss %.4f"
        % (epoch + 1, args.num_epochs, all_loss / len(train_dataloader))
    )

    return global_step


def main(args):

    print(yaml.dump(args.__dict__))

    world_size = comm.get_world_size()
    gpus = args.gpus

    # if world_size > 1 and not dist.is_initialized():
    if not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")
    local_rank = comm.get_rank()
    device_id = gpus[local_rank]
    device = torch.device("cuda", device_id)

    debug = args.debug

    if args.task_type == "active-site-categorie-prediction":

        dataset = EnzymeReactionSiteTypeSaProtDataset(
            path=args.dataset_path,
            save_precessed=False,
            debug=debug,
            verbose=1,
            lazy=True,
            nb_workers=args.propcess_core,
            foldseek_bin_path=foldseek_bin_path,
        )
    else:
        dataset = EnzymeReactionSaProtDataset(
            path=args.dataset_path,
            save_precessed=False,
            debug=debug,
            verbose=1,
            lazy=True,
            nb_workers=args.propcess_core,
            foldseek_bin_path=foldseek_bin_path,
        )
    train_set, valid_set, _ = dataset.split()

    train_sampler = torch_data.DistributedSampler(train_set)

    train_dataloader = torch_data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=enzyme_rxn_saprot_collate_extract,
        num_workers=args.num_workers,
    )

    if args.task_type == "active-site-position-prediction":
        model = EnzymeActiveSiteModel(
            rxn_model_path=args.pretrained_rxn_attn_model_path, use_saprot_esm=True
        )
    elif args.task_type == "active-site-categorie-prediction":
        model = EnzymeActiveSiteClsModel(
            rxn_model_path=args.pretrained_rxn_attn_model_path,
            num_active_site_type=dataset.num_active_site_type,
            use_saprot_esm=True,
        )
    elif args.task_type == "ablation-experiment-1":
        model = EnzymeActiveSiteESMGearNetModel(
            bridge_hidden_dim=args.bridge_hidden_dim
        )

    elif args.task_type == "ablation-experiment-2":
        model = EnzymeActiveSiteModel(
            rxn_model_path=args.pretrained_rxn_attn_model_path,
            from_scratch=True,
            use_saprot_esm=True,
        )  # 这里传递预训练的模型只是为了初始化模型的形状，但是模型state并不继承
    else:
        raise ValueError("Task erro")

    if args.train_from_checkpoint != "":
        model_state, _ = read_model_state(model_save_path=args.train_from_checkpoint)
        model = load_pretrain_model_state(model, model_state)
        print("Loaded checkpoint from {}".format(args.train_from_checkpoint))

    model.to(device)

    dataset_name = os.path.split(args.dataset_path)[-1]
    model_save_path = os.path.join(
        args.checkpoint_path,
        "train_in_{}_at_{}".format(
            dataset_name, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        ),
    )

    if debug:
        model_save_path = os.path.join(args.checkpoint_path, "debug")

    model = DDP(model, find_unused_parameters=True, device_ids=[device_id])

    if args.task_type == "ablation-experiment-2":
        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        if args.task_type != "ablation-experiment-1":
            pretrained_model_lr_ratio = {
                "enzyme_attn_model.model.sequence_model": args.protein_sequnece_model_lr_ratio,
                "rxn_attn_model": args.reaction_attention_model_lr_ratio,
            }
        else:
            pretrained_model_lr_ratio = {
                "enzyme_attn_model.model.sequence_model": args.protein_sequnece_model_lr_ratio,
            }

        params_groups = []

        for i, named_param in enumerate(model.named_parameters()):
            param_dict = {}
            name, param = named_param
            param_dict["params"] = param
            param_dict["lr"] = args.lr
            for pretrain_name in pretrained_model_lr_ratio:
                if pretrain_name in name:
                    param_dict["lr"] = (
                        args.lr * pretrained_model_lr_ratio[pretrain_name]
                    )
            param_dict["params_name"] = name
            params_groups.append(param_dict)

        optimizer = AdamW(params_groups, lr=args.lr, weight_decay=args.weight_decay)

    total_step = len(train_dataloader) * args.num_epochs
    warmup_steps = math.ceil(total_step * args.warmup_raio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_step,
    )

    if args.task_type == "active-site-categorie-prediction":
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 1, 1, 1]).to(device))
    else:
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 1]).to(device))

    recoder = Recorder(model_save_path, keep_num_ckpt=args.keep_num_ckpt)

    global_step = 0
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        if args.task_type == "active-site-categorie-prediction":
            global_step = train_one_epoch_vmulti(
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
                model_save_path=model_save_path,
                num_site_types=dataset.num_active_site_type,
            )
        else:
            global_step = train_one_epoch_vbin(
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
                model_save_path=model_save_path,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Training arguements")
    parser.add_argument(
        "--gpus", type=int, nargs="+", default=[0], help="CUDA devices id to use"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate of optimizer"
    )
    parser.add_argument(
        "--task_type",
        choices=[
            "active-site-position-prediction",
            "active-site-categorie-prediction",
            "ablation-experiment-1",  # 消融实验1： 研究反应分支及酶-反应相互作用网络的作用
            "ablation-experiment-2",  # 消融实验2： 研究反应分支预训练的作用
        ],
        default="active-site-position-prediction",
        help="Choose a task",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight Decay")
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size of dataloader"
    )
    parser.add_argument(
        "--bridge_hidden_dim", type=int, default=128, help="Bridge layer hidden size"
    )  # 搭配消融实验1
    parser.add_argument(
        "--keep_num_ckpt", type=int, default=10, help="Keep checkpoint numbers"
    )
    parser.add_argument(
        "--warmup_raio", type=float, default=0.01, help="Warmup Step Raio"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100",
        help="EC Site dataset path",
    )
    parser.add_argument(
        "--pretrained_rxn_attn_model_path",
        type=str,
        default="checkpoints/reaction_attn_net/model-ReactionMGMTurnNet_train_in_uspto_at_2023-04-05-23-46-25",
        help="Pretrained reaction attention model path",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/enzyme_site_type_predition_model/",
        help="Pretrained reaction attention model path",
    )
    parser.add_argument(
        "--train_from_checkpoint", type=str, default="", help="Train from checkpoint"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of processes for data loader"
    )
    parser.add_argument(
        "--propcess_core",
        type=int,
        default=16,
        help="Number of processes for data preprocess",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Maximum number of epochs for training",
    )
    parser.add_argument("--max_clip", type=int, default=20)
    parser.add_argument("--gradient_interval", type=int, default=1)
    parser.add_argument(
        "--evaluate_per_step",
        type=int,
        default=10000,
        help="evaluate pre step at training",
    )
    parser.add_argument(
        "--evaluate_num_samples",
        type=int,
        default=1000,
    )
    parser.add_argument("--protein_sequnece_model_lr_ratio", type=float, default=0.1)
    parser.add_argument("--reaction_attention_model_lr_ratio", type=float, default=0.1)
    parser.add_argument(
        "--criteria",
        type=str,
        default="overlap score",
    )

    args = parser.parse_args()

    args.debug = False

    main(args)
