import argparse
import os
from tqdm.auto import tqdm
from collections import defaultdict
import torch
from torch.utils import data as torch_data
from common.utils import (
    calculate_scores_vbin_test,
    calculate_scores_vmulti_test,
    convert_fn,
    cuda,
    read_model_state,
)
from data_loaders.enzyme_rxn_dataloader import (
    AugEnzymeReactionDataset,
    EnzymeReactionDataset,
    EnzymeReactionSiteTypeDataset,
    enzyme_rxn_collate_extract,
    EnzymeRxnSaprotCollate,
    EnzymeReactionSaProtDataset,
)
from model_structure.enzyme_site_model import (
    EnzymeActiveSiteClsModel,
    EnzymeActiveSiteESMGearNetModel,
    EnzymeActiveSiteModel,
)
from main_train import is_valid_outputs


def main(args):
    device = (
        torch.device(f"cuda:{args.gpu}")
        if (args.gpu >= 0) and torch.cuda.is_available()
        else torch.device("cpu")
    )

    if args.use_saprot:
        enzyme_rxn_saprot_collate_extract = EnzymeRxnSaprotCollate()
        foldseek_bin_path = "foldseek_bin/foldseek"
        assert os.path.exists(
            foldseek_bin_path
        ), f"Need foldseek binary file, download it: https://drive.google.com/file/d/1B_9t3n_nlj8Y3Kpc_mMjtMdY0OPYa7Re/view, and put it at {foldseek_bin_path}"

        dataset = EnzymeReactionSaProtDataset(
            path=args.dataset_path,
            structure_path=args.structure_path,
            save_precessed=False,
            debug=False,
            verbose=1,
            protein_max_length=1000,
            lazy=True,
            nb_workers=args.propcess_core,
            foldseek_bin_path=foldseek_bin_path,
        )
        args.collate_fn = enzyme_rxn_saprot_collate_extract
    else:

        dataset = EnzymeReactionDataset(
            path=args.dataset_path,
            structure_path=args.structure_path,
            save_precessed=False,
            debug=False,
            verbose=1,
            protein_max_length=1000,
            lazy=True,
            nb_workers=args.propcess_core,
        )
        args.collate_fn = enzyme_rxn_collate_extract
    _, _, test_dataset = dataset.split()

    test_dataloader = torch_data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=args.collate_fn,
        shuffle=False,
        num_workers=args.propcess_core,
    )

    model = EnzymeActiveSiteModel(
        rxn_model_path=args.pretrained_rxn_attn_model_path,
        use_saprot_esm=args.use_saprot,
    )

    model_state, model_args = read_model_state(model_save_path=args.checkpoint)
    need_convert = model_args.get("need_convert", False)
    model.load_state_dict(model_state)
    print("Loaded checkpoint from {}".format(os.path.abspath(args.checkpoint)))
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
    mcc_scores_list = []

    pbar = tqdm(test_dataloader, total=len(test_dataloader), desc="Testing")
    with torch.no_grad():
        for batch_id, batch in enumerate(pbar):
            if device.type == "cuda":
                batch = cuda(batch, device=device)
            try:
                protein_node_logic, _ = model(batch)
            except:
                print(f"erro in batch: {batch_id}")
                continue

            targets = batch["targets"].long()
            if not is_valid_outputs(protein_node_logic, targets):
                print(f"erro in batch: {batch_id}")
                continue

            pred = torch.argmax(protein_node_logic.softmax(-1), dim=-1)
            if need_convert:
                pred = convert_fn(pred, to_list=False)
            correct = pred == targets
            node_correct_cnt += correct.sum().item()
            node_cnt += targets.size(0)

            (
                accuracy,
                precision,
                specificity,
                overlap_score,
                fpr,
                f1_scores,
                mcc_scores,
            ) = calculate_scores_vbin_test(
                pred, targets, batch["protein_graph"].num_residues.tolist()
            )
            accuracy_list += accuracy
            precision_list += precision
            specificity_list += specificity
            overlap_scores_list += overlap_score
            false_positive_rates_list += fpr
            f1_scores_list += f1_scores
            mcc_scores_list += mcc_scores

            pbar.set_description(
                "Accuracy: {:.4f}, Precision: {:.4f}, Specificity: {:.4f}, Overlap Score: {:.4f}, False Positive Rate: {:.4f}, F1: {:.4f}, MCC: {:.4f}".format(
                    sum(accuracy_list) / len(accuracy_list),
                    sum(precision_list) / len(precision_list),
                    sum(specificity_list) / len(specificity_list),
                    sum(overlap_scores_list) / len(overlap_scores_list),
                    sum(false_positive_rates_list) / len(false_positive_rates_list),
                    sum(f1_scores_list) / len(f1_scores_list),
                    sum(mcc_scores_list) / len(mcc_scores_list),
                )
            )

    print(f"Get {len(overlap_scores_list)} results")
    print(
        "Accuracy: {:.4f}, Precision: {:.4f}, Specificity: {:.4f}, Overlap Score: {:.4f}, False Positive Rate: {:.4f}, F1: {:.4f}, MCC: {:.4f}".format(
            sum(accuracy_list) / len(accuracy_list),
            sum(precision_list) / len(precision_list),
            sum(specificity_list) / len(specificity_list),
            sum(overlap_scores_list) / len(overlap_scores_list),
            sum(false_positive_rates_list) / len(false_positive_rates_list),
            sum(f1_scores_list) / len(f1_scores_list),
            sum(mcc_scores_list) / len(mcc_scores_list),
        )
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Test arguements")
    parser.add_argument("--gpu", type=int, default=-1, help="CUDA devices id to use")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/mcsa_fine_tune/normal_mcsa",
        help="Test dataset path",
    )
    parser.add_argument(
        "--pretrained_rxn_attn_model_path",
        type=str,
        default="checkpoints/reaction_attn_net/model-ReactionMGMTurnNet_train_in_uspto_at_2023-04-05-23-46-25",
        help="Pretrained reaction representation branch",
    )
    parser.add_argument(
        "--structure_path", type=str, default="dataset/mcsa_fine_tune/structures"
    )
    # parser.add_argument('--use_aug',
    #                     action='store_true')
    # parser.add_argument('--soft_check',
    #                     action='store_true')
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,  # 推荐batch size=1
        help="Batch size of dataloader",
    )

    parser.add_argument(
        "--bridge_hidden_dim", type=int, default=128, help="Bridge layer hidden size"
    )  # 搭配消融实验1
    parser.add_argument(
        "--propcess_core",
        type=int,
        default=16,
        help="Number of processes for data preprocess",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/enzyme_site_predition_model/train_in_uniprot_ecreact_merge_dataset_limit_100_at_2023-05-25-20-39-05/global_step_44000",
        help="Pretrained reaction attention model path",
    )
    parser.add_argument(
        "--use_saprot",
        action="store_true",
    )

    parser.add_argument("--test_remove_aegan_train", type=bool, default=False)

    args = parser.parse_args()
    main(args)
