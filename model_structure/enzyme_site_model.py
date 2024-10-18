import os
import sys

import pkg_resources


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from common.utils import read_model_state, load_pretrain_model_state

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel
from model_structure.enzyme_attn_model import (
    EnzymeFusionNetworkWrapper as EnzymeAttnNetwork,
    EnzymeSaProtFusionNetworkWrapper as EnzymeAttnSaProtNetwork,
    EnzymeESMWrapper as EnzymeESMNetwork,
)

from model_structure.rxn_attn_model import (
    ReactionMGMTurnNet as RXNAttnNetwork,
    GlobalMultiHeadAttention,
    GELU,
)


def pack_residue_feats(packedgraph, residue_feats):
    num_residues = packedgraph.num_residues.tolist()
    edit_feats = torch.split(residue_feats, num_residues)
    masks = [
        torch.ones(num_residue, dtype=torch.uint8, device=residue_feats.device)
        for num_residue in num_residues
    ]
    padded_feats = pad_sequence(edit_feats, batch_first=True, padding_value=0)
    masks = pad_sequence(masks, batch_first=True, padding_value=0)
    return padded_feats, masks


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim * 2, out_dim),
        )
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        return self.net(x)


class EnzymeActiveSiteModel(nn.Module):
    def __init__(
        self, rxn_model_path, from_scratch=False, use_saprot_esm=False
    ) -> None:
        super(EnzymeActiveSiteModel, self).__init__()
        self.from_scratch = from_scratch
        if not use_saprot_esm:
            self.enzyme_attn_model = EnzymeAttnNetwork(
                use_graph_construction_model=True
            )
        else:
            self.enzyme_attn_model = EnzymeAttnSaProtNetwork(
                use_graph_construction_model=True
            )

        self.rxn_attn_model = self._load_rxn_attn_model(rxn_model_path)

        self.brige_model = FeedForward(
            self.enzyme_attn_model.output_dim, self.rxn_attn_model.node_out_dim
        )

        self.interaction_net = GlobalMultiHeadAttention(
            self.rxn_attn_model.node_out_dim,
            heads=8,
            n_layers=3,
            cross_attn_h_rate=1,
            dropout=0.1,
            positional_number=0,
        )

        self.active_net = FeedForward(self.rxn_attn_model.node_out_dim, 2)

    def forward(self, batch):
        output_protein = self.enzyme_attn_model(batch)
        output_reaction = self.rxn_attn_model(
            **batch["reaction_graph"], return_rts=True
        )

        substrate_node_feature, substrate_mask = output_reaction

        protein_node_feature = self.brige_model(output_protein["node_feature"])
        # 这里要注意protein的node feature需要变形  使用batch_data['protein_graph'].num_residues

        protein_node_feature, protein_mask = pack_residue_feats(
            batch["protein_graph"], protein_node_feature
        )

        protein_node_feature, _, _, _, _ = self.interaction_net(
            src=protein_node_feature,
            tgt=substrate_node_feature,
            src_mask=protein_mask,
            tgt_mask=substrate_mask,
        )

        protein_node_feature = protein_node_feature[protein_mask.bool()]

        out = self.active_net(protein_node_feature)

        return out, protein_mask

    def _load_rxn_attn_model(self, model_state_path):
        model_state, rxn_attn_args = read_model_state(model_save_path=model_state_path)

        rxn_attn_model = RXNAttnNetwork(
            node_in_feats=rxn_attn_args["node_in_feats"],
            node_out_feats=rxn_attn_args["node_out_feats"],
            edge_in_feats=rxn_attn_args["edge_in_feats"],
            edge_hidden_feats=rxn_attn_args["edge_hidden_feats"],
            num_step_message_passing=rxn_attn_args["num_step_message_passing"],
            attention_heads=rxn_attn_args["attention_heads"],
            attention_layers=rxn_attn_args["attention_layers"],
            dropout=rxn_attn_args["dropout"],
            num_atom_types=rxn_attn_args["num_atom_types"],
            cross_attn_h_rate=rxn_attn_args["cross_attn_h_rate"],
            is_pretrain=False,
        )
        if not self.from_scratch:
            print(
                f"Loading reaction attention model checkpoint from {model_state_path}..."
            )
            rxn_attn_model = load_pretrain_model_state(rxn_attn_model, model_state)
        else:
            # print("Reaction attention model from scratch...")
            pass
        return rxn_attn_model


class EnzymeActiveSiteClsModel(nn.Module):
    def __init__(
        self,
        rxn_model_path,
        num_active_site_type,
        from_scratch=False,
        use_saprot_esm=False,
    ) -> None:
        super(EnzymeActiveSiteClsModel, self).__init__()
        self.from_scratch = from_scratch
        self.num_active_site_type = num_active_site_type

        if not use_saprot_esm:
            self.enzyme_attn_model = EnzymeAttnNetwork(
                use_graph_construction_model=True
            )
        else:
            self.enzyme_attn_model = EnzymeAttnSaProtNetwork(
                use_graph_construction_model=True
            )

        self.rxn_attn_model = self._load_rxn_attn_model(rxn_model_path)

        self.brige_model = FeedForward(
            self.enzyme_attn_model.output_dim, self.rxn_attn_model.node_out_dim
        )

        self.interaction_net = GlobalMultiHeadAttention(
            self.rxn_attn_model.node_out_dim,
            heads=8,
            n_layers=3,
            cross_attn_h_rate=1,
            dropout=0.1,
            positional_number=0,
        )

        self.active_site_type_net = FeedForward(
            self.rxn_attn_model.node_out_dim, num_active_site_type
        )

    def forward(self, batch):
        output_protein = self.enzyme_attn_model(batch)
        output_reaction = self.rxn_attn_model(
            **batch["reaction_graph"], return_rts=True
        )

        substrate_node_feature, substrate_mask = output_reaction

        protein_node_feature = self.brige_model(output_protein["node_feature"])

        protein_node_feature, protein_mask = pack_residue_feats(
            batch["protein_graph"], protein_node_feature
        )

        protein_node_feature, _, _, _, _ = self.interaction_net(
            src=protein_node_feature,
            tgt=substrate_node_feature,
            src_mask=protein_mask,
            tgt_mask=substrate_mask,
        )

        protein_node_feature = protein_node_feature[protein_mask.bool()]

        out = self.active_site_type_net(protein_node_feature)

        return out, protein_mask

    def _load_rxn_attn_model(self, model_state_path):
        model_state, rxn_attn_args = read_model_state(model_save_path=model_state_path)

        rxn_attn_model = RXNAttnNetwork(
            node_in_feats=rxn_attn_args["node_in_feats"],
            node_out_feats=rxn_attn_args["node_out_feats"],
            edge_in_feats=rxn_attn_args["edge_in_feats"],
            edge_hidden_feats=rxn_attn_args["edge_hidden_feats"],
            num_step_message_passing=rxn_attn_args["num_step_message_passing"],
            attention_heads=rxn_attn_args["attention_heads"],
            attention_layers=rxn_attn_args["attention_layers"],
            dropout=rxn_attn_args["dropout"],
            num_atom_types=rxn_attn_args["num_atom_types"],
            cross_attn_h_rate=rxn_attn_args["cross_attn_h_rate"],
            is_pretrain=False,
        )
        if not self.from_scratch:
            print(
                f"Loading reaction attention model checkpoint from {model_state_path}..."
            )
            rxn_attn_model = load_pretrain_model_state(rxn_attn_model, model_state)
        else:
            pass
            # print("Reaction attention model from scratch...")
        return rxn_attn_model


class EnzymeActiveSiteESMGearNetModel(nn.Module):
    def __init__(self, bridge_hidden_dim=128) -> None:
        super(EnzymeActiveSiteESMGearNetModel, self).__init__()

        self.enzyme_attn_model = EnzymeAttnNetwork(use_graph_construction_model=True)

        self.brige_model = FeedForward(
            self.enzyme_attn_model.output_dim, bridge_hidden_dim
        )

        self.active_net = FeedForward(bridge_hidden_dim, 2)

    def forward(self, batch):

        output_protein = self.enzyme_attn_model(batch)
        protein_node_feature = self.brige_model(output_protein["node_feature"])
        out = self.active_net(protein_node_feature)

        return out


class EnzymeActiveSiteESMModel(EnzymeActiveSiteModel):
    def __init__(self, rxn_model_path, from_scratch=False) -> None:
        super(EnzymeActiveSiteModel, self).__init__()
        self.from_scratch = from_scratch

        self.enzyme_attn_model = EnzymeESMNetwork(use_graph_construction_model=False)

        self.rxn_attn_model = self._load_rxn_attn_model(rxn_model_path)

        self.brige_model = FeedForward(
            self.enzyme_attn_model.output_dim, self.rxn_attn_model.node_out_dim
        )

        self.interaction_net = GlobalMultiHeadAttention(
            self.rxn_attn_model.node_out_dim,
            heads=8,
            n_layers=3,
            cross_attn_h_rate=1,
            dropout=0.1,
            positional_number=0,
        )

        self.active_net = FeedForward(self.rxn_attn_model.node_out_dim, 2)


class EnzymeActiveSiteRXNFPModel(nn.Module):
    def __init__(self, rxnfp_model_name="bert_ft") -> None:
        super(EnzymeActiveSiteRXNFPModel, self).__init__()
        self.enzyme_attn_model = EnzymeAttnNetwork(use_graph_construction_model=True)
        rxnfp_model_path = pkg_resources.resource_filename(
            "rxnfp", f"models/transformers/{rxnfp_model_name}"
        )

        self.rxn_attn_model = BertModel.from_pretrained(rxnfp_model_path)
        self.rxnfp_max_length = self.rxn_attn_model.config.max_position_embeddings

        self.brige_model = FeedForward(
            self.enzyme_attn_model.output_dim, self.rxn_attn_model.config.hidden_size
        )

        self.interaction_net = GlobalMultiHeadAttention(
            self.rxn_attn_model.config.hidden_size,
            heads=8,
            n_layers=3,
            cross_attn_h_rate=1,
            dropout=0.1,
            positional_number=0,
        )
        self.active_net = FeedForward(self.rxn_attn_model.config.hidden_size, 2)

    def forward(self, batch):
        output_protein = self.enzyme_attn_model(batch)
        output_reaction = self.rxn_attn_model(**batch["rxn_bert_inputs"])

        rxn_feature = output_reaction["last_hidden_state"]
        rxn_mask = batch["rxn_bert_inputs"]["attention_mask"]

        protein_node_feature = self.brige_model(output_protein["node_feature"])

        protein_node_feature, protein_mask = pack_residue_feats(
            batch["protein_graph"], protein_node_feature
        )

        protein_node_feature, _, _, _, _ = self.interaction_net(
            src=protein_node_feature,
            tgt=rxn_feature,
            src_mask=protein_mask,
            tgt_mask=rxn_mask,
        )

        protein_node_feature = protein_node_feature[protein_mask.bool()]

        out = self.active_net(protein_node_feature)
        return out, protein_mask
        pass


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils import data as torch_data
    from data_loaders.enzyme_rxn_dataloader import (
        EnzymeReactionDataset,
        enzyme_rxn_collate_extract,
        EnzymeReactionSiteTypeDataset,
        EnzymeReactionSiteTypeSaProtDataset,
        EnzymeRxnSaprotCollate,
        EnzymeReactionSaProtDataset,
        EnzymeReactionRXNFPDataset,
        EnzymeRxnfpCollate,
    )

    from data_loaders.enzyme_dataloader import EnzymeDataset
    from common.utils import cuda

    # import sys
    # sys.path.append('..')
    # from train_activate_site_model import is_valid_outputs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = EnzymeActiveSiteModel(rxn_model_path='../checkpoints/reaction_attn_net/model-ReactionMGMTurnNet_train_in_uspto_at_2023-04-05-23-46-25/checkpoint_epoch_72-global_step_1765001')

    # model.to(device)

    # batchs = []

    # save_erro_batch = os.path.join('../test_pipline', 'erro_batch.pkl')
    # if os.path.exists(save_erro_batch):
    #     batchs = torch.load(save_erro_batch, map_location=torch.device('cpu'))
    # if batchs:
    #     for batch_data in batchs:
    #         if device.type == "cuda":
    #             batch_data = cuda(batch_data, device=device)
    #         out, protein_mask = model(batch_data)
    #         targets = batch_data['targets']
    # else:
    #     dataset = EnzymeReactionDataset(path='dataset/ec_site_dataset/uniprot_ecreact_merge_dataset_limit_10000', save_precessed=False, debug=True, verbose=1, lazy=True, nb_workers=12)

    #     train_set, valid_set, test_set = dataset.split()

    #     remove_unused_parameters(model, train_set=train_set, device=device)

    #     for i in tqdm(range(10)):
    #         train_set[i]
    #         valid_set[i]
    #         test_set[i]

    #     train_loader = torch_data.DataLoader(
    #         train_set,
    #         batch_size=1,
    #         collate_fn=enzyme_rxn_collate_extract,
    #         num_workers=6)

    #     for batch_data in tqdm(train_loader, desc='train loader'):
    #         if device.type == "cuda":
    #             batch_data = cuda(batch_data, device=device)
    #         out, protein_mask = model(batch_data)
    #         targets = batch_data['targets']

    #         if not is_valid_outputs(out, targets):
    #             batchs.append(batch_data)

    #         if len(batchs) >= 10:
    #             torch.save(batchs, save_erro_batch)
    #             break

    # dataset = EnzymeReactionSiteTypeDataset(path='dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100', save_precessed=False, debug=False, verbose=1, lazy=True, nb_workers=12)

    # model = EnzymeActiveSiteClsModel(rxn_model_path='../checkpoints/reaction_attn_net/model-ReactionMGMTurnNet_train_in_uspto_at_2023-04-05-23-46-25/checkpoint_epoch_72-global_step_1765001', num_active_site_type=dataset.num_active_site_type)

    # dataset = EnzymeDataset(path='dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100', save_precessed=False, debug=False, verbose=1, lazy=True, nb_workers=12)

    # model = EnzymeActiveSiteESMModel()

    # model.to(device)
    # train_set, valid_set, test_set = dataset.split()

    # train_loader = torch_data.DataLoader(
    #     train_set,
    #     batch_size=1,
    #     collate_fn=enzyme_rxn_collate_extract,
    #     num_workers=6)

    # batchs = []
    # for batch_data in tqdm(train_loader, desc='train loader'):
    #     if device.type == "cuda":
    #         batch_data = cuda(batch_data, device=device)
    #     # out, protein_mask = model(batch_data)
    #     out = model(batch_data)
    #     targets = batch_data['targets']

    #     # if not is_valid_outputs(out, targets):
    #     #     batchs.append(batch_data)

    # dataset = EnzymeReactionSiteTypeSaProtDataset(
    #     path="dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100",
    #     save_precessed=False,
    #     debug=False,
    #     verbose=1,
    #     lazy=True,
    #     nb_workers=12,
    #     foldseek_bin_path="../foldseek_bin/foldseek",
    # )

    # dataset = EnzymeReactionSaProtDataset(
    #     path="dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100",
    #     save_precessed=False,
    #     debug=False,
    #     verbose=1,
    #     lazy=True,
    #     nb_workers=12,
    #     foldseek_bin_path="../foldseek_bin/foldseek",
    # )

    # model = EnzymeActiveSiteClsModel(
    #     rxn_model_path="../checkpoints/reaction_attn_net/model-ReactionMGMTurnNet_train_in_uspto_at_2023-04-05-23-46-25/checkpoint_epoch_72-global_step_1765001",
    #     num_active_site_type=dataset.num_active_site_type,
    #     use_saprot_esm=True,
    # )

    # model = EnzymeActiveSiteModel(
    #     rxn_model_path="../checkpoints/reaction_attn_net/model-ReactionMGMTurnNet_train_in_uspto_at_2023-04-05-23-46-25/checkpoint_epoch_72-global_step_1765001",
    #     use_saprot_esm=True,
    # )

    # model.to(device)
    # train_set, valid_set, test_set = dataset.split()
    # enzyme_rxn_saprot_collate_extract = EnzymeRxnSaprotCollate()
    # train_loader = torch_data.DataLoader(
    #     train_set,
    #     batch_size=1,
    #     collate_fn=enzyme_rxn_saprot_collate_extract,
    #     num_workers=6,
    # )

    # batchs = []
    # for batch_data in tqdm(train_loader, desc="train loader"):
    #     if device.type == "cuda":
    #         batch_data = cuda(batch_data, device=device)
    #     # out, protein_mask = model(batch_data)
    #     out, _ = model(batch_data)
    #     targets = batch_data["targets"]

    # if not is_valid_outputs(out, targets):
    #     batchs.append(batch_data)
    """
    消融实验测试区域
    """
    # dataset = EnzymeReactionDataset(path='dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100', save_precessed=False, debug=False, verbose=1, lazy=True, nb_workers=12)
    # train_set, valid_set, test_set = dataset.split()

    # model = EnzymeActiveSiteESMModel(
    #     rxn_model_path="../checkpoints/reaction_attn_net/model-ReactionMGMTurnNet_train_in_uspto_at_2023-04-05-23-46-25/checkpoint_epoch_72-global_step_1765001",
    # )

    # model.to(device)

    # train_loader = torch_data.DataLoader(
    #     train_set,
    #     batch_size=2,
    #     collate_fn=enzyme_rxn_collate_extract,
    #     num_workers=6)
    # for batch_data in tqdm(train_loader, desc="train loader"):
    #     if device.type == "cuda":
    #         batch_data = cuda(batch_data, device=device)
    #     # out, protein_mask = model(batch_data)
    #     out, _ = model(batch_data)
    #     targets = batch_data["targets"]

    """
    rxnfp模型测试区域
    """
    dataset = EnzymeReactionRXNFPDataset(
        path="dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100",
        save_precessed=False,
        debug=False,
        verbose=1,
        lazy=True,
        nb_workers=12,
    )
    model = EnzymeActiveSiteRXNFPModel()
    model.to(device)

    train_set, valid_set, test_set = dataset.split()
    # for i in tqdm(range(10)):
    #     train_set[i]
    #     valid_set[i]
    #     test_set[i]
    enzyme_rxnfp_collate_extract = EnzymeRxnfpCollate(model.rxnfp_max_length)
    train_loader = torch_data.DataLoader(
        train_set,
        batch_size=2,
        collate_fn=enzyme_rxnfp_collate_extract,
        # collate_fn=enzyme_rxn_collate_extract,
        num_workers=2,
    )
    for batch_data in tqdm(train_loader, desc="train loader"):
        if device.type == "cuda":
            batch_data = cuda(batch_data, device=device)

        model(batch_data)
