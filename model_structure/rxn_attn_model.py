from collections import namedtuple
import math
import torch
import torch.nn as nn
from dgllife.model import MPNNGNN
import dgl
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def pack_atom_feats(bg, atom_feats):
    bg.ndata['h'] = atom_feats
    gs = dgl.unbatch(bg)
    edit_feats = [g.ndata['h'] for g in gs]
    masks = [
        torch.ones(g.num_nodes(), dtype=torch.uint8, device=bg.device)
        for g in gs
    ]
    padded_feats = pad_sequence(edit_feats, batch_first=True, padding_value=0)
    masks = pad_sequence(masks, batch_first=True, padding_value=0)
    return padded_feats, masks


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, positional_number=5, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.p_k = positional_number
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        if self.p_k != 0:
            self.relative_k = nn.Parameter(torch.randn(self.p_k, self.d_k))
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),
                                      nn.Dropout(dropout),
                                      nn.Linear(d_model, d_model))
        self.gating = nn.Linear(d_model, d_model)
        self.to_out = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)

    def one_hot_embedding(self, labels):
        y = torch.eye(self.p_k, device=labels.device)
        return y[labels]

    def forward(self, src, tgt, gpm, src_mask=None, tgt_mask=None):
        bs, atom_size = src.size(0), src.size(1)
        src = self.layer_norm(src)
        tgt = self.layer_norm(tgt)
        k = self.k_linear(src)
        q = self.q_linear(tgt)
        v = self.v_linear(tgt)
        k1 = k.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        q1 = q.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v1 = v.view(bs, -1, self.h, self.d_k).transpose(1, 2)
        attn1 = torch.matmul(q1, k1.permute(0, 1, 3, 2))

        if (self.p_k == 0) or (gpm is None):
            attn = attn1 / math.sqrt(self.d_k)
        elif gpm is not None:
            gpms = self.one_hot_embedding(
                gpm.unsqueeze(1).repeat(1, self.h, 1, 1))
            attn2 = torch.matmul(q1, self.relative_k.transpose(0, 1))
            attn2 = torch.matmul(gpms, attn2.unsqueeze(-1)).squeeze(-1)
            attn = (attn1 + attn2) / math.sqrt(self.d_k)

        if (src_mask is not None) and (tgt_mask is None):
            src_mask = src_mask.bool()
            src_mask = src_mask.unsqueeze(1).repeat(1, src_mask.size(-1), 1)
            src_mask = src_mask.unsqueeze(1).repeat(1, attn.size(1), 1, 1)
            attn_mask = src_mask
            attn[~attn_mask] = float(-9e9)
        elif (src_mask is not None) and (tgt_mask is not None):
            src_mask = src_mask.bool()
            tgt_mask = tgt_mask.bool()
            attn_mask = tgt_mask.unsqueeze(1).repeat(1, src_mask.size(-1), 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, attn.size(1), 1, 1)
            attn = attn.permute(0, 1, 3, 2)
            attn[~attn_mask] = float(-9e9)

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout1(attn)
        v1 = v.view(bs, -1, self.h, self.d_k).permute(0, 2, 1, 3)
        output = torch.matmul(attn, v1)

        output = output.transpose(1, 2).contiguous().view(
            bs, -1, self.d_model).squeeze(-1)
        # gate self attention
        output = self.to_out(output * self.gating(src).sigmoid())
        return self.dropout2(output), attn, attn_mask


#         return output, attn


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model * 2), GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_model * 2, d_model))
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        return self.net(x)


class Global_Self_Attention(nn.Module):
    def __init__(self,
                 d_model,
                 heads=8,
                 n_layers=3,
                 positional_number=5,
                 dropout=0.1):
        super(Global_Self_Attention, self).__init__()
        self.n_layers = n_layers
        att_stack = []
        pff_stack = []
        for _ in range(n_layers):
            att_stack.append(
                MultiHeadAttention(heads, d_model, positional_number, dropout))
            pff_stack.append(FeedForward(d_model, dropout=dropout))
        self.att_stack = nn.ModuleList(att_stack)
        self.pff_stack = nn.ModuleList(pff_stack)

    def forward(self, x, rpm, mask=None):
        att_scores = {}
        for n in range(self.n_layers):
            m, att_score, attn_mask = self.att_stack[n](x, x, rpm, mask)
            x = x + self.pff_stack[n](x + m)
            att_scores[n] = att_score
        return x, att_scores, attn_mask


class Global_Cross_Attention(nn.Module):
    def __init__(self,
                 d_model,
                 heads=8,
                 n_layers=3,
                 positional_number=5,
                 dropout=0.1):
        super(Global_Cross_Attention, self).__init__()
        self.n_layers = n_layers
        att_stack = []
        pff_stack = []
        for _ in range(n_layers):
            att_stack.append(
                MultiHeadAttention(heads, d_model, positional_number, dropout))
            pff_stack.append(FeedForward(d_model, dropout=dropout))
        self.att_stack = nn.ModuleList(att_stack)
        self.pff_stack = nn.ModuleList(pff_stack)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        att_scores = {}
        for n in range(self.n_layers):
            m, att_score, attn_mask = self.att_stack[n](src, tgt, None,
                                                        src_mask, tgt_mask)
            src = src + self.pff_stack[n](src + m)
            att_scores[n] = att_score
        return src, att_scores, attn_mask


# TODO: self attention 和 cross attention 交叉进行
class GlobalMultiHeadAttentionLayer(nn.Module):
    def __init__(self,
                 d_model,
                 heads=8,
                 positional_number=5,
                 cross_attn_h_rate=1,
                 dropout=0.1,):
        super(GlobalMultiHeadAttentionLayer, self).__init__()

        self.cross_attn_h_rate = cross_attn_h_rate

        self.self_attn = MultiHeadAttention(heads, d_model, positional_number,
                                            dropout)
        self.cross_attn = MultiHeadAttention(heads, d_model, 0,
                                             dropout)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = GELU()

    def forward(self, src, tgt, rpm=None, src_mask=None, tgt_mask=None):
        src_m_s, att_score_s, attn_mask_s = self._sa_block(
            self.norm1(src), rpm, src_mask)
        src = src + src_m_s

        src_m_c, att_score_c, attn_mask_c = self._cra_block(
            self.norm2(src), tgt, src_mask, tgt_mask)
        src = src * self.cross_attn_h_rate + src_m_c  # self.cross_attn_h_rate 控制交叉注意力机制自身的比例
        src = src + self._ff_block(self.norm3(src))

        return src, att_score_s, attn_mask_s, att_score_c, attn_mask_c

    def _sa_block(self, x, rpm, mask):
        m, att_score, attn_mask = self.self_attn(x, x, rpm, mask)
        return self.dropout1(m), att_score, attn_mask

    def _cra_block(self, src, tgt, src_mask, tgt_mask):
        src_m, att_score, attn_mask = self.cross_attn(src, tgt, None, src_mask,
                                                      tgt_mask)
        return self.dropout2(src_m), att_score, attn_mask

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class GlobalMultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model,
                 heads=8,
                 n_layers=3,
                 positional_number=5,
                 cross_attn_h_rate=0.1,
                 dropout=0.1):
        super(GlobalMultiHeadAttention, self).__init__()
        self.n_layers = n_layers

        layer_stack = []

        for _ in range(n_layers):
            layer_stack.append(
                GlobalMultiHeadAttentionLayer(
                    d_model=d_model,
                    heads=heads,
                    positional_number=positional_number,
                    cross_attn_h_rate=cross_attn_h_rate,
                    dropout=dropout))

        self.layers = nn.ModuleList(layer_stack)

    def forward(self, src, tgt, rpm=None, src_mask=None, tgt_mask=None):
        self_att_scores = {}
        cross_att_scores = {}

        for n in range(self.n_layers):
            src, att_score_s, attn_mask_s, att_score_c, attn_mask_c = self.layers[
                n](src, tgt, rpm, src_mask, tgt_mask)
            self_att_scores[n] = att_score_s
            cross_att_scores[n] = att_score_c
        self_attn_mask = attn_mask_s
        cross_attn_mask = attn_mask_c
        return src, self_att_scores, self_attn_mask, cross_att_scores, cross_attn_mask


class ReactionMGMNet(nn.Module):
    def __init__(self,
                 node_in_feats,
                 node_out_feats,
                 edge_in_feats,
                 edge_hidden_feats,
                 num_step_message_passing,
                 attention_heads,
                 attention_layers,
                 dropout,
                 num_atom_types,
                 output_attention=False) -> None:
        super(ReactionMGMNet, self).__init__()
        self.output_attention = output_attention
        self.activation = GELU()

        self.mpnn = MPNNGNN(node_in_feats=node_in_feats,
                            node_out_feats=node_out_feats,
                            edge_in_feats=edge_in_feats,
                            edge_hidden_feats=edge_hidden_feats,
                            num_step_message_passing=num_step_message_passing)

        self.atom_self_att = Global_Self_Attention(node_out_feats,
                                                   attention_heads,
                                                   attention_layers, 8)

        self.atom_cross_att = Global_Cross_Attention(node_out_feats,
                                                     attention_heads,
                                                     attention_layers, 8)

        self.ffn1 = FeedForward(d_model=node_out_feats, dropout=dropout)

        self.ffn2 = FeedForward(d_model=node_out_feats, dropout=dropout)

        self.logic_atom_net = nn.Sequential(
            nn.Linear(node_out_feats, node_out_feats * 2), GELU(),
            nn.Dropout(dropout), nn.Linear(node_out_feats * 2, num_atom_types))

    def calculate_atom_self_attn_func(self, feat, adms, mask):
        feat_h, attn_scores, attn_mask = self.atom_self_att(feat, adms, mask)
        feat_h = self.ffn1(feat_h + feat)
        return feat_h, attn_scores, attn_mask

    def calcualte_atom_cross_attn_func(self, src_feat, tgt_feat, src_mask,
                                       tgt_mask):
        src_feat_h, src_cross_attn_score, attn_mask = self.atom_cross_att(
            src_feat, tgt_feat, src_mask, tgt_mask)
        src_feat_h = self.ffn2(src_feat_h + src_feat)
        return src_feat_h, src_cross_attn_score, attn_mask

    def embedded_net(
        self,
        rts_bg,
        rts_adms,
        rts_node_feats,
        rts_edge_feats,
        pds_bg,
        pds_adms,
        pds_node_feats,
        pds_edge_feats,
    ):

        rts_atom_feats = self.mpnn(rts_bg, rts_node_feats, rts_edge_feats)
        rts_atom_feats, rts_mask = pack_atom_feats(rts_bg, rts_atom_feats)
        rts_atom_feats_h, rts_atom_attention, _ = self.calculate_atom_self_attn_func(
            rts_atom_feats, rts_adms, rts_mask)

        pds_atom_feats = self.mpnn(pds_bg, pds_node_feats, pds_edge_feats)
        pds_atom_feats, pds_mask = pack_atom_feats(pds_bg, pds_atom_feats)
        pds_atom_feats_h, pds_atom_attention, _ = self.calculate_atom_self_attn_func(
            pds_atom_feats, pds_adms, pds_mask)

        return rts_atom_feats_h, rts_mask, rts_atom_attention, pds_atom_feats_h, pds_mask, pds_atom_attention

    def forward(
        self,
        rts_bg,
        rts_adms,
        rts_node_feats,
        rts_edge_feats,
        pds_bg,
        pds_adms,
        pds_node_feats,
        pds_edge_feats,
    ):

        rts_atom_feats_h, rts_mask, rts_atom_attention, pds_atom_feats_h, pds_mask, pds_atom_attention = self.embedded_net(
            rts_bg,
            rts_adms,
            rts_node_feats,
            rts_edge_feats,
            pds_bg,
            pds_adms,
            pds_node_feats,
            pds_edge_feats,
        )

        rts_atom_feats_h_cross_update, rts_atom_cross_attention, rts_atom_cross_attention_mask = self.calcualte_atom_cross_attn_func(
            rts_atom_feats_h, pds_atom_feats_h, rts_mask, pds_mask)
        pds_atom_feats_h_cross_update, pds_atom_cross_attention, pds_atom_cross_attention_mask = self.calcualte_atom_cross_attn_func(
            pds_atom_feats_h, rts_atom_feats_h, pds_mask, rts_mask)

        rts_atom_feats_h_cross_update = rts_atom_feats_h_cross_update[
            rts_mask.bool()]
        pds_atom_feats_h_cross_update = pds_atom_feats_h_cross_update[
            pds_mask.bool()]

        atom_feats_h_cross_update = torch.cat(
            [rts_atom_feats_h_cross_update, pds_atom_feats_h_cross_update],
            dim=0)
        atom_logic = self.logic_atom_net(atom_feats_h_cross_update)
        if self.output_attention:
            attentions_tuple = namedtuple('attentions', [
                'rts_atom_cross_attention', 'pds_atom_cross_attention',
                'rts_atom_attention', 'pds_atom_attention',
                'rts_atom_cross_attention_mask',
                'pds_atom_cross_attention_mask'
            ])
            attentions = attentions_tuple(
                rts_atom_cross_attention=rts_atom_cross_attention,
                pds_atom_cross_attention=pds_atom_cross_attention,
                rts_atom_attention=rts_atom_attention,
                pds_atom_attention=pds_atom_attention,
                rts_atom_cross_attention_mask=rts_atom_cross_attention_mask,
                pds_atom_cross_attention_mask=pds_atom_cross_attention_mask)

            return atom_logic, attentions
        else:
            return atom_logic, None


class ReactionMGMTurnNet(nn.Module):
    def __init__(self,
                 node_in_feats,
                 node_out_feats,
                 edge_in_feats,
                 edge_hidden_feats,
                 num_step_message_passing,
                 attention_heads,
                 attention_layers,
                 dropout,
                 num_atom_types,
                 cross_attn_h_rate,
                 output_attention=False,
                 is_pretrain=True) -> None:
        super(ReactionMGMTurnNet, self).__init__()
        self.output_attention = output_attention
        self.activation = GELU()

        self.mpnn = MPNNGNN(node_in_feats=node_in_feats,
                            node_out_feats=node_out_feats,
                            edge_in_feats=edge_in_feats,
                            edge_hidden_feats=edge_hidden_feats,
                            num_step_message_passing=num_step_message_passing)

        self.atom_attn_net = GlobalMultiHeadAttention(
            d_model=node_out_feats,
            heads=attention_heads,
            n_layers=attention_layers,
            positional_number=8,
            cross_attn_h_rate=cross_attn_h_rate,
            dropout=dropout)
        if is_pretrain:
            self.logic_atom_net = nn.Sequential(
                nn.Linear(node_out_feats, node_out_feats * 2), GELU(),
                nn.Dropout(dropout), nn.Linear(node_out_feats * 2, num_atom_types))
        self.node_out_dim = node_out_feats

    def embedded_net(
        self,
        rts_bg,
        rts_node_feats,
        rts_edge_feats,
        pds_bg,
        pds_node_feats,
        pds_edge_feats,
    ):

        rts_atom_feats = self.mpnn(rts_bg, rts_node_feats, rts_edge_feats)
        rts_atom_feats, rts_mask = pack_atom_feats(rts_bg, rts_atom_feats)

        pds_atom_feats = self.mpnn(pds_bg, pds_node_feats, pds_edge_feats)
        pds_atom_feats, pds_mask = pack_atom_feats(pds_bg, pds_atom_feats)

        return rts_atom_feats, rts_mask, pds_atom_feats, pds_mask

    def forward(self, 
                rts_bg, 
                rts_adms, 
                rts_node_feats, 
                rts_edge_feats, 
                pds_bg,
                pds_adms, 
                pds_node_feats, 
                pds_edge_feats,
                return_rts=False):

        rts_atom_feats, rts_mask, pds_atom_feats, pds_mask = self.embedded_net(
            rts_bg,
            rts_node_feats,
            rts_edge_feats,
            pds_bg,
            pds_node_feats,
            pds_edge_feats,
        )

        rts_atom_feats_h_cross_update, rts_self_att_scores, rts_self_attn_mask, rts_atom_cross_attention, rts_atom_cross_attention_mask = self.atom_attn_net(
            src=rts_atom_feats,
            tgt=pds_atom_feats,
            rpm=rts_adms,
            src_mask=rts_mask,
            tgt_mask=pds_mask)
        
        if return_rts:
            return rts_atom_feats_h_cross_update, rts_mask

        pds_atom_feats_h_cross_update, pds_self_att_scores, pds_self_attn_mask, pds_atom_cross_attention, pds_atom_cross_attention_mask = self.atom_attn_net(
            src=pds_atom_feats,
            tgt=rts_atom_feats,
            rpm=pds_adms,
            src_mask=pds_mask,
            tgt_mask=rts_mask)

        rts_atom_feats_h_cross_update = rts_atom_feats_h_cross_update[
            rts_mask.bool()]
        pds_atom_feats_h_cross_update = pds_atom_feats_h_cross_update[
            pds_mask.bool()]

        atom_feats_h_cross_update = torch.cat(
            [rts_atom_feats_h_cross_update, pds_atom_feats_h_cross_update],
            dim=0)
        atom_logic = self.logic_atom_net(atom_feats_h_cross_update)

        if self.output_attention:
            attentions_tuple = namedtuple('attentions', [
                'rts_atom_cross_attention', 'pds_atom_cross_attention',

                'rts_atom_cross_attention_mask',
                'pds_atom_cross_attention_mask'
            ])
            attentions = attentions_tuple(
                rts_atom_cross_attention=rts_atom_cross_attention,
                pds_atom_cross_attention=pds_atom_cross_attention,
                rts_atom_cross_attention_mask=rts_atom_cross_attention_mask,
                pds_atom_cross_attention_mask=pds_atom_cross_attention_mask)

            return atom_logic, attentions
        else:
            return atom_logic, None


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from data_loaders.rxn_dataloader import ReactionDataset, get_batch_data, MolGraphsCollator
    from dgl.data.utils import Subset
    from torch.utils.data import DataLoader

    debug = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = ReactionDataset('pistachio', debug=True, nb_workers=12)
    train_set, val_set, test_set = Subset(dataset, dataset.train_ids), Subset(
        dataset, dataset.val_ids), Subset(dataset, dataset.test_ids)

    collator = MolGraphsCollator(perform_mask=True, mask_percent=0.15)
    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=8,
                                  shuffle=True,
                                  collate_fn=collator,
                                  num_workers=14)

    # model = ReactionMGMNet(node_in_feats=dataset.node_dim,
    #                        node_out_feats=128,
    #                        edge_in_feats=dataset.edge_dim,
    #                        edge_hidden_feats=32,
    #                        num_step_message_passing=3,
    #                        attention_heads=4,
    #                        attention_layers=3,
    #                        dropout=0.3,
    #                        num_atom_types=dataset.num_atom_types)

    model = ReactionMGMTurnNet(node_in_feats=dataset.node_dim,
                               node_out_feats=128,
                               edge_in_feats=dataset.edge_dim,
                               edge_hidden_feats=32,
                               num_step_message_passing=3,
                               attention_heads=4,
                               attention_layers=3,
                               dropout=0.3,
                               cross_attn_h_rate=0.1,
                               num_atom_types=dataset.num_atom_types)

    model = model.to(device)

    for batch in tqdm(train_dataloader):
        rts_bg, rts_adms, rts_node_feats, rts_edge_feats, pds_bg, pds_adms, pds_node_feats, pds_edge_feats, labels = get_batch_data(
            batch, device)
        atom_logic = model(rts_bg, rts_adms, rts_node_feats, rts_edge_feats,
                           pds_bg, pds_adms, pds_node_feats, pds_edge_feats)
        atom_type_labels = torch.cat(labels[:2], dim=0)

        pass
