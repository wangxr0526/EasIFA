import itertools
import os
import warnings

import torch
from torch import nn
import esm
import re
import urllib
from argparse import Namespace
from pathlib import Path

from torchdrug import core, layers, utils, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import warnings

import torch
from torch import nn
import esm
import re
import urllib
from argparse import Namespace
from pathlib import Path

from torchdrug import core, layers, utils, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from esm.model.esm2 import ESM2

foldseek_seq_vocab = "ACDEFGHIKLMNPQRSTVWY#"
foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"

def load_esm_saprot(path: str):
    """
    Load SaProt model of esm version.
    Args:
        path: path to SaProt model
    """
    
    # Initialize the alphabet
    tokens = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>"]
    for seq_token, struc_token in itertools.product(foldseek_seq_vocab, foldseek_struc_vocab):
        token = seq_token + struc_token
        tokens.append(token)
    
    alphabet = esm.data.Alphabet(standard_toks=tokens,
                                 prepend_toks=[],
                                 append_toks=[],
                                 prepend_bos=True,
                                 append_eos=True,
                                 use_msa=False)
    
    alphabet.all_toks = alphabet.all_toks[:-2]
    alphabet.unique_no_split_tokens = alphabet.all_toks
    alphabet.tok_to_idx = {tok: i for i, tok in enumerate(alphabet.all_toks)}
    
    # Load weights
    data = torch.load(path)
    weights = data["model"]
    config = data["config"]
    
    # Initialize the model
    model = ESM2(
        num_layers=config["num_layers"],
        embed_dim=config["embed_dim"],
        attention_heads=config["attention_heads"],
        alphabet=alphabet,
        token_dropout=config["token_dropout"],
    )
    
    load_weights(model, weights)
    return model, alphabet


def load_weights(model, weights):
    model_dict = model.state_dict()

    unused_params = []
    missed_params = list(model_dict.keys())

    for k, v in weights.items():
        if k in model_dict.keys():
            model_dict[k] = v
            missed_params.remove(k)

        else:
            unused_params.append(k)

    if len(missed_params) > 0:
        print(f"\033[31mSome weights of {type(model).__name__} were not "
              f"initialized from the model checkpoint: {missed_params}\033[0m")

    if len(unused_params) > 0:
        print(f"\033[31mSome weights of the model checkpoint were not used: {unused_params}\033[0m")

    model.load_state_dict(model_dict)

class EvolutionaryScaleModelingSaProt(nn.Module):
    """
    The protein language model, Evolutionary Scale Modeling (ESM) proposed in
    `Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences`_.

    .. _Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences:
        https://www.biorxiv.org/content/10.1101/622803v1.full.pdf

    Parameters:
        path (str): path to store ESM model weights
        model (str, optional): model name. Available model names are ``ESM-1b``, ``ESM-1v`` and ``ESM-1b-regression``.
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """





    output_dim = {
        "SaProt_650M_AF2": 1280,
    }

    num_layer = {
        "SaProt_650M_AF2": 33,
    }
    
    max_input_length = 1024 - 2

    def __init__(self, path, model="SaProt_650M_AF2", readout="mean"):
        super(EvolutionaryScaleModelingSaProt, self).__init__()
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        model_path = os.path.join(path, f'{model}.pt')
        _model, alphabet = load_esm_saprot(model_path)

        self.output_dim = self.output_dim[model]
        self.model = _model
        self.alphabet = alphabet
        self.repr_layer = self.num_layer[model]


        if readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        else:
            raise ValueError("Unknown readout `%s`" % readout)



    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the residue representations and the graph representation(s).

        Parameters:
            graph (Protein): :math:`n` protein(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``residue_feature`` and ``graph_feature`` fields:
                residue representations of shape :math:`(|V_{res}|, d)`, graph representations of shape :math:`(n, d)`
        """
        size = graph.num_residues
        if (size > self.max_input_length).any():   # 这部分切割还没做适配，不过数据集已经限制单个的酶最大只能有1000个残基，应该触及不到这里
            warnings.warn("ESM can only encode proteins within %d residues. Truncate the input to fit into ESM."
                          % self.max_input_length)
            starts = size.cumsum(0) - size
            size = size.clamp(max=self.max_input_length)
            ends = starts + size
            mask = functional.multi_slice_mask(starts, ends, graph.num_residue)
            input = input[mask]
            graph = graph.subresidue(mask)
        size_ext = size
        size_ext = size_ext + 2    # 添加bos和eos的空余
        # if self.alphabet.prepend_bos:
        #     bos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.alphabet.cls_idx
        #     input, size_ext = functional._extend(bos, torch.ones_like(size_ext), input, size_ext)
        # if self.alphabet.append_eos:
        #     eos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.alphabet.eos_idx
        #     input, size_ext = functional._extend(input, size_ext, eos, torch.ones_like(size_ext))
        # input = functional.variadic_to_padded(input, size_ext, value=self.alphabet.padding_idx)[0]

        output = self.model(input, repr_layers=[self.repr_layer])
        residue_feature = output["representations"][self.repr_layer]

        residue_feature = functional.padded_to_variadic(residue_feature, size_ext)
        starts = size_ext.cumsum(0) - size_ext
        if self.alphabet.prepend_bos:
            starts = starts + 1
        ends = starts + size
        mask = functional.multi_slice_mask(starts, ends, len(residue_feature))
        residue_feature = residue_feature[mask]
        graph_feature = self.readout(graph, residue_feature)

        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature
        }

