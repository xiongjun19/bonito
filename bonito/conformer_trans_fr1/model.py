"""
Bonito CTC-CRF Model, replace rnn with transformer.
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math
from typing import Optional
from torchaudio import transforms
from collections import OrderedDict

from bonito.nn import Module, Convolution, LinearCRFEncoder, Serial, Permute, layers, from_dict
from .conformer import ConformerEncoder
from .conformer import RelPosEncXL
from .searcher import TransducerSearcher
from bonito import util as bo_util


def get_stride(m):
    if hasattr(m, 'stride'):
        return m.stride if isinstance(m.stride, int) else m.stride[0]
    if isinstance(m, Convolution):
        return get_stride(m.conv)
    if isinstance(m, Serial):
        return int(np.prod([get_stride(x) for x in m]))
    return 1


def conv(c_in, c_out, ks, stride=1, bias=False, activation=None):
    return Convolution(c_in, c_out, ks, stride=stride, padding=ks//2, bias=bias, activation=activation)


def transformer_encoder(n_base, state_len, insize=1, stride=5, winlen=19, activation='swish', rnn_type='lstm', features=768, scale=5.0, blank_score=None, expand_blanks=True, num_layers=5, dropout=0.1):
    n_heads = features // 64
    return Serial([
            conv(insize, 4, ks=5, bias=True, activation=activation),
            conv(4, 16, ks=5, bias=True, activation=activation),
            conv(16, features, ks=winlen, stride=stride, bias=True, activation=activation),
            Permute([0, 2, 1]),
            CusEncoder(num_layers, n_heads, 4 * features, d_model=features, dropout=dropout),
    ])


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state_len = config['global_norm']['state_len']
        self.alphabet = config['labels']['labels']
        self.n_base = self.state_len - 1
        self.encoder = transformer_encoder(self.n_base, self.state_len, insize=config['input']['features'], **config['encoder'])
        self.pred_net = PredNet(self.state_len, **config['pred_net'])
        self.joint_net = JointNet(self.state_len, **config['joint'])
        self.loss_func = transforms.RNNTLoss(blank=0)
        self.searcher = TransducerSearcher(self.pred_net, self.joint_net, 0, 4, state_beam=2.3, expand_beam=2.3)
        self._load_pretrain_enc()

    def train(self):
        self.encoder.train()
        self.pred_net.train()
        self.joint_net.train()

    def eval(self):
        self.encoder.eval()
        self.pred_net.eval()
        self.joint_net.eval()

    def _load_pretrain_enc(self):
        if 'pretrain' in self.config:
            sub_cfg = self.config['pretrain']
            if 'enc_path' in sub_cfg:
                enc_path = sub_cfg['enc_path']
                state_dict = torch.load(enc_path, map_location='cpu')
                name_mps = self.match_names(state_dict, self.encoder)
                state_dict = {k2: state_dict[k1] for k1, k2 in name_mps.items()}
                self.encoder.load_state_dict(state_dict)
                self.encoder.eval()
    
    def match_names(self, state_dict, model):
        keys_and_shapes = lambda state_dict: zip(*[
            (k, s) for s, i, k in sorted([(v.shape, i, k)
            for i, (k, v) in enumerate(state_dict.items())])
        ])
        k1, s1 = keys_and_shapes(state_dict)
        k2, s2 = keys_and_shapes(model.state_dict())
        remap = dict(zip(k1[: len(k2)], k2))
        return OrderedDict([(k, remap[k]) for k in state_dict.keys() if k in remap])  

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x)

    def decode_batch(self, scores):
        n_best_match, n_match_score = self.searcher.beam_search(scores)
        res = [self._ids_to_str(match) for match in n_best_match]
        return res

    def _ids_to_str(self, ids):
        res = [self.alphabet[_id] for _id in ids if _id > 0]
        return "".join(res)

    def loss(self, enc, targets, target_lengths, **kwargs):
        """
         enc: [B, T, H]
         targets: [B, U],
         target_lengths  containing lengths of eatch sequence from encoder
        """  
        raw_targets = torch.clone(targets)
        targets = self.prepend(targets, 0)
        max_len = torch.max(target_lengths)
        targets = targets[:, :max_len+1]
        raw_targets = raw_targets[:, :max_len]
        preds, hid = self.pred_net(targets)
        scores = self.joint_net(enc, preds) # [B, T, U+1, state_len]
        B, T, *_ = enc.size()
        logit_lengths = torch.full((B, ), T, dtype=torch.int, device=scores.device)
        raw_targets = raw_targets.type_as(logit_lengths)
         
        target_lengths = target_lengths.type_as(logit_lengths)
        # B, U, *_ = targets.size()
        # target_lengths = torch.full((B, ), U-1, dtype=torch.int, device=scores.device) 
        return self.loss_func(scores, raw_targets, logit_lengths, target_lengths)

    def prepend(self, x, val):
        B = x.size()[0]
        val_padding = torch.full((B, 1), val, device=x.device, dtype=x.dtype)
        res = torch.cat([val_padding, x], dim=1)
        return res


class PredNet(nn.Module):
    def __init__(self, alphabet, emb_dim=512, hid_dim=512, layers=1, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=1, batch_first=True)
        self.emb = nn.Embedding(alphabet, emb_dim)

    def forward(self, x, hidden=None):
        emb = self.emb(x)
        emb = self.dropout(emb)
        if hidden is None:
            output, new_h = self.rnn(emb)
        else:
            output, new_h = self.rnn(emb, hidden)
        return output, new_h


class JointNet(nn.Module):
    def __init__(self, alphabet, hid_dim=512, dropout=0.1):
        super().__init__()
        self.alphabet = alphabet
        self.linear = nn.Linear(hid_dim, self.alphabet)
        self.no_lin = nn.LeakyReLU(0.1)

    def forward(self, trans, preds):
        """
            trans: [B, T, h];
            preds: [B, U+1, h];
        """
        new_trans = trans.unsqueeze(2)
        new_preds = preds.unsqueeze(1)
        joint = new_trans + new_preds
        joint = self.linear(joint)
        joint = self.no_lin(joint) 
        probs = F.log_softmax(joint, dim=-1)
        return joint 

    def pred_logits(self, trans, preds):
        """
            trans: [B, T, h];
            preds: [B, U+1, h];
        """
        new_trans = trans.unsqueeze(2)
        new_preds = preds.unsqueeze(1)
        joint = new_trans + new_preds
        logit = self.linear(joint)
        return logit


class CusEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation='relu',
        normalize_before=False,
        causal=False,
        batch_first=True
    ):
        super().__init__()
        self.pos_enc = RelPosEncXL(d_model)
        self.encoder = ConformerEncoder(num_layers, d_model, d_ffn, nhead,
                                        dropout=dropout)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ):
        output = src
        pe = self.pos_enc(output)
        output, _ = self.encoder(output, src_mask, src_key_padding_mask,
                                 pos_embs=pe)
        return output

