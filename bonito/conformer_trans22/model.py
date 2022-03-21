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
from transducer import TransducerLoss

from bonito.nn import Module, Convolution, LinearCRFEncoder, Serial, Permute, layers, from_dict
from .conformer import ConformerEncoder
from .conformer import RelPosEncXL
from bonito import util as bo_util
from .searcher3 import TransducerSearcher


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
        self.n_gram = 5
        self.voc_size = self.n_base ** self.n_gram + 1
        self.blank_id = 0
        self.encoder = transformer_encoder(self.n_base, self.state_len, insize=config['input']['features'], **config['encoder'])
        self.enc_linear = nn.Linear(config['encoder']['features'], self.voc_size)
        # self.ln = nn.LayerNorm(config['encoder']['features'])
        self.pred_net = PredNet(self.voc_size, **config['pred_net'])
        self.searcher = TransducerSearcher(self.pred_net, 0, 4, 2.3, 2.3)
        self.criterion = TransducerLoss()
        self._freeze = self._load_pretrain_enc()

    def train(self):
        self.encoder.train()
        self.enc_linear.train()
        # self.ln.train()
        self.pred_net.train()

    def eval(self):
        self.encoder.eval()
        self.enc_linear.eval()
        # self.ln.eval()
        self.pred_net.eval()
        # self.joint_net.eval()

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
                return True
        return False

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
        if self._freeze:
            self.encoder.eval()
            with torch.no_grad():
                enc = self.encoder(x)
        else:
            enc = self.encoder(x)
        # enc = self.ln(enc)
        enc = self.enc_linear(enc)
        return enc

    def decode_batch(self, scores):
        n_best_match, n_match_score = self.searcher.beam_search(scores)
        id_list = [self._scores2ids(match) for match in n_best_match]
        res = [self._ids_to_str(match) for match in id_list]
        return res

    def _scores2ids(self, score_list):
        if len(score_list) > 0:
            offset = self.n_gram - 1
            res = [0] * (self.len(score_list) + self.n_gram - 1)
            for score in score_list:
                new_score = score % self.n_base + 1
                res[offset] = new_score
                offset += 1
            begin_score = score_list[0]
            for i in range(self.n_base):
                base = self.n_base ** (self.n_gram - 1 -i)
                val = begin_score // base  % self.n_base + 1
                res[i] = val
            return res
        return []

    def _ids_to_str(self, ids):
        res = [self.alphabet[_id] for _id in ids if _id > 0]
        return "".join(res)

    def loss(self, enc, targets, target_lengths, **kwargs):
        """
         enc: [B, T, H]
         targets: [B, U],
         target_lengths  containing lengths of eatch sequence from encoder
        """
        targets, target_lengths = self._cvt_targets(targets, target_lengths)
        raw_targets = torch.clone(targets)
        targets = self.prepend(targets, self.blank_id)
        max_len = torch.max(target_lengths)
        targets = targets[:, :max_len+1]
        raw_targets = raw_targets[:, :max_len]
        preds, hid = self.pred_net(targets)
        B, T, *_ = enc.size()
        logit_lengths = torch.full((B, ), T, dtype=torch.int, device=enc.device)
        raw_targets = raw_targets.type_as(logit_lengths)
        target_lengths = target_lengths.type_as(logit_lengths)
        # enc_norm = torch.norm(enc.detach(), p=2,  dim=2)
        # pred_norm = torch.norm(preds.detach(), p=2,  dim=2)
        # print("enc norm is: ")
        # print(enc_norm)
        # print("pred norm is: ")
        # print(pred_norm)
        return self.criterion(enc, preds, raw_targets, logit_lengths, target_lengths).mean()

    def _cvt_targets(self, targets, target_lengths):
        bs, padded_len = targets.size()
        res_lens = target_lengths - self.n_gram + 1
        raw_targets  = torch.clamp(targets - 1, 0)
        raw_targets_arr = [raw_targets[:, i:(padded_len - self.n_gram + 1 + i)].unsqueeze(0) * (self.n_base ** (self.n_gram - 1 -i))
                           for i in range(self.n_gram)]
        res_targets = torch.cat(raw_targets_arr, 0).sum(dim=0) + 1
        idx_ts = torch.arange(padded_len).unsqueeze(0).repeat(bs, 1)
        mask = idx_ts >= res_lens
        res_targets = res_targets.mask_fill(mask, 0)
        return res_targets, res_lens

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
        self.linear = nn.Linear(hid_dim, alphabet)
        # self.ln = nn.LayerNorm(hid_dim)

    def forward(self, x, hidden=None):
        emb = self.emb(x)
        emb = self.dropout(emb)
        if hidden is None:
            output, new_h = self.rnn(emb)
        else:
            output, new_h = self.rnn(emb, hidden)
        # output = self.ln(output)
        output = self.linear(output)
        return output, new_h

    def init_hid(self, dtype, device, bs=1):
        num_directions = 2 if self.rnn.bidirectional else 1
        real_hidden_size = self.rnn.proj_size if self.rnn.proj_size > 0 else self.rnn.hidden_size
        h_zeros = torch.zeros(self.rnn.num_layers * num_directions,
                              bs, real_hidden_size,
                              dtype=dtype, device=device)
        c_zeros = torch.zeros(self.rnn.num_layers * num_directions,
                              bs, self.rnn.hidden_size,
                              dtype=dtype, device=device)
        hx = (h_zeros, c_zeros)
        return hx


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

