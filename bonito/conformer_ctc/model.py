"""
Bonito CTC-CRF Model, replace rnn with transformer.
"""

import torch
from torch import nn
import numpy as np
import math
from typing import Optional

from bonito.nn import Module, Convolution, LinearCRFEncoder, Serial, Permute, layers, from_dict

import seqdist.sparse
from seqdist.ctc_simple import logZ_cupy, viterbi_alignments
from seqdist.core import SequenceDist, Max, Log, semiring
from fast_ctc_decode import viterbi_search

from .conformer import ConformerEncoder
from .conformer import RelPosEncXL


def get_stride(m):
    if hasattr(m, 'stride'):
        return m.stride if isinstance(m.stride, int) else m.stride[0]
    if isinstance(m, Convolution):
        return get_stride(m.conv)
    if isinstance(m, Serial):
        return int(np.prod([get_stride(x) for x in m]))
    return 1


class CTC(object):
    def __init__(self, state_len, alphabet):
        super().__init__()
        self.alphabet = alphabet
        self.state_len = state_len
        self.n_base = len(alphabet[1:])
        self.loss_func = nn.CTCLoss()

    def viterbi(self, scores):
        scores = scores.log_softmax(2).to('cpu').numpy()
        res = []
        T, B, S = scores.shape
        for i in range(B):
            seq, path = viterbi_search(scores[:, i, :], self.alphabet)
            res.append(seq)
        return res

    def ctc_loss(self, scores, targets, target_lengths, reduction='mean'):
        scores = scores.log_softmax(2)
        T, B, S = scores.size()
        input_lengths = torch.full(size=(B, ), fill_value=T, dtype=torch.long, device=scores.device)
        loss = self.loss_func(scores, targets, input_lengths, target_lengths)
        return loss



def conv(c_in, c_out, ks, stride=1, bias=False, activation=None):
    return Convolution(c_in, c_out, ks, stride=stride, padding=ks//2, bias=bias, activation=activation)


def transformer_encoder(n_base, state_len, insize=1, stride=5, winlen=19, activation='swish', rnn_type='lstm', features=768, scale=5.0, blank_score=None, expand_blanks=True, num_layers=5):
    n_heads = features // 64
    return Serial([
            conv(insize, 4, ks=5, bias=True, activation=activation),
            conv(4, 16, ks=5, bias=True, activation=activation),
            conv(16, features, ks=winlen, stride=stride, bias=True, activation=activation),
            Permute([0, 2, 1]),
            CusEncoder(num_layers, n_heads, 4 * features, d_model=features, dropout=0.1),
            Permute([1, 0, 2]),
            nn.Linear(features, state_len)
    ])


class SeqdistModel(Module):
    def __init__(self, encoder, seqdist):
        super().__init__()
        self.seqdist = seqdist
        self.encoder = encoder
        self.stride = get_stride(encoder)
        self.alphabet = seqdist.alphabet

    def forward(self, x):
        return self.encoder(x)

    def decode_batch(self, x):
        return self.seqdist.viterbi(x)
        # return [self.seqdist.path_to_str(x) for x in tracebacks.cpu().numpy()]

    def decode(self, x):
        return self.decode_batch(x.unsqueeze(1))[0]

    def loss(self, scores, targets, target_lengths, **kwargs):
        return self.seqdist.ctc_loss(scores.to(torch.float32), targets, target_lengths, **kwargs)

class Model(SeqdistModel):

    def __init__(self, config):
        seqdist = CTC(
            state_len=config['global_norm']['state_len'],
            alphabet=config['labels']['labels']
        )
        if 'type' in config['encoder']: #new-style config
            encoder = from_dict(config['encoder'])
        else: #old-style
            encoder = transformer_encoder(seqdist.n_base, seqdist.state_len, insize=config['input']['features'], **config['encoder'])
        super().__init__(encoder, seqdist)
        self.config = config


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

