"""
Bonito CTC-CRF Model, replace rnn with transformer.
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import math
from typing import Optional

from bonito.nn import Convolution, Serial, Permute
from .transformer import TransformerEncoder
from .transformer import TransformerDecoder
from .transformer import PositionalEncoding
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


def transformer_encoder(n_base, state_len, insize=1, stride=5, winlen=19, activation='swish',
                        rnn_type='lstm', features=768, scale=5.0, blank_score=None, expand_blanks=True,
                        num_layers=5, dropout=0.1):
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
        self.voc_size = self.n_base ** self.n_gram + 2
        self.blank_id = 0
        self.bos = self.voc_size - 1
        self.encoder = transformer_encoder(self.n_base, self.state_len,
                                           insize=config['input']['features'], **config['encoder'])
        self.decoder = Decoder(self.voc_size, self.blank_id, **config['decoder'])
        self.lb_sm = config['decoder'].get('lb_sm', 0.)
        # self.searcher = TransducerSearcher(self.pred_net, 0, 4, 2.3, 2.3)
        self._freeze = False

    def forward(self, x):
        if self._freeze:
            self.encoder.eval()
            with torch.no_grad():
                enc = self.encoder(x)
        else:
            enc = self.encoder(x)
        return enc

    def decode_batch(self, scores):
        b = scores.size(0)
        res = [[0]] * b
        # n_best_match, n_match_score = self.searcher.beam_search(scores)
        # id_list = [self._scores2ids(match) for match in n_best_match]
        # res = [self._ids_to_str(match) for match in id_list]
        return res

    def _scores2ids(self, score_list):
        if len(score_list) > 0:
            offset = self.n_gram - 1
            res = [0] * (len(score_list) + self.n_gram - 1)
            for score in score_list:
                score -= 1
                new_score = score % self.n_base + 1
                res[offset] = new_score
                offset += 1
            begin_score = score_list[0] - 1
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
        tf_loss = self._comp_tf_loss(enc, targets, target_lengths)
        return tf_loss

    def _comp_tf_loss(self, enc, targets, target_lengths):
        bos_targets, eos_targets, new_lengths = self._prep_targets(targets, target_lengths)
        tgt_mask, tgt_key_padding = self._get_mask(bos_targets)
        dec_logit = self.decoder(
                bos_targets,
                enc,
                src_mask=None,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding,
                src_key_padding_mask=None
        )
        log_probs = F.log_softmax(dec_logit, dim=-1)
        loss = kldiv_loss(log_probs, eos_targets, label_smoothing=self.lb_sm, pad_idx=self.blank_id)
        return loss

    def _prep_targets(self, x, target_lengths):
        bs = x.size()[0]
        bos_padding = torch.full((bs, 1), self.bos, device=x.device, dtype=x.dtype)
        res = torch.cat([bos_padding, x], dim=1)
        tail_padding = torch.full((bs, 1), self.blank_id, device=x.device, dtype=x.dtype)
        res_tail = torch.cat([x, tail_padding], dim=1)
        res_tail.scatter_(1, target_lengths.unsqueeze(1), self.bos)
        res_lengths = target_lengths + 1
        return res, res_tail, res_lengths

    def _get_mask(self, x):
        key_padding = (x == self.blank_id).detach()
        att_mask = get_lookahead_mask(x)
        return att_mask, key_padding

    def _cvt_targets(self, targets, target_lengths):
        bs, padded_len = targets.size()
        res_lens = target_lengths - self.n_gram + 1
        raw_targets = torch.clamp(targets - 1, 0)
        raw_targets_arr = [
            raw_targets[:, i:(padded_len - self.n_gram + 1 + i)].unsqueeze(0)
            * (self.n_base ** (self.n_gram - 1 - i))
            for i in range(self.n_gram)
        ]
        res_targets = torch.cat(raw_targets_arr, 0).sum(dim=0) + 1
        idx_ts = torch.arange(padded_len - self.n_gram + 1).unsqueeze(0).repeat(bs, 1)
        idx_ts = idx_ts.to(targets.device)
        mask = idx_ts >= res_lens.unsqueeze(1)
        res_targets = res_targets.masked_fill(mask, self.blank_id)
        return res_targets, res_lens

    def prepend(self, x, val):
        B = x.size()[0]
        val_padding = torch.full((B, 1), val, device=x.device, dtype=x.dtype)
        res = torch.cat([val_padding, x], dim=1)
        return res


class Decoder(nn.Module):
    def __init__(self, vocab_size, blank_id,
                 d_model=512, nhead=8,
                 layers=6, dropout=0.1, normalize_before=True,
                 **kwargs):
        super().__init__()
        self.decoder = TransformerDecoder(
                num_layers=layers,
                nhead=nhead,
                d_ffn=4 * d_model,
                d_model=d_model,
                dropout=dropout,
                normalize_before=normalize_before,
                causal=True
        )
        self.dropout = nn.Dropout(dropout)
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=blank_id)
        self.linear = nn.Linear(d_model, vocab_size)
        self.emb_scale = math.sqrt(d_model)
        self.pos_emb = PositionalEncoding(d_model)

    def forward(
            self, x, enc_out,
            src_mask=None,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            src_key_padding_mask=None):
        emb = self.emb(x) * self.emb_scale
        pe = self.pos_emb(emb)
        in_x = emb + pe

        decoder_out, _, _ = self.decoder(
            tgt=in_x,
            memory=enc_out,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.linear(decoder_out)
        return logits


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
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(
                num_layers, nhead, d_ffn,
                d_model=d_model,
                dropout=dropout,
                normalize_before=normalize_before
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ):
        output = self.norm(src)
        pe = self.pos_enc(output)
        output = output + pe
        output, _ = self.encoder(output, src_mask, src_key_padding_mask)
        return output


def get_lookahead_mask(padded_input):
    seq_len = padded_input.shape[1]
    mask = (
        torch.triu(torch.ones((seq_len, seq_len), device=padded_input.device))
        == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask.detach()


def kldiv_loss(
    log_probabilities,
    targets,
    length=None,
    label_smoothing=0.0,
    pad_idx=0,
    reduction="batchmean",
):
    if log_probabilities.dim() == 2:
        log_probabilities = log_probabilities.unsqueeze(1)

    bz, time, n_class = log_probabilities.shape
    targets = targets.long().detach()

    confidence = 1 - label_smoothing

    log_probabilities = log_probabilities.view(-1, n_class)
    targets = targets.view(-1)
    with torch.no_grad():
        true_distribution = log_probabilities.clone()
        true_distribution.fill_(label_smoothing / (n_class - 1))
        ignore = targets == pad_idx
        targets = targets.masked_fill(ignore, 0)
        true_distribution.scatter_(1, targets.unsqueeze(1), confidence)

    loss = torch.nn.functional.kl_div(
        log_probabilities, true_distribution, reduction="none"
    )
    loss = loss.masked_fill(ignore.unsqueeze(1), 0)

    # return loss according to reduction specified
    if reduction == "mean":
        return loss.sum().mean()
    elif reduction == "batchmean":
        return loss.sum() / bz
    elif reduction == "batch":
        return loss.view(bz, -1).sum(1) / length
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
