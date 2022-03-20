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
import k2

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
        self.blank_id=0
        self.config = config
        self.state_len = config['global_norm']['state_len']
        self.alphabet = config['labels']['labels']
        self.n_base = self.state_len - 1
        self.encoder = transformer_encoder(self.n_base, self.state_len, insize=config['input']['features'], **config['encoder'])
        self.enc_linear = nn.Linear(config['encoder']['features'], self.state_len)
        self.pred_net = PredNet(self.state_len, self.blank_id,  **config['pred_net'])
        self.joint = JointNet(self.state_len, self.state_len, **config['joint'])
        self.searcher = TransducerSearcher(self.pred_net, self.joint, self.blank_id, 4, 2.3, 2.3)
        self._freeze = self._load_pretrain_enc()

    # def train(self):
    #     self.encoder.train()
    #     self.enc_linear.train()
    #     # self.ln.train()
    #     self.pred_net.train()

    # def eval(self):
    #     self.encoder.eval()
    #     self.enc_linear.eval()
    #     # self.ln.eval()
    #     self.pred_net.eval()
    #     # self.joint_net.eval()

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
        enc = self.enc_linear(enc)
        return enc

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
        targets = self.prepend(targets, self.blank_id)
        max_len = torch.max(target_lengths)
        targets = targets[:, :max_len+1]
        raw_targets = raw_targets[:, :max_len]
        preds = self.pred_net(targets)

        B, T, *_ = enc.size()
        logit_lengths = torch.full((B, ), T, dtype=torch.int, device=enc.device)
        target_lengths = target_lengths.type_as(logit_lengths)

        boundary = torch.zeros(
            (x.size(0), 4), dtype=torch.int64, device=x.device
        )
        boundary[:, 2] = target_lengths
        boundary[:, 3] = logit_lengths

        lm_scale = 0.25
        am_scale = 0.0
        simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
            lm=preds,
            am=encoder_out,
            symbols=raw_targets,
            termination_symbol=self.blank_id,
            lm_only_scale=lm_scale,
            am_only_scale=am_scale,
            boundary=boundary,
            reduction="sum",
            return_grad=True,
        )

        prune_range = 5
        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=enc, lm=preds, ranges=ranges
        )

        logits = self.joint(am_pruned, lm_pruned)
        pruned_loss = k2.rnnt_loss_pruned(
            logits=logits,
            symbols=raw_targets,
            ranges=ranges,
            termination_symbol=self.blank_id,
            boundary=boundary,
            reduction="sum",
        )
        loss = 0.5 * simple_loss + pruned_loss
        return loss

    def prepend(self, x, val):
        B = x.size()[0]
        val_padding = torch.full((B, 1), val, device=x.device, dtype=x.dtype)
        res = torch.cat([val_padding, x], dim=1)
        return res


class PredNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        blank_id: int,
        emb_dim=512,
        context_size=5,
        dropout=0.1
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          emb_dim:
            Dimension of the input embedding.
          blank_id:
            The ID of the blank symbol.
          context_size:
            Number of previous words to use to predict the next word.
            1 means bigram; 2 means trigram. n means (n+1)-gram.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=blank_id,
        )
        self.blank_id = blank_id
        self.dropout = nn.Dropout(dropout)

        assert context_size >= 1, context_size
        self.context_size = context_size
        if context_size > 1:
            self.conv = nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                kernel_size=context_size,
                padding=0,
                groups=embedding_dim,
                bias=False,
            )
        self.output_linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, y: torch.Tensor, need_pad: bool = True) -> torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U) with blank prepended.
          need_pad:
            True to left pad the input. Should be True during training.
            False to not pad the input. Should be False during inference.
        Returns:
          Return a tensor of shape (N, U, embedding_dim).
        """
        embedding_out = self.embedding(y)
        if self.context_size > 1:
            embedding_out = embedding_out.permute(0, 2, 1)
            if need_pad is True:
                embedding_out = F.pad(
                    embedding_out, pad=(self.context_size - 1, 0)
                )
            else:
                # During inference time, there is no need to do extra padding
                # as we only need one output
                assert embedding_out.size(-1) == self.context_size
            embedding_out = self.conv(embedding_out)
            embedding_out = embedding_out.permute(0, 2, 1)
        embedding_out = self.output_linear(F.relu(embedding_out))
        return embedding_out


class JointNet(nn.Module):
    def __init__(self, vocab_size, input_dim, inner_dim=32, dropout=0.1):
        super().__init__()
        self.inner_linear = nn.Linear(input_dim, inner_dim)
        self.output_linear = nn.Linear(inner_dim, vocab_size)
        self.dropout=nn.Dropout(dropout)

    def forward(
        self, encoder_out: torch.Tensor, decoder_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, s_range, C).
          decoder_out:
            Output from the decoder. Its shape is (N, T, s_range, C).
        Returns:
          Return a tensor of shape (N, T, s_range, C).
        """
        assert encoder_out.ndim == decoder_out.ndim == 4
        assert encoder_out.shape == decoder_out.shape

        logit = encoder_out + decoder_out
        logit = self.inner_linear(torch.tanh(logit))
        output = self.output_linear(F.relu(logit))
        return output


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
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ):
        output = src
        pe = self.pos_enc(output)
        pe = self.dropout(pe)
        output = self.dropout(output)
        output, _ = self.encoder(output, src_mask, src_key_padding_mask,
                                 pos_embs=pe)
        return output

