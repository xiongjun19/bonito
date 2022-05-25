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

from .conformer import ConformerEncoder
from .conformer import RelPosEncXL
from scipy.special import logsumexp



def get_stride(m):
    if hasattr(m, 'stride'):
        return m.stride if isinstance(m.stride, int) else m.stride[0]
    if isinstance(m, Convolution):
        return get_stride(m.conv)
    if isinstance(m, Serial):
        return int(np.prod([get_stride(x) for x in m]))
    return 1


class CTC_CRF(SequenceDist):

    def __init__(self, state_len, alphabet):
        super().__init__()
        self.alphabet = alphabet
        self.state_len = state_len
        self.n_base = len(alphabet[1:])
        self.idx = torch.cat([
            torch.arange(self.n_base**(self.state_len))[:, None],
            torch.arange(
                self.n_base**(self.state_len)
            ).repeat_interleave(self.n_base).reshape(self.n_base, -1).T
        ], dim=1).to(torch.int32)

    def n_score(self):
        return len(self.alphabet) * self.n_base**(self.state_len)

    def logZ(self, scores, S:semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, len(self.alphabet))
        alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return seqdist.sparse.logZ(Ms, self.idx, alpha_0, beta_T, S)

    def normalise(self, scores):
        return (scores - self.logZ(scores)[:, None] / len(scores))

    def forward_scores(self, scores, S: semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return seqdist.sparse.fwd_scores_cupy(Ms, self.idx, alpha_0, S, K=1)

    def backward_scores(self, scores, S: semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return seqdist.sparse.bwd_scores_cupy(Ms, self.idx, beta_T, S, K=1)

    def compute_transition_probs(self, scores, betas):
        T, N, C = scores.shape
        # add bwd scores to edge scores
        log_trans_probs = (scores.reshape(T, N, -1, self.n_base + 1) + betas[1:, :, :, None])
        # transpose from (new_state, dropped_base) to (old_state, emitted_base) layout
        log_trans_probs = torch.cat([
            log_trans_probs[:, :, :, [0]],
            log_trans_probs[:, :, :, 1:].transpose(3, 2).reshape(T, N, -1, self.n_base)
        ], dim=-1)
        # convert from log probs to probs by exponentiating and normalising
        trans_probs = torch.softmax(log_trans_probs, dim=-1)
        #convert first bwd score to initial state probabilities
        init_state_probs = torch.softmax(betas[0], dim=-1)
        return trans_probs, init_state_probs

    def reverse_complement(self, scores):
        T, N, C = scores.shape
        expand_dims = T, N, *(self.n_base for _ in range(self.state_len)), self.n_base + 1
        scores = scores.reshape(*expand_dims)
        blanks = torch.flip(scores[..., 0].permute(
            0, 1, *range(self.state_len + 1, 1, -1)).reshape(T, N, -1, 1), [0, 2]
        )
        emissions = torch.flip(scores[..., 1:].permute(
            0, 1, *range(self.state_len, 1, -1),
            self.state_len +2,
            self.state_len + 1).reshape(T, N, -1, self.n_base), [0, 2, 3]
        )
        return torch.cat([blanks, emissions], dim=-1).reshape(T, N, -1)

    def viterbi(self, scores):
        traceback = self.posteriors(scores, Max)
        paths = traceback.argmax(2) % len(self.alphabet)
        return paths

    def path_to_str(self, path):
        alphabet = np.frombuffer(''.join(self.alphabet).encode(), dtype='u1')
        seq = alphabet[path[path != 0]]
        return seq.tobytes().decode()

    def prepare_ctc_scores(self, scores, targets):
        # convert from CTC targets (with blank=0) to zero indexed
        targets = torch.clamp(targets - 1, 0)

        T, N, C = scores.shape
        scores = scores.to(torch.float32)
        n = targets.size(1) - (self.state_len - 1)
        stay_indices = sum(
            targets[:, i:n + i] * self.n_base ** (self.state_len - i - 1)
            for i in range(self.state_len)
        ) * len(self.alphabet)
        move_indices = stay_indices[:, 1:] + targets[:, :n - 1] + 1
        stay_scores = scores.gather(2, stay_indices.expand(T, -1, -1))
        move_scores = scores.gather(2, move_indices.expand(T, -1, -1))
        return stay_scores, move_scores

    def ctc_loss(self, scores, targets, target_lengths, loss_clip=None, reduction='mean', normalise_scores=True):
        if normalise_scores:
            scores = self.normalise(scores)
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        logz = logZ_cupy(stay_scores, move_scores, target_lengths + 1 - self.state_len)
        loss = - (logz / target_lengths)
        if loss_clip:
            loss = torch.clamp(loss, 0.0, loss_clip)
        if reduction == 'mean':
            return loss.mean()
        elif reduction in ('none', None):
            return loss
        else:
            raise ValueError('Unknown reduction type {}'.format(reduction))

    def ctc_viterbi_alignments(self, scores, targets, target_lengths):
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        return viterbi_alignments(stay_scores, move_scores, target_lengths + 1 - self.state_len)


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
            Permute([1, 0, 2]),
            LinearCRFEncoder(
                features, n_base, state_len, activation='tanh', scale=scale,
                blank_score=blank_score, expand_blanks=expand_blanks
            )
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
        scores = self.seqdist.posteriors(x.to(torch.float32)) + 1e-8
        tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
        # tracebacks2 = self.np_decode(x)
        return [self.seqdist.path_to_str(x) for x in tracebacks.cpu().numpy()]

    def np_decode(self, x):
        idx = self.seqdist.idx.cpu().numpy()
        idx_T = self.seqdist.idx.flatten().argsort().cpu().numpy()
        scores = self.np_dec_first(x.to(torch.float32), idx, idx_T) + 1e-8
        tracebacks = self.np_viterbi(scores.log(), idx, idx_T).to(torch.int16).T
        return [self.seqdist.path_to_str(x) for x in tracebacks.cpu().numpy()]

    def _init_for_dec(self, scores):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, 5)
        alpha_0 = np.full((N, 4 **(5)), 0., dtype=Ms.dtype)
        beta_T = np.full((N, 4**(5)), 0., dtype=Ms.dtype)
        return Ms, alpha_0, beta_T

    def np_dec_first(self, x, idx, idx_T):
        x = x.cpu().numpy()
        Ms, alpha_0, beta_T = self._init_for_dec(x)
        T, N, C, NZ = Ms.shape
        Ms_grad = self.np_fwd_first(Ms, idx, alpha_0)
        betas = self.np_bwd_first(Ms, idx_T, beta_T)
        Ms_grad = Ms_grad + betas[1:, :, :, None]
        ts = torch.FloatTensor(Ms_grad)
        prob = torch.softmax(ts.reshape(T, N, -1), dim=2)
        return prob


    def np_fwd_first(self, Ms, idx, v0):
        T, N, C, NZ = Ms.shape
        Ms_grad = np.full((T, N, C, NZ), -1e+38)
        for bx in range(N):
            mem = np.copy(v0[bx])
            tmp_a = np.copy(mem)
            s = [0] * NZ
            for t in range(T):
                mem = np.copy(tmp_a)
                for tx in range(C):
                    for j in range(NZ):
                        s[j] = mem[idx[tx, j]] + Ms[t, bx, tx, j]
                        Ms_grad[t, bx, tx, j] = s[j]
                    tmp_a[tx] = logsumexp(s)
        return Ms_grad

    def np_bwd_first(self, Ms, idx_T, vT):
        T, N, C, NZ = Ms.shape
        betas = np.full((T+1, N, C), -1e+38)
        for bx in range(N):
            mem = np.copy(vT[bx])
            betas[T, bx, :] = vT[bx]
            tmp_a = np.copy(mem)
            s = [0] * NZ
            for t in range(T-1, -1, -1):
                mem = np.copy(tmp_a)
                for tx in range(C):
                    for j in range(NZ):
                        ix = idx_T[tx * NZ + j]
                        n_tx = ix // NZ
                        n_j = ix - NZ * n_tx
                        s[j] = mem[n_tx] + Ms[t, bx, n_tx, n_j]
                    tmp_a[tx] = logsumexp(s)
                    betas[t, bx, tx] = tmp_a[tx]
        return betas

    def np_viterbi(self, x, idx, idx_T):
        x = x.cpu().numpy()
        Ms, alpha_0, beta_T = self._init_for_dec(x)
        T, N, C, NZ = Ms.shape
        Ms_grad = self.np_fwd_viterbi(Ms, idx, alpha_0)
        betas = self.np_bwd_viterbi(Ms, idx_T, beta_T)
        Ms_grad = Ms_grad + betas[1:, :, :, None]
        ts = torch.FloatTensor(Ms_grad).reshape(T, N, -1)
        dim = 2
        res = torch.zeros_like(ts).scatter_(dim, ts.argmax(dim, True), 1.0)
        paths = res.argmax(2) % 5
        return paths

    def np_fwd_viterbi(self, Ms, idx, v0):
        T, N, C, NZ = Ms.shape
        Ms_grad = np.full((T, N, C, NZ), -1e+38)
        for bx in range(N):
            mem = np.copy(v0[bx])
            tmp_a = np.copy(mem)
            s = [0] * NZ
            for t in range(T):
                mem = np.copy(tmp_a)
                for tx in range(C):
                    for j in range(NZ):
                        s[j] = mem[idx[tx, j]] + Ms[t, bx, tx, j]
                        Ms_grad[t, bx, tx, j] = s[j]
                    tmp_a[tx] = np.max(s)
        return Ms_grad

    def np_bwd_viterbi(self, Ms, idx_T, vT):
        T, N, C, NZ = Ms.shape
        betas = np.full((T+1, N, C), -1e+38)
        for bx in range(N):
            mem = np.copy(vT[bx])
            betas[T, bx, :] = vT[bx]
            tmp_a = np.copy(mem)
            s = [0] * NZ
            for t in range(T-1, -1, -1):
                mem = np.copy(tmp_a)
                for tx in range(C):
                    for j in range(NZ):
                        ix = idx_T[tx * NZ + j]
                        n_tx = ix // NZ
                        n_j = ix - NZ * n_tx
                        s[j] = mem[n_tx] + Ms[t, bx, n_tx, n_j]
                    tmp_a[tx] = np.max(s)
                    betas[t, bx, tx] = tmp_a[tx]
        return betas

    def decode(self, x):
        return self.decode_batch(x.unsqueeze(1))[0]

    def loss(self, scores, targets, target_lengths, **kwargs):
        return self.seqdist.ctc_loss(scores.to(torch.float32), targets, target_lengths, **kwargs)

class Model(SeqdistModel):

    def __init__(self, config):
        seqdist = CTC_CRF(
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

