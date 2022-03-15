# coding=utf8

import torch
import numpy as np
from typing import Dict, Optional
from typing import List, Any
from dataclasses import dataclass


class TransducerSearcher(object):
    def __init__(self, pred_net, blank_id,
                 beam_size=2, state_beam=2.3, expand_beam=2.3):
        self.pred_net = pred_net
        self.blank_id = blank_id
        self.state_beam = state_beam
        self.expand_beam = expand_beam
        self.beam_size = beam_size

    def beam_search(self, encs):
        B = encs.size()[0]
        n_best_match = [0] * B
        n_match_score = [0] * B
        with torch.no_grad():
            for i in range(B):
                best_match = self.search_single(encs, i)
                # best_match, match_score = self.pool.apply_asyc(self.search_single, (encs, i))
                n_best_match[i] = best_match
                n_match_score[i] = None 
        return n_best_match, n_match_score

    def search_single(self, encoder_out, i_batch):
        device = encoder_out.device
        T = encoder_out.size(1)
        B = HypothesisList()
        B.add(
            Hypothesis(
                ys=[self.blank_id],
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                hid=self.pred_net.init_hid(encoder_out.dtype, device, 1)
            )
        )

        for t in range(T):
            A = list(B)
            B = HypothesisList()
            log_probs, new_hid = self.comp_trans_probs(encoder_out, A, i_batch, t,  device)
            self._update_beam(log_probs, new_hid, A, B, device)

        best_hyp = B.get_most_probable(length_norm=True)
        ys = best_hyp.ys[:]
        return ys

    def _update_beam(self, log_probs, next_hid, A, B, device):
        voc_size = log_probs.size(-1)
        ys_log_probs = torch.cat([hyp.log_prob.reshape([1, 1]) for hyp in A])
        log_probs.add_(ys_log_probs)
        log_probs = log_probs.reshape(-1)
        topk_log_probs, topk_indexes = log_probs.topk(self.beam_size)

        # topk_hyp_indexes are indexes into `A`
        topk_hyp_indexes = topk_indexes // voc_size
        topk_token_indexes = topk_indexes % voc_size

        topk_hyp_indexes = topk_hyp_indexes.tolist()
        topk_token_indexes = topk_token_indexes.tolist()

        for i in range(len(topk_hyp_indexes)):
            b_idx = topk_hyp_indexes[i]
            hyp = A[b_idx]
            new_ys = hyp.ys[:]
            new_hid = hyp.hid
            new_token = topk_token_indexes[i]
            if new_token != self.blank_id:
                new_ys.append(new_token)
                new_hid = tuple([x[:, b_idx:b_idx+1] for x in next_hid])
            new_log_prob = topk_log_probs[i]
            new_hyp = Hypothesis(ys=new_ys, log_prob=new_log_prob, hid=new_hid)
            B.add(new_hyp)

    def comp_trans_probs(self, enc, A, i_batch, t, device):
        current_encoder_out = enc[i_batch:i_batch + 1, t:t+1, :]
        # current_encoder_out is of shape (1, 1, encoder_out_dim)
        decoder_input = torch.tensor(
            [hyp.ys[-1:] for hyp in A],
            device=device,
        )
        hid_inputs = self._init_hids(A)
        decoder_out, new_hid = self.pred_net(decoder_input, hid_inputs)
        # decoder_output is of shape (num_hyps, 1, decoder_output_dim)
        joint_logits = current_encoder_out.unsqueeze(2) + decoder_out.unsqueeze(1)

        # logits is of shape (num_hyps, vocab_size)
        log_probs = joint_logits.log_softmax(dim=-1)
        return log_probs.squeeze(1).squeeze(1), new_hid

    def _init_hids(self, A):
        c_list = zip(*[hyp.hid for hyp in A])
        res = [torch.cat(x, dim=1) for x in c_list]
        res = tuple(res)
        return res


@dataclass
class Hypothesis:
    # The predicted tokens so far.
    # Newly predicted tokens are appended to `ys`.
    ys: List[int]

    # The log prob of ys
    log_prob: float
    hid: Any = None

    @property
    def key(self) -> str:
        """Return a string representation of self.ys"""
        return "_".join(map(str, self.ys))


class HypothesisList(object):
    def __init__(self, data: Optional[Dict[str, Hypothesis]] = None) -> None:
        """
        Args:
          data:
            A dict of Hypotheses. Its key is its `value.key`.
        """
        if data is None:
            self._data = {}
        else:
            self._data = data

    @property
    def data(self) -> Dict[str, Hypothesis]:
        return self._data

    def add(self, hyp: Hypothesis) -> None:
        """Add a Hypothesis to `self`.
        If `hyp` already exists in `self`, its probability is updated using
        `log-sum-exp` with the existed one.
        Args:
          hyp:
            The hypothesis to be added.
        """
        key = hyp.key
        if key in self:
            old_hyp = self._data[key]  # shallow copy
            # torch.logaddexp(
            #     old_hyp.log_prob, hyp.log_prob, out=old_hyp.log_prob
            # )
            old_hyp.log_prob = torch.logsumexp(
                torch.cat([old_hyp.log_prob.view([1]), hyp.log_prob.view([1])],  dim=0),0) 
        else:
            self._data[key] = hyp

    def get_most_probable(self, length_norm: bool = False) -> Hypothesis:
        """Get the most probable hypothesis, i.e., the one with
        the largest `log_prob`.
        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        Returns:
          Return the hypothesis that has the largest `log_prob`.
        """
        if length_norm:
            return max(
                self._data.values(), key=lambda hyp: hyp.log_prob / len(hyp.ys)
            )
        else:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob)

    def remove(self, hyp: Hypothesis) -> None:
        """Remove a given hypothesis.
        Caution:
          `self` is modified **in-place**.
        Args:
          hyp:
            The hypothesis to be removed from `self`.
            Note: It must be contained in `self`. Otherwise,
            an exception is raised.
        """
        key = hyp.key
        assert key in self, f"{key} does not exist"
        del self._data[key]

    def filter(self, threshold: torch.Tensor) -> "HypothesisList":
        """Remove all Hypotheses whose log_prob is less than threshold.
        Caution:
          `self` is not modified. Instead, a new HypothesisList is returned.
        Returns:
          Return a new HypothesisList containing all hypotheses from `self`
          with `log_prob` being greater than the given `threshold`.
        """
        ans = HypothesisList()
        for _, hyp in self._data.items():
            if hyp.log_prob > threshold:
                ans.add(hyp)  # shallow copy
        return ans

    def topk(self, k: int) -> "HypothesisList":
        """Return the top-k hypothesis."""
        hyps = list(self._data.items())

        hyps = sorted(hyps, key=lambda h: h[1].log_prob, reverse=True)[:k]

        ans = HypothesisList(dict(hyps))
        return ans

    def __contains__(self, key: str):
        return key in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        s = []
        for key in self:
            s.append(key)
        return ", ".join(s)


