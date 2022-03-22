# coding=utf8

import torch
from typing import Dict, Optional
from typing import List
from dataclasses import dataclass


class TransducerSearcher(object):
    def __init__(self, pred_net, joint_net, blank_id,
                 beam_size=2, state_beam=2.3, expand_beam=2.3):
        self.pred_net = pred_net
        self.joint_net = joint_net
        self.blank_id = blank_id
        self.state_beam = state_beam
        self.expand_beam = expand_beam
        self.beam_size = beam_size

    def beam_search(self, encs):
        with torch.no_grad():
            best_match = self.search_single(encs)
        return best_match, None

    def search_single(self, encoder_out):
        device = encoder_out.device
        bs, T = encoder_out.size()[0:2]
        B = self._init_batch_list(bs, encoder_out.dtype, device)
        new_order = torch.arange(bs).view(-1, 1).repeat(1, self.beam_size).view(-1)
        new_order = new_order.to(device).long()
        encoder_out = self.reorder_enc(encoder_out, new_order)
        beam_idxes = self._get_beam_indices(bs, self.beam_size)
        beam_idxes = beam_idxes.to(device)

        for t in range(T):
            A = B
            B = BatchList(bs, device)
            log_probs = self.comp_trans_probs(encoder_out, A, t,  device, beam_idxes)
            self._update_beam(log_probs, A, B, device, bs)

        best_hyps = [hyp_list.get_most_probable(length_norm=True) for hyp_list in B]
        ys_arr = [best_hyp.ys[self.pred_net.context_size:] for best_hyp in best_hyps]
        return ys_arr

    def _init_batch_list(self, bs, data_type, device):
        res = BatchList(bs, device)
        for hyp_list in res:
            hyp = Hypothesis(
                ys=[self.blank_id] * self.pred_net.context_size,
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
            )
            hyp_list.add(hyp)
        return res

    def reorder_enc(self, enc, new_order):
        return enc.index_select(0, new_order)

    def _update_beam(self, log_probs, A, B, device, batch_size):
        voc_size = log_probs.size(-1)
        log_probs = log_probs.reshape(batch_size, -1)
        topk_log_probs, topk_indexes = log_probs.topk(self.beam_size, dim=-1)

        # topk_hyp_indexes are indexes into `A`
        topk_hyp_indexes = topk_indexes // voc_size
        topk_token_indexes = topk_indexes % voc_size

        topk_hyp_indexes = topk_hyp_indexes.tolist()
        topk_token_indexes = topk_token_indexes.tolist()
        for i_batch in range(batch_size):
            k_hyp_idxes = topk_hyp_indexes[i_batch]
            k_token_idxes = topk_token_indexes[i_batch]
            a_hyp_list = A[i_batch]
            b_hyp_list = B[i_batch]
            self._update_beam_single(i_batch, k_hyp_idxes, k_token_idxes,
                                     a_hyp_list, b_hyp_list, topk_log_probs)

    def _update_beam_single(self, i_batch, k_hyp_idxes, k_token_idxes,
                            a_hyp_list, b_hyp_list, topk_log_probs):
        a_hyp_list = list(a_hyp_list)
        for i in range(len(k_hyp_idxes)):
            b_idx = k_hyp_idxes[i]
            hyp = a_hyp_list[b_idx]
            new_ys = hyp.ys[:]
            new_token = k_token_idxes[i]
            if new_token != self.blank_id:
                new_ys.append(new_token)
            new_log_prob = topk_log_probs[i_batch, i]
            new_hyp = Hypothesis(ys=new_ys, log_prob=new_log_prob)
            b_hyp_list.add(new_hyp)

    def comp_trans_probs(self, enc, A, t, device, beam_indices):
        current_encoder_out = enc[:, t:t+1, :]
        # current_encoder_out is of shape (bs * beam, 1, encoder_out_dim)
        len_arr, input_ids, prev_score = A.get_inputs(self.beam_size, self.blank_id, self.pred_net.context_size)
        decoder_out = self.pred_net(input_ids, need_pad=False)
        # decoder_output is of shape (bs * beam, 1, decoder_output_dim)
        logits = self.joint_net(
            current_encoder_out.unsqueeze(2),
            decoder_out.unsqueeze(1)
        )
        # logits is of shape (num_hyps, 1, 1, vocab_size)
        logits = logits.squeeze(1).squeeze(1)

        # now logits is of shape (num_hyps, vocab_size)
        log_probs = logits.log_softmax(dim=-1)
        log_probs.add_(prev_score)
        log_probs = self._mask_pad_probs(log_probs, beam_indices, len_arr)
        return log_probs

    def _mask_pad_probs(self, probs, beam_indices, len_arr):
        mask = beam_indices >= len_arr
        mask = mask.view(-1, 1)
        res = probs.masked_fill(mask, -1e+4)
        return res

    def _get_beam_indices(self, bs, beam):
        x = torch.arange(beam).unsqueeze(0)
        x = x.repeat(bs, 1)
        return x


@dataclass
class Hypothesis:
    # The predicted tokens so far.
    # Newly predicted tokens are appended to `ys`.
    ys: List[int]

    # The log prob of ys
    log_prob: float
    # hid: Any = None

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
            old_hyp.log_prob = torch.logsumexp(
                torch.cat([old_hyp.log_prob.view([1]), hyp.log_prob.view([1])],  dim=0), 0)
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


class BatchList(object):
    def __init__(self, batch_size, device):
        self.bs = batch_size
        self.device = device
        self.batches = []
        for i in range(batch_size):
            self.batches.append(HypothesisList())

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, item):
        return self.batches[item]

    def __iter__(self):
        return iter(self.batches)

    def get_inputs(self, beam_size, blank_id, context_size):
        len_arr = [[len(hyp_list)] for hyp_list in self.batches]
        len_arr = torch.tensor(len_arr, device=self.device)
        paded_hyps = self._pad_batches(beam_size)
        input_ids = self.get_input_ts(paded_hyps, context_size)
        # hids = self.get_hid_ts(paded_hyps)
        prev_scores = self.get_scores(paded_hyps)
        return len_arr, input_ids, prev_scores

    def _pad_batches(self, beam_size):
        res = [
            self._pad_hyp_list(hyp_list, beam_size) for hyp_list in self.batches
        ]
        return res

    def _pad_hyp_list(self, hyp_list, beam_size):
        hyps = list(hyp_list)
        if len(hyps) >= beam_size:
            return hyps[:beam_size]
        hyps = hyps + [hyps[0]] * (beam_size - len(hyps))
        return hyps

    def get_input_ts(self, pad_hyps, context_size):
        list_arr = [
            [hyp.ys[-context_size:] for hyp in hyp_list]
            for hyp_list in pad_hyps
        ]
        input_ids = torch.tensor(list_arr, device=self.device).view(-1, context_size)
        return input_ids

    def get_hid_ts(self, paded_hyps):
        flat_hyps = []
        for hyp_list in paded_hyps:
            flat_hyps.extend(hyp_list)
        return self._init_hids(flat_hyps)

    def _init_hids(self, hyp_list):
        c_list = zip(*[hyp.hid for hyp in hyp_list])
        res = [torch.cat(x, dim=1) for x in c_list]
        res = tuple(res)
        return res

    def get_scores(self, padded_hyps):
        ts_arr = []
        for hyp_list in padded_hyps:
            ts_arr.extend([hyp.log_prob.view([1, 1]) for hyp in hyp_list])
        ys_log_probs = torch.cat(ts_arr)
        return ys_log_probs
