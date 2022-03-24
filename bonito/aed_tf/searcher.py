# coding=utf8

import torch
import torch.nn.functional as F
from typing import Dict, Optional
from typing import List
from dataclasses import dataclass


class DecodeSearcher(object):
    def __init__(self, decoder, pad_id, eos,
                 beam_size=4,
                 max_decode_ratio=0.8,
                 min_decode_ratio=0.0,
                 len_norm=True,
                 **kwargs):
        self.decoder = decoder
        self.pad_id = pad_id
        self.eos = eos
        self.bos = eos
        self.beam_size = beam_size
        self.len_norm = len_norm
        self.max_dec_ratio = max_decode_ratio
        self.min_dec_ratio = min_decode_ratio
        self.minus_inf = float('-inf')

    def beam_search(self, encs):
        with torch.no_grad():
            best_match = self.search_impl(encs)
        return best_match, None

    def search_impl(self, encoder_out):
        device = encoder_out.device
        bs, time_steps = encoder_out.size()[0:2]
        input_tgt, prev_scores, beam_offset, memory, finished_arr, result_dict = self._init_vars(bs, device)
        min_len = int(self.min_dec_ratio * time_steps)
        max_len = int(self.max_dec_ratio * time_steps)
        # init encoder_out
        new_order = torch.arange(bs, device=device, dtype=torch.int)
        new_order = new_order.view(-1, 1).repeat(1, self.beam_size).view(-1)
        # new_order = new_order.to(device).long()
        encoder_out = self.reorder_enc(encoder_out, new_order)

        for t in range(max_len + 1):
            if self._check_finished(finished_arr):
                break
            log_probs, memory = self.decode_step(input_tgt, memory, encoder_out)
            next_inp, top_scores, select_beam_idx \
                = self._get_cand_info(log_probs, prev_scores, beam_offset, min_len, bs, t)
            # update status
            eos_mask = next_inp.eq(self.bos)
            memory = self.reorder_enc(memory, select_beam_idx)
            # update  hyps
            self._update_hyps(memory, eos_mask, top_scores, next_inp, finished_arr, result_dict)
            # update new scores and new input
            prev_scores = top_scores.masked_fill(eos_mask, self.minus_inf)
            input_tgt = next_inp

        # when not finished
        if not self._check_finished(finished_arr):
            eos_mask = torch.ones([self.beam_size * bs, 1], dtype=torch.int, device=device)
            self._update_hyps(memory, eos_mask, prev_scores, input_tgt, finished_arr, result_dict)

        best_hyps = [hyp_list.get_most_probable(length_norm=True) for hyp_list in result_dict]
        ys_arr = [best_hyp.ys for best_hyp in best_hyps]
        return ys_arr

    def _init_vars(self, bs, device):
        """
            解码开始的初始化函数，用来返回初始化的重要变量
            1. input tgt
            2. prev_scores # 就是此前每个beam路径所保存的分数;
            3. memory: 就是每个beam必须要保留的东西， 这一版做的简单就保留了前面的字符，todo 替换成前面的attention等
            4. finished_arr: 用来记录哪些beam 都finished了;
            5. result_dict: 用来保存已经搜索好的路径，
        """
        input_tgt = torch.full([bs * self.beam_size, 1], self.bos, device=device)
        prev_scores = torch.full([bs * self.beam_size, 1], self.minus_inf, device=device)
        beam_offset = torch.arange(bs, device=device) * self.beam_size
        prev_scores.index_fill_(0, beam_offset, 0.)
        memory = None
        finished_arr = [False] * bs
        result_dict = [HypothesisList() for _ in range(bs)]  # key 为batch_id, value 为所有已经搜索好路径的列表;
        return input_tgt, prev_scores, beam_offset.unsqueeze(1), memory, finished_arr, result_dict

    def _check_finished(self, finished_arr):
        return all(finished_arr)

    def decode_step(self, input_tgt, memory, encoder_out):
        memory = self._update_mem(input_tgt, memory)
        logits = self.decoder.decode(memory, encoder_out)
        prob_dist = F.log_softmax(logits, dim=-1)
        return prob_dist[:, -1, :], memory

    def _update_mem(self, inp_tokens, memory):
        if memory is None:
            return inp_tokens.clone()
        return torch.cat([memory, inp_tokens], dim=-1)

    def _get_cand_info(self, log_probs, prev_scores, beam_offset, min_len, bs, t):
        vocab_size = log_probs.shape[-1]
        if t < min_len:
            log_probs[:, self.bos] = self.minus_inf
        scores = prev_scores + log_probs

        if self.len_norm:
            scores = scores / (t+1)
        # now to choose top beam candidates
        scores = scores.view(bs, -1)
        top_scores, candidates = scores.topk(self.beam_size, dim=-1)
        next_inp = (candidates % vocab_size).view(-1, 1)
        beam_num = candidates // vocab_size
        top_scores = top_scores.view(self.beam_size * bs, 1)
        select_beam_idx = (beam_num + beam_offset).view(-1)
        if self.len_norm:
            top_scores = (t+1) * top_scores
        return next_inp, top_scores, select_beam_idx

    def reorder_enc(self, enc, new_order):
        return enc.index_select(0, new_order)

    def _update_hyps(self, memory, eos_mask,
                     top_scores, next_inp,
                     finished_arr, result_dict):
        eos_indices = torch.nonzero(eos_mask, as_tuple=True)[0]
        if eos_indices.shape[0] > 0:
            eos_indices = eos_indices.cpu().numpy()
            for index in eos_indices:
                batch_id = index // self.beam_size
                if finished_arr[batch_id]:
                    continue
                hyplist = result_dict[batch_id]
                ys = memory[index, :].cpu().numpy()
                log_prob = top_scores[index, 0]
                new_hyp = Hypothesis(ys, log_prob)
                hyplist.add(new_hyp)
                if len(hyplist) >= self.beam_size:
                    finished_arr[batch_id] = True


@dataclass
class Hypothesis:
    # The predicted tokens so far.
    # Newly predicted tokens are appended to `ys`.
    ys: List[int]

    # The log prob of ys
    log_prob: float

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

