# coding=utf8

import torch
from torch.nn import functional as F


class TransducerSearcher(object):
    def __init__(self, pred_net, joint_net, blank_id,
                 beam_size=3, state_beam=2.3, expand_beam=2.3):
        self.pred_net = pred_net
        self.joint_net = joint_net
        self.blank_id = blank_id
        self.state_beam = state_beam
        self.expand_beam = expand_beam
        self.beam_size = beam_size

    def set_eval(self):
        self.pred_net.eval()
        self.joint_net.eval()

    def beam_search(self, encs):
        n_best_match = []
        n_match_score = []
        B = encs.size()[0]
        with torch.no_grad():
            for i in range(B):
                best_match, match_score = self.search_single(encs, i)
                n_best_match.append(best_match)
                n_match_score.append(match_score)
        return n_best_match, n_match_score
   
    def print_info(self, hyps):
        len_arr = [str(len(hyp['prediction'])) for hyp in hyps]
        print("\t".join(len_arr))

    def search_single(self, encs, i_rec):
        # prepare
        hyp = {
            'prediction': [self.blank_id],
            'log_score': 0.0,
            'hidden': None,
        }
        beam_hyps = [hyp]
        for t in range(encs.size(1)):
            # print(f"hyps len info a timestep is : {t} is: ")
            # self.print_info(beam_hyps)
            process_hyps = beam_hyps
            beam_hyps = []
            while True:
                if len(beam_hyps) >= self.beam_size:
                    break
                if len(process_hyps) < 1:
                    break
                a_best_hyp = max(process_hyps, key=lambda x: x['log_score'] / len(x['prediction']))
                if self.check_state_break(process_hyps, beam_hyps, a_best_hyp):
                    break
                process_hyps.remove(a_best_hyp)
                trans_probs, new_hid = self.comp_trans_prob(encs, a_best_hyp, i_rec, t)
                if len(a_best_hyp['prediction']) >= t + 2:
                    new_hyp = {
                       'prediction': a_best_hyp['prediction'].copy(),
                       'log_score':  a_best_hyp['log_score'] + trans_probs.view(-1)[self.blank_id],
                       'hidden': a_best_hyp['hidden'],
                     }
                    beam_hyps.append(new_hyp)
                    continue
                target_probs, pos_arr = torch.topk(trans_probs.view(-1), k=self.beam_size)
                best_prob = target_probs[0] if pos_arr[0] != self.blank_id  else target_probs[1]
                self.extend_search(beam_hyps, process_hyps, a_best_hyp, best_prob, target_probs, pos_arr, new_hid)
        res_hyps = sorted(beam_hyps, key=lambda x: x['log_score'] / len(x['prediction']), reverse=True)[0]
        return res_hyps['prediction'][1:],  res_hyps['log_score'] / len(res_hyps['prediction'])

    def extend_search(self,  beam_hyps, process_hyps, a_best_hyp,  best_prob, target_probs, pos_arr, new_hid):
        for k, target_prob in enumerate(target_probs):
            new_hyp = {
                'prediction': a_best_hyp['prediction'].copy(),
                'log_score':  a_best_hyp['log_score'] + target_prob,
                'hidden': a_best_hyp['hidden'],
            }
            if pos_arr[k] == self.blank_id:
                beam_hyps.append(new_hyp)
            elif target_prob >= best_prob - self.expand_beam:
                new_hyp['prediction'].append(pos_arr[k].item())
                new_hyp['hidden'] = new_hid
                process_hyps.append(new_hyp)

    def comp_trans_prob(self, encs, cur_hyp, i_rec, t):
        enc = encs[i_rec, t]
        new_enc = enc.unsqueeze(0).unsqueeze(0)
        cur_y = cur_hyp['prediction'][-1]
        targets = torch.full([1, 1], cur_y, device=encs.device, dtype=torch.int32)
        preds, new_hid = self.pred_net(targets, cur_hyp['hidden'])
        joint_logit = self.joint_net(new_enc, preds)
        probs = F.log_softmax(joint_logit, dim=-1)
        return probs, new_hid

    def check_state_break(self, process_hyps, beam_hyps, a_best_hyp):
        if len(beam_hyps) > 0:
            b_best_hyp = max(beam_hyps, key=lambda x: x['log_score'] / len(x['prediction']))
            a_best_prob = a_best_hyp['log_score']
            b_best_prob = b_best_hyp['log_score']
            if b_best_prob >= a_best_prob + self.state_beam:
                return True
        return False
