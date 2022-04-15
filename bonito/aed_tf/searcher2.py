# coding=utf8

import numpy as np
import torch
import math
import os
import subprocess
import shlex


class DecodeSearcher(object):
    def __init__(self, decoder, pad_id, eos,
                 vocab_size,
                 beam_size=4,
                 max_decode_ratio=0.7,
                 min_decode_ratio=0.0,
                 len_norm=True,
                 **kwargs):
        self.vocab_size = vocab_size
        self.decoder = decoder
        self.pad_id = pad_id
        self.eos = eos
        self.bos = eos
        self.beam_size = beam_size
        self.len_norm = len_norm
        self.max_dec_ratio = max_decode_ratio
        self.min_dec_ratio = min_decode_ratio
        self.minus_inf = float('-inf')
        self.layer_num = kwargs.get('dec_layer_num', 3)
        self.args = kwargs
        self.ths_path = kwargs['ths_path']
        self.max_step_for_pe = 2500
        self.hidden_dim = kwargs['hid_dim']
        self.init_sate = False
        # self.decoding = self._init_ft_decoding()
        print("after first init")
        self.max_len = int(720 * self.max_dec_ratio) 

    def beam_search(self, encs):
        with torch.no_grad():
            best_match = self.search_impl(encs)
            # with open("test_search2.json", 'w') as out_:
            #     for arr_ in best_match:
            #         num = len(arr_)
            #         str_ = str(num) + "\t" + ",".join(map(str, arr_)) + "\n"
            #         out_.write(str_)
        return best_match, None

    def search_impl(self, encoder_out):
        with open('test_enc_out.txt', 'w') as out_:
            out_.write(f'{encoder_out}\n')
        device = encoder_out.device
        bs, time_steps = encoder_out.size()[0:2]
        max_len = int(self.max_dec_ratio * time_steps)
        # init encoder_out
        new_order = torch.arange(bs, device=device, dtype=torch.int)
        new_order = new_order.view(-1, 1).repeat(1, self.beam_size).view(-1)
        encoder_out = self.reorder_enc(encoder_out, new_order)
        # encoder_out = encoder_out.float()
        src_lengths = torch.full([bs * self.beam_size], time_steps, device=device,
                                 dtype=torch.int)
        # print(bs)
        # print(self.beam_size)
        # print(max_len)
        # print(encoder_out.shape)
        # print(encoder_out.dtype)
        # print(src_lengths.shape)
        # print(src_lengths.dtype)
        output_ids, parent_ids, out_seq_lens = \
            self.decoding.forward(bs, self.beam_size, max_len,
                                  encoder_out, src_lengths)
        parent_ids = parent_ids % self.beam_size
        beams = self.finalize(output_ids, parent_ids, out_seq_lens,
                              self.eos, max_len)
        beams = beams[:, :, 0].cpu().numpy()
        
        ys_arr = [
            self._remove_tail(hyps) for hyps in beams
        ]
        return ys_arr

    def _remove_tail(self, _arr):
        idx_arr = np.where(_arr == self.eos)[0]
        if len(idx_arr) < 1:
            return _arr
        return _arr[:idx_arr[0]]
        


    def reorder_enc(self, enc, new_order):
        return enc.index_select(0, new_order)

    def finalize(self, output_ids, parent_ids, out_seq_lens,
                 end_id, max_seq_len):
        beam_size = self.beam_size
        out_seq_lens = torch.reshape(out_seq_lens, (-1, beam_size))
        max_lens = torch.max(out_seq_lens, 1)[0]
        if max_seq_len:
            shape = (max_seq_len, -1, beam_size)
        else:
            shape = (torch.max(max_lens), -1, beam_size)
        output_ids = torch.reshape(output_ids, shape)
        # with open("test_output_ids.txt", 'w') as out_:
        #     str_ = f"{output_ids}\n"
        #     out_.write(str_)
        parent_ids = torch.reshape(parent_ids, shape)
        # torch.classes.load_library(args.ths_path)
        ids = torch.ops.fastertransformer.gather_tree(output_ids.to(torch.int32), parent_ids.to(torch.int32), max_lens.to(torch.int32), end_id)
        ids = torch.einsum('ijk->jik', ids)
        return ids

    def prep_dec(self, bs, is_half):
        head_num = self.args.get('head_num')
        hidden_dim = self.args.get('hid_dim')
        head_size = hidden_dim // head_num
        layer_num = self.layer_num
        status = 1 if is_half else 0
        cmd_str = f"/workspace/FasterTransformer/build/bin/decoding_gemm {bs} {self.beam_size} {head_num} {head_size} {self.vocab_size} {self.max_len} {self.hidden_dim} {status}"
        print("running config: ", cmd_str)
        cmd_args = shlex.split(cmd_str)
        subprocess.call(cmd_args)    
        print("finished config")
        self.decoding = self._init_ft_decoding()

    def _init_ft_decoding(self):
        w = self._init_weights()
        # f_name = 'first_weights.txt' if not self.init_sate else "sec_weigts.txt"
        # with open(f_name, 'w') as out_:
        #     for i, a_w in enumerate(w):
        #         shape = a_w.shape
        #         dtype = a_w.dtype
        #         str_ = f'{i}th is: shape:{shape}, type: {dtype}\n'
        #         out_.write(str_)
        #         str_ = f'{a_w}\n'
        #         out_.write(str_)
                 
        w = [x.cuda() for x in w]
        torch.classes.load_library(os.path.abspath(self.ths_path))
        head_num = self.args.get('head_num')
        hidden_dim = self.args.get('hid_dim')
        head_size = hidden_dim // head_num
        layer_num = self.layer_num
        self.init_sate = True
        try:
            decoding = torch.classes.FasterTransformer.Decoding(head_num, head_size, hidden_dim, layer_num,
                                                                self.vocab_size, self.eos, self.eos,
                                                                0.0, *w)
            return decoding
        except:
            decoding = torch.classes.FasterTransformerDecoding(head_num, head_size, hidden_dim, layer_num,
                                                               self.vocab_size, self.eos, self.eos,
                                                               0.0, *w)
            return decoding

    def _init_weights(self):
        _dict = self.decoder.state_dict()
        pre1 = ''
        dec_pref = 'decoder.layers'
        w = list()
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'norm1.norm.weight'])]
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'norm1.norm.bias'])]
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'self_attn.att.in_proj_weight'])][:512].transpose(-1, -2)
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'self_attn.att.in_proj_weight'])][512:1024].transpose(-1, -2)
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'self_attn.att.in_proj_weight'])][1024:].transpose(-1, -2)
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'self_attn.att.in_proj_bias'])][:512]
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'self_attn.att.in_proj_bias'])][512:1024]
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'self_attn.att.in_proj_bias'])][1024:]
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'self_attn.att.out_proj.weight'])].transpose(-1, -2)
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'self_attn.att.out_proj.bias'])]
                 for i in range(self.layer_num)],
                0).contiguous()
        )

        # cross att
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'norm2.norm.weight'])]
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'norm2.norm.bias'])]
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'mutihead_attn.att.in_proj_weight'])][:512].transpose(-1, -2)
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'mutihead_attn.att.in_proj_weight'])][512:1024].transpose(-1, -2)
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'mutihead_attn.att.in_proj_weight'])][1024:].transpose(-1, -2)
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'mutihead_attn.att.in_proj_bias'])][:512]
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'mutihead_attn.att.in_proj_bias'])][512:1024]
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'mutihead_attn.att.in_proj_bias'])][1024:]
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'mutihead_attn.att.out_proj.weight'])].transpose(-1, -2)
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'mutihead_attn.att.out_proj.bias'])]
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        # ffn
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'norm3.norm.weight'])]
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'norm3.norm.bias'])]
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'pos_ffn.ffn.0.weight'])].transpose(-1, -2)
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'pos_ffn.ffn.0.bias'])]
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'pos_ffn.ffn.3.weight'])].transpose(-1, -2)
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        w.append(
            torch.stack(
                [_dict[self._join_str([pre1, dec_pref, str(i), 'pos_ffn.ffn.3.bias'])]
                 for i in range(self.layer_num)],
                0).contiguous()
        )
        # final layernorm
        w.append(
            _dict[self._join_str([pre1, 'decoder.norm.norm.weight'])]
        )
        w.append(
            _dict[self._join_str([pre1, 'decoder.norm.norm.bias'])]
        )
        w.append(
            _dict[self._join_str([pre1, 'emb.weight'])]
        )
        pe_enc = self._get_position_encoding()
        pe_enc = pe_enc.type(w[-1].dtype)
        w.append(pe_enc)  # pe_encoding
        # final linear
        w.append(
            _dict[self._join_str([pre1, 'linear.weight'])].transpose(-1, -2).contiguous()
        )
        w.append(
            _dict[self._join_str([pre1, 'linear.bias'])]
        )
        return w

    def _join_str(self, str_arr):
        tmp_str_arr = [_str for _str in str_arr if len(_str) > 0]
        return ".".join(tmp_str_arr)

    def _get_position_encoding(self):
        pe = torch.zeros(self.max_step_for_pe, self.hidden_dim)
        position = torch.arange(0, self.max_step_for_pe).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.hidden_dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / self.hidden_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe
