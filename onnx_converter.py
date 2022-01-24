# coding=utf8

import os
import torch
from torch import nn
from typing import Optional
from typing import Union 


def test(model_path, onnx_dir, seq_len=None):
   model = torch.load(model_path)
   model.eval()
   os.makedirs(onnx_dir, exist_ok=True)
   out_model_name = os.path.basename(model_path)
   model_path = f'{onnx_dir}/{out_model_name}.onnx'
   op_set = 11

   dyn_state = False
   if seq_len is None:
       seq_len = 256
       dyn_state = True

   in_ids = torch.rand([64, 1, seq_len], dtype=torch.float32)
   in_ids = in_ids.to("cuda")
   print(in_ids.dtype)
   if dyn_state:
       torch.onnx.export(
           model,
           (in_ids, att_mask, type_ids),
           model_path,
           export_params=True,
           opset_version=op_set,
           do_constant_folding=True,
           input_names=['input_ids', 'att_mask', 'type_ids'],
           output_names =['enc_out'],
           use_external_data_format=True,
           dynamic_axes = {
               'input_ids': {0: 'batch_size', 1: 'seq_len'},
               'att_mask': {0: 'batch_size', 1: 'seq_len'},
               'type_ids': {0: 'batch_size', 1: 'seq_len'},
               'enc_out': {0: 'batch_size', 1: 'seq_len'},
               }
        )
   else:
        torch.onnx.export(
           model,
           (in_ids,),
           model_path,
           export_params=True,
           opset_version=op_set,
           do_constant_folding=True,
           input_names=['x'],
           output_names =['enc_out'],
           use_external_data_format=False
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, help="dir to store onnx file")
    parser.add_argument('-m', '--model_name', type=str, help="the model name of gpt")
    parser.add_argument('-s', '--seq_len', type=int, help="the model name of gpt")
    args = parser.parse_args()
    test(args.model_name, args.output, args.seq_len)
