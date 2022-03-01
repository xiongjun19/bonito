"""
Bonito model onnx exporter 
"""

import os
import time
import torch
from torch import nn
from typing import Optional
from typing import Union 

from itertools import starmap
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

from bonito.data import load_numpy, load_script
from bonito.util import accuracy, poa, decode_ref, half_supported
from bonito.util import init, load_model, concat, permute


def main(args):
    model = load_model(args.model_directory, "cuda",
	               weights=None, half=False)
    model = model.to('cpu')
    model.eval()
    os.makedirs(args.onnx_dir, exist_ok=True)
    out_model_name = os.path.basename(args.model_name)
    onnx_dir = args.onnx_dir
    model_path = f'{onnx_dir}/{out_model_name}.onnx'
    op_set = 11
    in_ids = torch.rand([args.batchsize, 1, args.seq_len], dtype=torch.float32)
    in_ids = in_ids.to("cpu")
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

  
def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument('-o', '--onnx_dir', type=str, help="dir to store onnx file")
    parser.add_argument('-s', '--seq_len', type=int, help="the length of sequence")
    parser.add_argument('-m', '--model_name', type=str, help="the model name of gpt")
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--batchsize", default=8, type=int)
    return parser



if __name__ == '__main__':
    parser = argparser()
    args = parser.parse_args()
    main(args)
