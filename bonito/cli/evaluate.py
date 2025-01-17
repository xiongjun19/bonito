"""
Bonito model evaluator
"""

import os
import time
import torch
import numpy as np
from itertools import starmap
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

from bonito.data import load_numpy, load_script
from bonito.util import accuracy, poa, decode_ref, half_supported
from bonito.util import init, load_model, concat, permute
from tqdm import tqdm
from torch.utils.data import DataLoader


def main(args):

    poas = []
    init(args.seed, args.device)

    print("* loading data")
    try:
        _, valid_loader_kwargs = load_numpy(args.chunks, args.directory)
    except FileNotFoundError:
        _, valid_loader_kwargs = load_script(
            args.directory,
            seed=args.seed,
            chunks=args.chunks,
            valid_chunks=args.chunks
        )

    dataloader = DataLoader(
        batch_size=args.batchsize, num_workers=4, pin_memory=True,
        **valid_loader_kwargs
    )

    accuracy_with_cov = lambda ref, seq: accuracy(ref, seq, min_coverage=args.min_coverage)
    t_enc = 0
    t_dec = 0

    for w in [int(i) for i in args.weights.split(',')]:

        seqs = []

        print("* loading model", w)

        model = load_model(args.model_directory, args.device, weights=w)

        print("* calling")
        t0 = time.perf_counter()
        
        targets = []

        with torch.no_grad():
            for data, target, *_ in tqdm(dataloader):
                targets.extend(torch.unbind(target, 0))
                if half_supported():
                    data = data.type(torch.float16).to(args.device)
                else:
                    data = data.to(args.device)
                t1 = time.perf_counter()
                log_probs = model(data)
                t_enc += (time.perf_counter() - t1)
                t2 = time.perf_counter()
                if hasattr(model, 'decode_batch'):
                    decs = model.decode_batch(log_probs)
                    print(decs)
                    seqs.extend(decs)
                else:
                    seqs.extend([model.decode(p) for p in permute(log_probs, 'TNC', 'NTC')])
                t_dec += (time.perf_counter() - t2)

        duration = time.perf_counter() - t0

        refs = [decode_ref(target, model.alphabet) for target in targets]
        accuracies = [accuracy_with_cov(ref, seq) if len(seq) else 0. for ref, seq in zip(refs, seqs)]

        if args.poa: poas.append(sequences)

        print("* mean      %.2f%%" % np.mean(accuracies))
        print("* median    %.2f%%" % np.median(accuracies))
        print("* time      %.2f" % duration)
        print("* samples/s %.2E" % (args.chunks * data.shape[2] / duration))
        print(f'enc time is {t_enc}')
        print(f'dec time is {t_dec}')

    if args.poa:

        print("* doing poa")
        t0 = time.perf_counter()
        # group each sequence prediction per model together
        poas = [list(seq) for seq in zip(*poas)]
        consensuses = poa(poas)
        duration = time.perf_counter() - t0
        accuracies = list(starmap(accuracy_with_coverage_filter, zip(references, consensuses)))

        print("* mean      %.2f%%" % np.mean(accuracies))
        print("* median    %.2f%%" % np.median(accuracies))
        print("* time      %.2f" % duration)


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=9, type=int)
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--chunks", default=1000, type=int)
    parser.add_argument("--batchsize", default=96, type=int)
    parser.add_argument("--beamsize", default=5, type=int)
    parser.add_argument("--poa", action="store_true", default=False)
    parser.add_argument("--min-coverage", default=0.5, type=float)
    return parser
