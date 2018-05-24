#! /usr/bin/env python3

import sys
import segyio

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, nargs='+')
    parser.add_argument('--iline', '-i', type=int, default=189)
    parser.add_argument('--xline', '-x', type=int, default=193)
    args = parser.parse_args()

    if len(args.files) % 2 != 0:
        raise ValueError('Must be even number of files (base-reference pair)')

    references = args.files[::2]
    bases = args.files[1::2]

    for base, reference in zip(bases, references):
        with segyio.open(base) as src, segyio.open(reference) as ref:
            xs = src.trace.raw[:]
            ys = ref.trace.raw[:]

            diff = abs(xs - ys)

            s = sum(sum(diff)) / sum(sum(ys))
            if s > 2e-1:
                msg = 'Error in {}, avg too high: {}'
                sys.exit(msg.format(base, s))

            if not diff.max() < 1e-1:
                msg = 'Error in {}, max too high: {}'
                sys.exit(msg.format(base, diff.max()))

if __name__ == '__main__':
    main()
