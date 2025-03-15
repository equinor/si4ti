#! /usr/bin/env python3

import numpy.testing as npt
import sys
import segyio

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, nargs='+')
    parser.add_argument('--iline', '-i', type=int, default=189)
    parser.add_argument('--xline', '-x', type=int, default=193)
    parser.add_argument('--reverse', '-r', action="store_true")

    parser.add_argument('--avg', '-a', type=float, default=8e-3)

    parser.add_argument('--max', '-m', type=float, default=4e-4)

    #parser.add_argument('--atol', type=float, default=0)
    #parser.add_argument('--rtol', type=float, default=1e-3)


    args = parser.parse_args()

    if len(args.files) % 2 != 0:
        raise ValueError('Must be even number of files (base-reference pair)')

    references = args.files[::2]
    bases = args.files[1::2]

    for base, reference in zip(bases, references):
        with segyio.open(base, iline=args.iline, xline=args.xline) as f:
            xs = f.trace.raw[:]

        with segyio.open(reference, iline=args.iline, xline=args.xline) as f:
            ys = f.trace.raw[:]

        if args.reverse: ys = -ys

        diff = abs(xs - ys)

        s = abs(sum(sum(diff)) / sum(sum(ys)))

        if s > args.avg:
            msg = 'Error in {}, avg too high: {}'
            sys.exit(msg.format(base, s))

        if not diff.max() < args.max:
            msg = 'Error in {}, max too high: {}'
            sys.exit(msg.format(base, diff.max()))

        #npt.assert_allclose(xs, ys, rtol=args.rtol, atol=args.atol, strict=True)
        #npt.assert_allclose(xs, ys, rtol=args.rtol, strict=True)
        #npt.assert_allclose(xs, ys, rtol=0, atol=args.max, strict=True)

if __name__ == '__main__':
    main()
