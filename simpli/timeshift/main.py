from __future__ import division, print_function, absolute_import
import os
import itertools as itr
import argparse

import numpy as np
import segyio

import simpli.timeshift as ts
import time

from scipy.sparse import lil_matrix
from scipy.sparse import bsr_matrix

def lpos(i, height):
    return slice(i * height, (i + 1) * height)

def smal_system(filenames):
    root = './'
    filenames = [os.path.join(root, fname) for fname in filenames]
    surveys = [segyio.open(fname, iline = 5, xline = 21)
               for fname in filenames
              ]

    vintage1, vintage2 = surveys[0], surveys[1]

    ilines, xlines = list(vintage1.ilines), list(vintage1.xlines)
    jks = list(itr.product(ilines, xlines))
    jmax, kmax = len(ilines), len(xlines)

    tracelen = len(vintage1.samples)
    traces = len(vintage1.trace)
    normalizer = ts.normalize_surveys(surveys)
    B = ts.bspline(tracelen, 0.05, 3)
    constraints = ts.constraints(B, 0.1, 0.01)
    omega = ts.angular_frequency(1, tracelen)

    Lsize = traces * B.shape[0]
    Lt = lil_matrix((Lsize, Lsize), dtype=np.float64)
    b = []

    Bnn = -.25 * B.dot(B.T) * 0.01

    for i in range(len(vintage1.trace)):
        tr1 = vintage1.trace[i]/normalizer
        tr2 = vintage2.trace[i]/normalizer

        D1 = ts.derive(tr1, omega).reshape(tr1.shape)
        D2 = ts.derive(tr2, omega).reshape(tr1.shape)
        D  = 0.5 * (D1 + D2)

        delta = tr2 - tr1

        F = ts.linear_operator(D, B)

        L = F + constraints

        Lpos = lpos(i, B.shape[0])
        Lt[Lpos, Lpos] = L
        b.append(ts.solution(D, delta, B))

        # + horizontal smoothing
        j, k = jks[i]
        j, k = ilines.index(j), xlines.index(k)

        i1 = i - 1 if k != 0 else i
        i2 = i + 1 if k != len(xlines) - 1 else i
        i3 = i - len(xlines) if j != 0 else i
        i4 = i + len(xlines) if j < len(ilines) - 1 else i

        for j, k in enumerate([i1, i2, i3, i4]):
            Lt[lpos(k, B.shape[0]), Lpos] += Bnn

    b = np.concatenate(b)

    return Lt, b, B

def system(filenames):
    n_vintages = len(filenames)

    f = enumerate(filenames)
    combinations = itr.combinations(f, r = 2)

    rows = np.zeros([1])
    cols = np.zeros([1])
    data = np.zeros([1])

    dummy = smal_system([filenames[0], filenames[0]])

    M = dummy[0].shape[0]

    B = dummy[2]

    b_in = np.zeros([(n_vintages-1)*M])

    for vintage1, vintage2 in combinations:
        mask_L = np.zeros([n_vintages-1, n_vintages-1])
        mask_L[vintage1[0]:vintage2[0], vintage1[0]:vintage2[0]] = 1

        mask_b = np.zeros([n_vintages-1])
        mask_b[vintage1[0]:vintage2[0]] = 1

        L, b, _ = smal_system([vintage1[1], vintage2[1]])
        L2 = lil_matrix(L.dot(L.T))
        b2 = L.dot(b)

        r, c = L2.nonzero()
        d = np.concatenate(L2.data)

        for i, row in enumerate(mask_L):
            for j, mask in enumerate(row):
                if mask:
                    rows = np.concatenate([ rows,  r + M*i ])
                    cols = np.concatenate([ cols,  c + M*j ])
                    data = np.concatenate([ data,  d       ])


        for i, mask in enumerate(mask_b):
            if mask: b_in[i*M: i*M + M] += b2


    L_in = bsr_matrix(
                      (data, (rows, cols)),
                      shape = (M*(n_vintages-1), M*(n_vintages-1)),
                      dtype = np.single
                     )

    return L_in, b_in, B

def dump(c, srvs, spline):
    i = 0
    dsts = []
    for srv in srvs:
        with segyio.open(srv, iline=5, xline=21) as f:
            spec = segyio.tools.metadata(f)
            dst = segyio.create('timeshift' + str(i) + '.sgy', spec)
            i += 1
            dst.text[0] = f.text[0]
            dst.bin = f.bin
            dst.header = f.header
            dsts.append(dst)


    scale = dst.samples[1] - dst.samples[0]
    samples = len(dst.samples)
    tracecount = len(dst.trace)
    xlen = spline.shape[0]
    base_offset = 0
    shape = (len(dst.xlines), len(dst.samples))

    for dst in dsts:
        for i in range(dst.tracecount):
            s = slice(base_offset, base_offset + xlen)
            ln = np.asarray(c[s], dtype=np.single)
            ln = np.asarray(scale * spline.T.dot(ln), dtype=np.single)
            dst.trace[i] = ln
            base_offset += xlen
        dst.close()

def solve(L, b):
    import scipy.sparse.linalg as linalg
    print("solving...")

    Lx = L.tocsc()
    M_x = lambda x: linalg.spsolve(Lx, x)
    M = linalg.LinearOperator(Lx.shape, M_x)
    return linalg.cg(L, b, M = M, maxiter = 100)[0]

def main():
    parser = argparse.ArgumentParser('simpli.timeshift')
    parser.add_argument('surveys', type=str, nargs='+')
    parser.add_argument('--outdir', type=str, default=os.getcwd())

    args = parser.parse_args()

    print("surveys:", args.surveys)
    print("outdir:", args.outdir)

    surveys = args.surveys
    L, b, spline = system(surveys)
    c = solve(L, b)
    dump(c, surveys[:-1], spline)

if __name__ == '__main__':
    main()
