from __future__ import division, print_function, absolute_import
import os
import itertools as itr

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
    L_01, b_01, B = smal_system([filenames[0], filenames[1]])
    L_12, b_12, _ = smal_system([filenames[1], filenames[2]])
    L_02, b_02, _ = smal_system([filenames[0], filenames[2]])

    M = L_01.shape[0]
    N = L_01.shape[1]

    rows = np.concatenate([L_01.nonzero()[0],
                                    L_12.nonzero()[0]+M,
                                    L_02.nonzero()[0]+M,
                                    L_02.nonzero()[0],
                                    ])
    cols = np.concatenate([
                    L_01.nonzero()[1],
                    L_12.nonzero()[1]+N,
                    L_02.nonzero()[1]+N*2,
                    L_02.nonzero()[1]+N*2,
                 ])

    data = np.concatenate(np.concatenate([L_01.data,L_12.data,L_02.data,L_02.data]))

    L = bsr_matrix((data, (rows, cols)),
                   shape = (M*2, N*3), dtype=np.float64
                   )

    b = np.concatenate([b_01,b_12,b_02])

    return L, b, B

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

    Lx = L.dot(L.T).tocsc()
    M_x = lambda x: linalg.spsolve(Lx, x)
    M = linalg.LinearOperator(Lx.shape, M_x)
    return linalg.cg(Lx, L.dot(b), M = M, maxiter = 100)[0]

if __name__ == '__main__':
    surveys = [
        '92NmoUpd_8-16stkEps_985_1281-cropped.sgy',
        '01NmoUpd_8-16stkEps_985_1281-cropped.sgy',
        '06NmoUpd_8-16stkEps_985_1281-cropped.sgy',
    ]
    L, b, spline = system(surveys)
    c = solve(L, b)
    dump(c, surveys[0:2], spline)
