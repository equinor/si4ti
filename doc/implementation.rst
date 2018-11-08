**************
Implementation
**************

Because the memory requirements for solving timeshift grows so fast, a few
techniques are necessary to make the program practical and fast.

.. math::

    \newcommand{\d}{\mathbf{d}}
    \newcommand{\F}{\mathbf{F}}
    \newcommand{\t}{\mathbf{t}}
    \newcommand{\FM}{
        \begin{pmatrix}
            \F_{1,2} & 0        \\
            0        & \F_{2,3} \\
            \F_{1,3} & \F_{1,3} \\
        \end{pmatrix}
    }
    \newcommand{\FMT}{
        \begin{pmatrix}
            \F_{1,2} & 0        & \F_{1,3} \\
            0        & \F_{2,3} & \F_{1,3} \\
        \end{pmatrix}
    }
    \newcommand{Ftilde}{\tilde{\F}}
    \newcommand{FTtilde}{\tilde{\F}^{T}}

Linear operator and masking
===========================

The vintage-pair system
-----------------------
Without lateral smoothing, the vintage-pair system would be a symmetric band
diagonal matrix. With lateral smoothing, you get the non-zero pattern
illustrated in the figure :ref:`sparsity pattern`. The off-diagonal elements in
this matrix are constributions from the lateral smoothing. These contributions
are all consisting of an identical matrices.

Leveraging this, the contributions from lateral smoothing is not explicitly
represented, but stored once, and distributed as a part of a custom
matrix-vector product function.

We are then left with a diagonal, symmetric, banded matrix. Only the upper
non-zero diagonals are stored.

The matrices described here are the matrices :math:`\F_{m,n}` in the next
section.

The multi-vintage system
------------------------
Consider the system :eq:`eq_multivintage`. This is an overdetermined
system, and we're trying to solve it with a least-square approach. Because the
system is impractically large, and with a lot of repretition of sub-matrices,
the main linear operator is not represented directly, but through an addressing
system.

We have computed the sub-matrices :math:`\F`.  Because of the least-square
approach, we multiply both sides of the equation with the matrix

.. math::

    \FMT

and get the system:

.. math::

    \FMT \FM
    \begin{pmatrix}
        \Delta \t_1 \\
        \Delta \t_2 \\
        \Delta \t_3 \\
    \end{pmatrix}
    =
    \FMT
    \begin{pmatrix}
        \Delta \d_{1,2} \\
        \Delta \d_{2,3} \\
        \Delta \d_{1,3} \\
    \end{pmatrix}

or in shorter form:

.. math::

    \Ftilde^{2} \tilde{\Delta t} = \Ftilde^{T} \tilde{\Delta d}

The term :math:`\Ftilde^{2}` can be computed more efficiently as

.. math::

    \FTtilde \Ftilde
    =
    \begin{pmatrix}
        \F_{1,2}^{2} + \F_{1,3}^2 & \F_{1,3}^2                  \\
        \F_{1,3}^{2}              & \F_{2,3}^2 + \F_{1,3}^{2}   \\
    \end{pmatrix}

Only the sub-matrices :math:`\F_{n,m}` are explicitly computed. Their
contributions to the larger system are recorded in mask matrices:

.. math::

    \F_{1,2}^2 \Rightarrow
    \begin{pmatrix}
        1 & 0 \\
        0 & 0 \\
    \end{pmatrix}

    \F_{1,3}^2 \Rightarrow
    \begin{pmatrix}
        1 & 1 \\
        1 & 1 \\
    \end{pmatrix}

    \F_{2,3}^2 \Rightarrow
    \begin{pmatrix}
        0 & 0 \\
        0 & 1 \\
    \end{pmatrix}
