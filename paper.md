---
title: 'ExaPF.jl: A Power Flow Solver for GPUs'
tags:
  - Julia
authors:
  - name: Adrian M. Price-Whelan^[Custom footnotes for e.g. denoting who the corresponding author is can be included like this.]
    orcid: 0000-0003-0872-7098
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    affiliation: 2
  - name: Author with no affiliation
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University
   index: 1
 - name: Institution Name
   index: 2
 - name: Independent Researcher
   index: 3
date: 30 September 2020
bibliography: presentation.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Solving optimal power flow is an important tool in the secure and cost
effective operation of the transmission power grids. `ExaPF.jl` aims to
implement a reduced method for solving the optimal power flow problem (OPF)
fully on GPUs. Reduced methods enforce the constraints, represented here by
the power flow's (PF) system of nonlinear equations, separately at each
iteration of the optimization in the reduced space. This paper describes the
API of `ExaPF.jl` for solving the power flow's nonlinear equations entirely on the GPU.
This includes the computation of the derivatives using automatic
differentiation, an iterative linear solver with a preconditioner, and a
Newton-Raphson implementation. All of these steps allow us to run the main
computational loop entirely on the GPU with no transfer from host to device.

This implementation will serve as the basis for the future OPF implementation
in the reduced space.

# Statement of Need 

The current state-of-the-art for solving optimal power flow is the
interior-point method (IPM) in optimization implemented by the solver Ipopt
[@wachter2004implementation] and is the algorithm of reference
implementations like MATPOWER [@matpower]. However, its reliance on
unstructured sparse indefinite inertia revealing direct linear solvers makes
this algorithm hard to port to GPUs. `ExaPF.jl` aims at applying a reduced
gradient method to tackle this problem, which allows us to leverage iterative
linear solvers for solving the PF.

Our final goal is a reduced method optimization solver that provides a
flexible API for models and formulations outside of the domain of OPF.

# Components

To make our implementation portable to CPU and GPU architectures we leverage
two abstractions: arrays and kernels. Both of these abstractions are
supported through the packages `CUDA.jl` [@besard2018juliagpu; @besard2019prototyping] and `KernelAbstractions.jl`

## AutoDiff

Given a set of equations `F(x) = 0`, the Newton-Raphson algorithm for
solving nonlinear equations (see below) requires the Jacobian `J = jacobian(x)` 
of `F`. At each iteration a new step `dx` is computed by
solving a linear system. In our case `J` is sparse and indefinite.

```julia
  go = true
  while(go)
    dx .= jacobian(x)\f(x)
    x  .= x .- dx
    go = norm(f(x)) < tol ? true : false
  end
```
There are two modes of differentiation called *forward/tangent* or
*reverse/adjoint*. The latter is known in machine learning as
*backpropagation*. The forward mode generates Jacobian-vector product code
`tgt(x,d) = J(x) * d`, while the adjoint mode generates code for the
transposed Jacobian-vector product `adj(x,y) = (J(x)'*y)`. We recommend
@griewank2008evaluating for a more in-depth introduction to automatic
differentiation. The computational complexity of both models favors the
adjoint mode if the number of outputs of `F` is much smaller than the
number of inputs `size(x) >> size(F)`, like for example the loss functions
in machine learning. However, in our case `F` is a multivariate vector
function from $\mathbb{R}^n$ to $\mathbb{R}^n$, where $n$ is the number of
buses.
\newcommand{\bigo}[1]{\mathcal{O}\left( #1 \right)}

![Jacobian coloring \label{fig:coloring}](figures/compression.png)

<!-- To avoid a complexity of $\bigo{n} \cdot cost(F)$ by letting the tangent mode -->
run over all Cartesian basis vectors of $\mathbb{R}^n$, we apply the technique of Jacobian
coloring to compress the sparse Jacobian `J`. Running the tangent mode, it
allows to compute columns of the Jacobian concurrently, by combining
independent columns in one Jacobian-vector evaluation (see
\autoref{fig:coloring}). For sparsity detection we rely on the greedy
algorithm implemented by `SparseDiffTools.jl` 
[@sparsedifftools].

# Remarks

# References