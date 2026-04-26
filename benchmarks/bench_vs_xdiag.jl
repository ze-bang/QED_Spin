#!/usr/bin/env julia
# benchmarks/bench_vs_xdiag.jl
#
# Time the same 1D Heisenberg PBC chain workload that
# `benchmarks/bench_vs_quspin.py` and `benchmarks/bench_all_backends.py`
# benchmark on this codebase, but using XDiag.jl as the peer.
#
# For each requested system size we measure:
#
#   * Kernel A: one matrix-vector product (`apply`) on a unit-norm
#     random vector. Reported in microseconds.
#   * Kernel B: the full ground-state Lanczos (`eigval0`) to numerical
#     convergence. Reported in milliseconds, plus the eigenvalue.
#
# Results are written as JSON so the Python orchestrator
# (`benchmarks/bench_vs_xdiag.py`) can splice them next to the
# quantum_ed numbers.
#
# Usage (driven by the orchestrator; can also be run standalone):
#
#   julia --project=benchmarks/xdiag_env benchmarks/bench_vs_xdiag.jl \
#         --sizes 12,14,16,18 --threads 16 --output xdiag_results.json
#
# Requires:
#   * Julia >= 1.9
#   * XDiag.jl (installed once via:
#       julia --project=benchmarks/xdiag_env -e 'using Pkg; Pkg.add("XDiag"); Pkg.add("JSON")')

using XDiag
using JSON
using Printf
using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# Argument parsing (no Argparse dep -- we keep this trivial)
# ---------------------------------------------------------------------------

function parse_args()
    sizes  = [12, 14, 16, 18]
    output = "xdiag_results.json"
    threads = Threads.nthreads()
    fixed_sz = false
    n_apply = 5
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--sizes"
            sizes = parse.(Int, split(ARGS[i+1], ","))
            i += 2
        elseif a == "--output"
            output = ARGS[i+1]
            i += 2
        elseif a == "--threads"
            threads = parse(Int, ARGS[i+1])
            i += 2
        elseif a == "--n-apply"
            n_apply = parse(Int, ARGS[i+1])
            i += 2
        elseif a == "--fixed-sz"
            fixed_sz = true
            i += 1
        else
            error("Unknown arg: $a")
        end
    end
    return (; sizes, output, threads, fixed_sz, n_apply)
end


# ---------------------------------------------------------------------------
# Build the XDiag operator + block for a 1D Heisenberg PBC chain.
#
# We use the SdotS one-line shortcut so XDiag picks the optimal kernel; this
# matches what `quantum_ed.input.HamiltonianBuilder.heisenberg(...)` does.
# Setting `fixed_sz = false` selects the **full** Hilbert space (Spinhalf(N)),
# matching the no-symmetry, no-conservation runs we use as the reference for
# QuSpin / scipy / the CPU `Operator` path. With `--fixed-sz` we instead use
# Spinhalf(N, N÷2) which is the fairer comparison for our `FixedSzOperator`.
# ---------------------------------------------------------------------------

function build_chain(N::Int; fixed_sz::Bool=false)
    block = fixed_sz ? Spinhalf(N, N ÷ 2) : Spinhalf(N)
    ops = OpSum()
    for i in 1:N
        ops += "J" * Op("SdotS", [i, mod1(i + 1, N)])
    end
    ops["J"] = 1.0
    return ops, block
end


# ---------------------------------------------------------------------------
# Time one matrix-vector product. XDiag exposes `apply(ops, block, v)`
# which returns ops*v and is the routine every iterative solver calls
# under the hood.
#
# We use the median of `n_calls` timings, after one warmup. The vector is
# normalised so units match QuSpin / scipy / Operator::apply.
# ---------------------------------------------------------------------------

function time_apply(ops, block; n_calls::Int=5)
    dim = size(block)
    Random.seed!(42)
    v = randn(Float64, dim)
    v ./= norm(v)
    psi = State(block, v)
    apply(ops, psi)                         # warmup (compiles + basis prep)
    samples = zeros(Float64, n_calls)
    for k in 1:n_calls
        t0 = time_ns()
        out = apply(ops, psi)
        samples[k] = (time_ns() - t0) / 1e3 # microseconds
        # Touch the result so the optimiser can't drop the call.
        _ = sum(vector(out))
    end
    sort!(samples)
    return samples[div(n_calls + 1, 2)]     # median in us
end


# ---------------------------------------------------------------------------
# Time the ground-state computation. XDiag's `eigval0` is the canonical
# ground-state Lanczos entry point; under the hood it runs an iterative
# Lanczos with full reorthogonalisation, mirroring `quantum_ed.lanczos`.
#
# We do 1 warmup call (absorbs JIT + basis prep) then `n_calls` timed
# calls; the median is reported. A single timed call is very noisy at
# moderate N because it's dominated by transient effects (the first
# call after warmup still pays for some pool/cache faults), and a
# misleading "fast" number can come out of a single hot cache run. The
# median of 3 is what every other peer in this repo's benchmark suite
# reports.
# ---------------------------------------------------------------------------

function time_eigval0(ops, block; n_calls::Int=3)
    e0_warm = eigval0(ops, block)        # warmup -- absorbs basis prep + JIT
    samples = zeros(Float64, n_calls)
    e0 = e0_warm
    for k in 1:n_calls
        t0 = time_ns()
        e0 = eigval0(ops, block)
        samples[k] = (time_ns() - t0) / 1e6   # milliseconds
    end
    sort!(samples)
    return samples[div(n_calls + 1, 2)], float(e0)   # median in ms
end


function main()
    args = parse_args()

    @info "XDiag benchmark" sizes=args.sizes threads=args.threads fixed_sz=args.fixed_sz

    rows = Vector{Dict{String,Any}}()
    for N in args.sizes
        @info "N=$N starting..."
        try
            ops, block = build_chain(N; fixed_sz=args.fixed_sz)
            dim = size(block)

            t_apply_us = time_apply(ops, block; n_calls=args.n_apply)
            t_lanczos_ms, e0 = time_eigval0(ops, block)

            push!(rows, Dict(
                "N"             => N,
                "dim"           => dim,
                "fixed_sz"      => args.fixed_sz,
                "apply_us"      => t_apply_us,
                "lanczos_ms"    => t_lanczos_ms,
                "e0"            => e0,
                "ok"            => true,
                "error"         => nothing,
            ))
            @info "N=$N done" dim=dim apply_us=t_apply_us lanczos_ms=t_lanczos_ms e0=e0
        catch err
            msg = sprint(showerror, err)
            @warn "N=$N failed: $msg"
            push!(rows, Dict(
                "N"             => N,
                "dim"           => fixed_sz_dim(N, args.fixed_sz),
                "fixed_sz"      => args.fixed_sz,
                "apply_us"      => nothing,
                "lanczos_ms"    => nothing,
                "e0"            => nothing,
                "ok"            => false,
                "error"         => msg,
            ))
        end
    end

    payload = Dict(
        "peer"     => "xdiag",
        "version"  => string(pkgversion(XDiag)),
        "threads"  => args.threads,
        "fixed_sz" => args.fixed_sz,
        "sizes"    => args.sizes,
        "rows"     => rows,
    )
    open(args.output, "w") do io
        JSON.print(io, payload, 2)
    end
    @info "wrote $(args.output)"
end


# Helper: combinatorial dimension when the run errors before block is built.
function fixed_sz_dim(N::Int, fixed_sz::Bool)
    if fixed_sz
        # binomial(N, N÷2) without overflow for moderate N
        return Int(binomial(N, N ÷ 2))
    else
        return 1 << N
    end
end


main()
