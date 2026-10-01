# XDiag side of a twin: the model file written by make_model.py (bonds and the full
# enumerated group, every character 1: k = 0 / A1), n_up = N/2, flip and TR off.
#   julia --project=<xdiag env> xdiag_bench.jl <model.txt> <tag> <out.json>
# Prints progress and writes one JSON record (E0, dim, timings, peak RSS) to <out.json>.
using XDiag
path, tag, out = ARGS[1], ARGS[2], ARGS[3]
lines = readlines(path)
N, nb, ngen, ng = parse.(Int, split(lines[1]))
bonds = [parse.(Int, split(lines[1 + k])) for k in 1:nb]
off = 1 + nb + ngen
perms = [Permutation(parse.(Int, split(lines[off + k])) .+ 1) for k in 1:ng]
t0 = time()
group = PermutationGroup(perms)
irrep = Representation(group, ones(Float64, ng))
block = Spinhalf(N, N ÷ 2, irrep)
t1 = time()
D = length(block)
println("[xdiag $tag] dim=$D real=$(isreal(block)) block build $(round(t1 - t0, digits=1)) s"); flush(stdout)
ops = OpSum()
for b in bonds
    global ops += Op("SdotS", [b[1] + 1, b[2] + 1])
end
vin = randn(D); vout = zeros(D)
tm1 = @elapsed apply(ops, block, vin, block, vout)
tm2 = @elapsed apply(ops, block, vin, block, vout)
println("[xdiag $tag] matvec $(round(tm1, digits=2)) s (first), $(round(tm2, digits=2)) s (second)"); flush(stdout)
t2 = time()
e0 = eigval0(ops, block)
t3 = time()
println("[xdiag $tag] E0=$(round(e0, digits=12)) eigval0 $(round(t3 - t2, digits=1)) s total $(round(t3 - t0, digits=1)) s")
open(out, "w") do f
    print(f, "{\"tag\": \"$tag\", \"E0\": $(repr(e0)), \"dim\": $D, \"block_build_s\": $(t1 - t0), ",
          "\"matvec_s\": $tm2, \"eigval0_s\": $(t3 - t2), \"wall_s\": $(t3 - t0), ",
          "\"peak_rss_gib\": $(Sys.maxrss() / 2^30), \"julia_threads\": $(Threads.nthreads()), ",
          "\"omp_threads\": \"$(get(ENV, "OMP_NUM_THREADS", ""))\"}\n")
end
