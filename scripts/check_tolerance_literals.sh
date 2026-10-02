#!/bin/bash
# No tolerance literal from 1e-8 to 1e-15 in C++ code outside include/ed/core/numerics.h (pure
# perl over the sources). A threshold that judges an energy must be relative to the scale of H
# (numerics.h: kBreakdownRel, kLockRel, kGsResidRel, kRealBlockRel, scale_or_one(norm_bound()));
# a literal on a scale-free quantity (a unit-vector norm, a character, a phase, a relative
# tolerance, geometry) is allowed when its line, or the line above, carries a comment
# "// scale-free: <why>". Comments and string literals are ignored. Exit 1 on any hit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
hits=$(perl -e '
use strict; use warnings; use File::Find;
my @files;
find(sub { push @files, $File::Find::name if /\.(h|hpp|cpp|cu|cuh)$/ }, "include", "src", "python/qed/_bindings");
for my $f (sort @files) {
    next if $f eq "include/ed/core/numerics.h";
    open(my $fh, "<", $f) or die "$f: $!";
    my ($in_block, $ln, $prev) = (0, 0, "");
    while (my $l = <$fh>) {
        $ln++;
        my $code = $l;
        if ($in_block) { if ($code =~ s{^.*?\*/}{}) { $in_block = 0 } else { $prev = $l; next } }
        $code =~ s{/\*.*?\*/}{}g;
        $in_block = 1 if $code =~ s{/\*.*$}{};
        $code =~ s{//.*$}{};
        $code =~ s{"(?:\\.|[^"\\])*"}{""}g;
        if ($code =~ /(?<![\w.])\d*\.?\d+[eE]-(?:0?8|0?9|1[0-5])(?!\d)/
            && $l !~ m{//\s*scale-free} && $prev !~ m{//\s*scale-free}) {
            chomp(my $t = $l); $t =~ s/^\s+//; print "$f:$ln: $t\n";
        }
        $prev = $l;
    }
}')
status=0
if [ -n "${hits}" ]; then
    echo "TOLERANCE LITERALS (relative to s_H via include/ed/core/numerics.h, or mark '// scale-free: <why>'):"
    echo "${hits}" | sed 's/^/  /'
    status=1
fi
echo "tolerance_literals: status=${status}"
exit ${status}
