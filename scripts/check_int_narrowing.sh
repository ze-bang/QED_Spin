#!/bin/bash
# No silent narrowing of a size to int (pure perl over the sources). A cast to int / int32 of a
# dimension (dim, n, nb, local_n, rows(), cols(), reps.size(), ...) wraps once a sector
# passes 2^31 states. Use ed::core::checked_narrow<T>(x, what) (<ed/core/numerics.h>), which throws
# instead; a cast whose operand is bounded by construction (a group order, a site count, a block
# the dense path caps) is allowed when its line, or the line above, carries a comment
# "// narrow-ok: <why>". Comments and string literals are ignored. Exit 1 on any hit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
hits=$(perl -e '
use strict; use warnings; use File::Find;
my @files;
find(sub { push @files, $File::Find::name if /\.(h|hpp|cpp|cu|cuh)$/ }, "include", "src", "python/qed/_bindings");
my $size = qr/\b(?:dim\w*|n|nb|local_n|n_rows|n_cols|n_states)\b|\brows\(\)|\bcols\(\)|\breps\.size\(\)/;
for my $f (sort @files) {
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
        my $hit = 0;
        while ($code =~ /static_cast<\s*(?:int|std::int32_t|int32_t)\s*>\s*(\((?:[^()]++|(?-1))*\))/g) {
            my $arg = $1;
            $hit = 1 if $arg =~ $size;
        }
        if ($hit && $l !~ m{//\s*narrow-ok} && $prev !~ m{//\s*narrow-ok}) {
            chomp(my $t = $l); $t =~ s/^\s+//; print "$f:$ln: $t\n";
        }
        $prev = $l;
    }
}')
status=0
if [ -n "${hits}" ]; then
    echo "INT NARROWING (use ed::core::checked_narrow, or mark '// narrow-ok: <why>'):"
    echo "${hits}" | sed 's/^/  /'
    status=1
fi
echo "int_narrowing: status=${status}"
exit ${status}
