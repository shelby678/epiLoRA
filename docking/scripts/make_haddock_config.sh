#!/bin/bash
# Usage: make_haddock_config.sh TEMPLATE AB AG AMBIG UNAMBIG OUT [REF]
set -euo pipefail

TEMPLATE="$1"
AB=$(realpath "$2")
AG=$(realpath "$3")
AMBIG=$(realpath "$4")
UNAMBIG=$(realpath "$5")
OUT="$6"
REF="${7:-}"
RUNDIR=$(realpath "$(dirname "$OUT")")/haddock_out

mkdir -p "$(dirname "$OUT")"

sed \
    -e "s|__AB__|${AB}|g" \
    -e "s|__AG__|${AG}|g" \
    -e "s|__AMBIG__|${AMBIG}|g" \
    -e "s|__UNAMBIG__|${UNAMBIG}|g" \
    -e "s|__RUNDIR__|${RUNDIR}|g" \
    "$TEMPLATE" > "$OUT"

# Optionally insert reference_fname after each [caprieval] line
if [ -n "$REF" ] && [ -f "$REF" ]; then
    REF_ABS=$(realpath "$REF")
    sed -i "s|\[caprieval\]|[caprieval]\nreference_fname = \"${REF_ABS}\"|g" "$OUT"
fi

echo "Wrote $OUT"
