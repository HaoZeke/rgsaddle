#!/usr/bin/env bash
# Build and run ci/meson-consumer against this checkout's meson.build.
#
#   ci/meson_consumer.sh subproject|pkgconfig RUNTIME
#
# subproject: the consumer pulls HEAD of this checkout through a
# wrap-git, the way gpridge pins rgsaddle. pkgconfig: this checkout is
# installed to a scratch prefix and the consumer finds
# rgsaddle-headers.pc with fallbacks off. RUNTIME is the cdylib from
# `cargo build --features capi`, which the consumer dlopens. The
# subproject mode reads committed files only.
set -euo pipefail

mode=${1:?subproject or pkgconfig}
runtime=$(realpath "${2:?path to the rgsaddle cdylib}")
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$root/Cargo.toml" | head -n 1)

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp -R "$root/ci/meson-consumer" "$work/consumer"

case $mode in
subproject)
    mkdir "$work/consumer/subprojects"
    cat >"$work/consumer/subprojects/rgsaddle-headers.wrap" <<EOF
[wrap-git]
directory = rgsaddle-headers
url = $root
revision = $(git -C "$root" rev-parse HEAD)

[provide]
dependency_names = rgsaddle-headers, rgsaddle
EOF
    meson setup "$work/build" "$work/consumer" \
        --force-fallback-for=rgsaddle-headers \
        -Dexpect=internal -Drgsaddle_version="$version"
    ;;
pkgconfig)
    meson setup "$work/rgsaddle" "$root" --prefix "$work/prefix"
    meson install -C "$work/rgsaddle"
    PKG_CONFIG_PATH="$work/prefix/share/pkgconfig" \
        meson setup "$work/build" "$work/consumer" \
        --wrap-mode=nofallback \
        -Dexpect=pkgconfig -Drgsaddle_version="$version"
    ;;
*)
    echo "unknown mode $mode" >&2
    exit 2
    ;;
esac

meson compile -C "$work/build"
RGSADDLE_LIBRARY="$runtime" meson test -C "$work/build" --verbose
