#!/bin/sh
set -eu

usage() {
	printf 'Usage: %s DIRECTORY\n' "${0##*/}"
	printf '  Combine all images from each frame into one\n'
	printf '  and place it in DIRECTORY_montage\n'
}

frame_number() {
	printf '%s\n' "$1" | sed -E 's/^.*([[:digit:]]{5}).png/\1/' | bc
}



if [ "$#" -ne 1 ]; then
	usage
	exit 2
fi

dir="${1%%/}"
out_dir="$dir"_montage

depth_renders=$(find "$dir" -name '*depth_*.png' | sort -n)
mkdir -p "$out_dir"

for depth in $depth_renders; do
	rgba=$(printf '%s\n' "$depth" | sed 's/depth/rgba/')
	segm=$(printf '%s\n' "$depth" | sed 's/depth/segm/')
	instance=$(printf '%s\n' "$depth" | sed 's/depth/instance/')
	class=$(printf '%s\n' "$depth" | sed 's/depth/class/')
	raycast=$(printf '%s\n' "$depth" | sed 's/depth/raycast/')
	volume=$(printf '%s\n' "$depth" | sed 's/depth/volume/')
	volume_aabb=$(printf '%s\n' "$depth" | sed 's/depth/volume_aabb/')
	volume_color=$(printf '%s\n' "$depth" | sed 's/depth/volume_color/')

	n=$(frame_number "$depth")
	out="$out_dir/render_$(printf '%05d' "$n").png"

	printf 'montage -label %%t -font Liberation-Mono %s %s %s %s %s %s %s %s %s -geometry +2+2 -tile 3x3 %s\n' \
		"$rgba" "$segm" "$depth" \
		"$instance" "$class" "$raycast" \
		"$volume" "$volume_aabb" "$volume_color" \
		"$out"
done | parallel

