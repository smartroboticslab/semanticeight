#!/bin/sh
set -eu

usage() {
	printf 'Usage: %s DIRECTORY\n' "${0##*/}"
	printf '  Combine all images from each frame into one\n'
	printf '  and place it in DIRECTORY_montage\n'
}

frame_number() {
	printf '%s\n' "$1" | sed -E 's/^.*planning_([[:digit:]]{5})_.*\.png/\1/' | bc
}

null_if_not_file() {
	f=$(cat)
	if [ -f "$f" ]; then
		printf '%s\n' "$f"
	else
		printf 'null:\n'
	fi
}



if [ "$#" -ne 1 ]; then
	usage
	exit 2
fi

dir="${1%%/}"
out_dir="$dir"_montage
mkdir -p "$out_dir"

depth_renders=$(find "$dir" -type f -name '*_depth.png' | sort -n)
for depth in $depth_renders; do
	n=$(frame_number "$depth")
	entropy=$(printf '%s\n' "$depth" | sed 's/depth/entropy/' | null_if_not_file)
	bg_gain=$(printf '%s\n' "$depth" | sed 's/depth/bg_gain/' | null_if_not_file)
	object_gain=$(printf '%s\n' "$depth" | sed 's/depth/object_gain/' | null_if_not_file)
	object_dist_gain=$(printf '%s\n' "$depth" | sed 's/depth/object_dist_gain/' | null_if_not_file)
	object_compl_gain=$(printf '%s\n' "$depth" | sed 's/depth/object_compl_gain/' | null_if_not_file)
	out=$(printf '%s/%s\n' "$out_dir" "$(basename "$depth")" | sed 's/depth/render/')

	printf 'montage -label %%t -font Liberation-Mono %s %s %s %s %s %s -geometry +2+2 -tile 1x %s\n' \
		"$depth" "$entropy" "$bg_gain" "$object_gain" "$object_dist_gain" "$object_compl_gain" "$out"
done | parallel


