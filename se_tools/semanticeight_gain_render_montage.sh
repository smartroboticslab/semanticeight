#!/bin/sh
set -eu

frame_number() {
	printf '%s\n' "$1" | sed -E 's/^.*([[:digit:]]{5}).png/\1/' | bc
}

null_if_not_file() {
	f=$(cat)
	if [ -f "$f" ]
	then
		printf '%s\n' "$f"
	else
		printf 'null:\n'
	fi
}



if [ "$#" -ne 1 ]
then
	printf 'Usage: %s DIRECTORY\n' "${0##*/}"
	printf '  Combine all images from each frame into one\n'
	printf '  and place it in DIRECTORY_gain_montage\n'
	exit 2
fi

dir="${1%%/}"
out_dir="$dir"_gain_montage

depth_renders=$(find "$dir" -name 'depth_[[:digit:]]*.png' | sort -n)
mkdir -p "$out_dir"

for depth in $depth_renders; do
	rgba=$(printf '%s\n' "$depth" | sed 's/depth/rgba/' | null_if_not_file)
	segm=$(printf '%s\n' "$depth" | sed 's/depth/segm/' | null_if_not_file)
	volume=$(printf '%s\n' "$depth" | sed 's/depth/volume/' | null_if_not_file)
	entropy_pre=$(printf '%s\n' "$depth" | sed 's/depth/entropy_pre/' | null_if_not_file)
	entropy_post=$(printf '%s\n' "$depth" | sed 's/depth/entropy_post/' | null_if_not_file)
	obj_dist_pre=$(printf '%s\n' "$depth" | sed 's/depth/obj_dist_pre/' | null_if_not_file)
	obj_dist_post=$(printf '%s\n' "$depth" | sed 's/depth/obj_dist_post/' | null_if_not_file)
	obj_compl_pre=$(printf '%s\n' "$depth" | sed 's/depth/obj_compl_pre/' | null_if_not_file)
	obj_compl_post=$(printf '%s\n' "$depth" | sed 's/depth/obj_compl_post/' | null_if_not_file)
	gain_pre=$(printf '%s\n' "$depth" | sed 's/depth/gain_pre/' | null_if_not_file)
	gain_post=$(printf '%s\n' "$depth" | sed 's/depth/gain_post/' | null_if_not_file)

	n=$(frame_number "$depth")
	out="$out_dir/render_$(printf '%05d' "$n").png"

	printf 'montage -label %%t -font Liberation-Mono %s %s %s %s %s %s %s %s %s %s %s %s -geometry +2+2 -tile 4x %s\n' \
		"$depth" "$volume" "$rgba" "$segm" \
		"$entropy_pre" "$obj_dist_pre" "$obj_dist_pre" "$gain_pre" \
		"$entropy_post" "$obj_dist_post" "$obj_dist_post" "$gain_post" \
		"$out"
done | parallel
