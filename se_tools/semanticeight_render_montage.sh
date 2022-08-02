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

depth_renders=$(find "$dir" -name 'depth_[[:digit:]]*.png' | sort -n)

for depth in $depth_renders; do
	rgba=$(printf '%s\n' "$depth" | sed 's/depth/rgba/' | null_if_not_file)
	segm=$(printf '%s\n' "$depth" | sed 's/depth/segm/' | null_if_not_file)
	instance=$(printf '%s\n' "$depth" | sed 's/depth/instance/' | null_if_not_file)
	class=$(printf '%s\n' "$depth" | sed 's/depth/class/' | null_if_not_file)
	raycast=$(printf '%s\n' "$depth" | sed 's/depth/raycast/' | null_if_not_file)
	volume=$(printf '%s\n' "$depth" | sed 's/depth/volume/' | null_if_not_file)
	volume_aabb=$(printf '%s\n' "$depth" | sed 's/depth/volume_aabb/' | null_if_not_file)
	volume_color=$(printf '%s\n' "$depth" | sed 's/depth/volume_color/' | null_if_not_file)
	volume_scale=$(printf '%s\n' "$depth" | sed 's/depth/volume_scale/' | null_if_not_file)
	volume_min_scale=$(printf '%s\n' "$depth" | sed 's/depth/volume_min_scale/' | null_if_not_file)

	n=$(frame_number "$depth")
	mkdir -p "$out_dir"
	out="$out_dir/render_$(printf '%05d' "$n").png"

	printf 'montage -label %%t -font Liberation-Mono '
	printf '%s %s %s null: ' "$rgba" "$segm" "$depth"
	printf '%s %s %s %s ' "$instance" "$class" "$raycast" "$volume_color"
	printf '%s %s %s %s ' "$volume" "$volume_aabb" "$volume_scale" "$volume_min_scale"
	printf ' -geometry 320x+2+2 -tile 4x %s\n' "$out"
done | parallel

out_dir_2="$dir"_montage_aabb_mask
volume_aabb_renders=$(find "$dir" -name '*volume_aabb_*.png' | sort -n)

for volume_aabb in $volume_aabb_renders
do
	n=$(frame_number "$volume_aabb")
	name_pattern=$(printf '*aabb_mask_%05d_*.png' "$n")
	aabb_mask_renders=$(find "$dir" -name "$name_pattern" | sort -n)
	if [ -z "$aabb_mask_renders" ]
	then
		continue
	fi
	printf 'montage -label %%t -font Liberation-Mono '
	for aabb_mask in $aabb_mask_renders
	do
		printf '%s ' "$volume_aabb"
	done
	for aabb_mask in $aabb_mask_renders
	do
		printf '%s ' "$aabb_mask"
	done
	mkdir -p "$out_dir_2"
	out="$out_dir_2/render_$(printf '%05d' "$n").png"
	printf -- '-geometry +2+2 -tile x2 %s\n' "$out"
done | parallel

