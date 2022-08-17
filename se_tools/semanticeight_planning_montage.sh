#!/bin/sh
set -eu

sibling_file() {
	f=$(printf '%s\n' "$1" | sed 's/_depth\.png/_'"$2"'.png/')
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
	printf '  and place it in DIRECTORY_montage\n'
	exit 2
fi

output_dir=$(printf '%s\n' "$1" | sed 's|/*$|_montage|')
find "$1" -type f -name '*_depth.png' | while IFS= read -r depth
do
	entropy=$(sibling_file "$depth" 'entropy')
	object_dist_gain=$(sibling_file "$depth" 'object_dist_gain')
	bg_dist_gain=$(sibling_file "$depth" 'bg_dist_gain')

	mkdir -p "$output_dir"
	out=$(printf '%s/%s\n' "$output_dir" "${depth##*/}" |
		sed 's/_depth\.png/_render.png/')

	printf 'montage -label %%t -font Liberation-Mono '
	printf '%s %s %s %s ' "$depth" "$entropy" "$object_dist_gain" "$bg_dist_gain"
	printf ' -geometry 320x+2+2 -tile 1x %s\n' "$out"
done | parallel
