#!/bin/bash

input_folder="/home/nf1104/work/trec-dl-2019/dl19-runs"

# Change to the input folder
cd "$input_folder"

# Decompress each .gz file to corresponding .run file
for gz_file in *.gz; do
    # Check if the file exists
    if [ -e "$gz_file" ]; then
        # Extract the base name without extension and remove "input." part
        base_name=$(basename "$gz_file" .gz | sed 's/^input\.//')
        
        # Decompress the file and save as .run in the same directory
        gunzip -c "$gz_file" > "${input_folder}/${base_name}.run"
    fi
done
