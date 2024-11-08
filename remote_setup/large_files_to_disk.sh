#!/bin/bash

# Directory to move large files to
DISK_DIR=$1

# Ensure the target directory exists
if [ ! -d "$DISK_DIR" ]; then
    echo "Target directory $DISK_DIR does not exist. Creating it now."
    mkdir -p "$DISK_DIR"
fi

# Find the 20 largest files in the root directory (excluding /mnt/disk1)
find / -path "$DISK_DIR" -prune -o -type f -exec du -h {} + 2>/dev/null | sort -hr | head -n 20 | while read -r size filepath; do
    # Extract the directory and filename
    dir=$(dirname "$filepath")
    filename=$(basename "$filepath")

    # Create the same directory structure in the target directory
    new_dir="$DISK_DIR$dir"
    mkdir -p "$new_dir"

    # Move the file to the new directory
    mv "$filepath" "$new_dir/"

    # Create a symbolic link from the old location to the new location
    ln -s "$new_dir/$filename" "$filepath"

    echo "Moved $filepath to $new_dir/ and created a symbolic link."
done
