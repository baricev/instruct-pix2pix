#!/bin/bash

# Define the API endpoint URL
url="https://huggingface.co/api/datasets/timbrooks/instructpix2pix-clip-filtered/parquet/default/train"

# Function to fetch URLs and get their sizes
get_parquet_files_list() {

    # produces a 'newline' separated text file of individual file urls:
    
    # https://huggingface.co/api/datasets/timbrooks/instructpix2pix-clip-filtered/parquet/default/train/0.parquet
    # 499831414
    # "\n"
    # https://huggingface.co/api/datasets/timbrooks/instructpix2pix-clip-filtered/parquet/default/train/1.parquet
    # 489231476
    # "\n"


    # Fetch the list of file URLs using wget and parse it with jq
    local urls=$(wget -qO- "$url" | jq -r '.[]')  

    # Output file 
    local output_file="file_sizes.txt"

    # Prepare the output file
    echo "" > "$output_file"  # Clear the existing output file or create a new one

    echo "Fetching file sizes..."
    # Loop through each URL to get the file size
    for file_url in $urls; do
        # Use wget to perform a spider operation to fetch the header information
        local file_info=$(wget --spider --server-response "$file_url" 2>&1)
        
        # Extract the file size from the response
        local file_size=$(echo "$file_info" | grep "Content-Length:" | tail -1 | awk '{print $2}' | tr -d '[[:space:]]')

        # Output the URL and file size to the file
        echo "$file_url" >> "$output_file"
        echo "$file_size" >> "$output_file"
        echo "" >> "$output_file"  # Add a newline for separation
    done

    echo "File sizes have been written to $output_file"
}

# Check for dependency
if ! command -v wget &> /dev/null || ! command -v jq &> /dev/null; then
    echo "This script requires 'wget' and 'jq'. Please install them first."
    exit 1
fi

# Execute the function
get_parquet_files_list
