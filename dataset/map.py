import jax
import os
import requests

def parse_file_data(filepath):
    """ Parses the file data from the given file path. """
    file_info = []
    with open(filepath, 'r') as file:
        content = file.read().strip().split('\n\n')
        for entry in content:
            if entry:
                lines = entry.split('\n')
                url = lines[0].strip()
                size = int(lines[1].strip())
                file_info.append((url, size))
    return file_info


def partition_files(file_info, n_bins):
    """ Greedy algorithm to partition files into N bins with roughly equal total sizes. """

    # NOTE: Karmarkar--Karp seems overkill at this point, but if necessary should probably incorporate an existing solution instead of hand-rolling it: 
    # https://github.com/fuglede/numberpartitioning/tree/master
    # https://github.com/google/or-tools

    # Sort files by size in descending order
    file_info_sorted = sorted(file_info, key=lambda x: x[1], reverse=True)
    # Initialize bins
    bins = {i: {'urls': [], 'total_size': 0} for i in range(n_bins)}

    for url, size in file_info_sorted:
        # Find the bin with the minimum total size
        min_bin = min(bins, key=lambda x: bins[x]['total_size'])
        bins[min_bin]['urls'].append(url)
        bins[min_bin]['total_size'] += size
    return bins

# Divides up a list of filenames into N evenly sized chunks
def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def get_parquet_files_list(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Assumes this returns a list of URLs directly
    else:
        raise Exception(f"Failed to retrieve Parquet files. Status code: {response.status_code}")

# Optionally download a subset of the full dataset by specifying number of files ( -1 downloads entire dataset )
SUBSET_SIZE = -1 

def main():
    """
    This script partitions a list of file downloads between N jax processes. 

    To create the file list run:
    'bash datasets/get_file_sizes.sh'

    This produces a 'newline' separated text file:
    
    ```
    https://huggingface.co/api/datasets/timbrooks/instructpix2pix-clip-filtered/parquet/default/train/0.parquet
499831414

https://huggingface.co/api/datasets/timbrooks/instructpix2pix-clip-filtered/parquet/default/train/1.parquet
489231476

    ```
    
    """

    # File path of text file containing urls and file sizes 
    filepath = 'dataset/file_sizes.txt' 

    home_dir = os.getenv('HOME')
    data_dir = os.path.join(home_dir, filepath)
    
    n_bins = jax.process_count()  # Number of bins (processes) 

    # Parse the file data
    file_info = parse_file_data(filepath)
    
    # Partition the files into bins
    bins = partition_files(file_info, n_bins)

    # Get the current JAX process device index
    worker_id = jax.process_index()

    # Find the url files list this worker should download
    worker_files = bins[worker_id]['urls']

    # Write the list of files this worker should download to a file
    with open(f"worker_{worker_id}_files.txt", "w") as file:
        for url in worker_files:
            file.write(f"{url}\n")

    print(f"MAP.PY: {len(worker_files)} URLs assigned for download by worker {worker_id}.")


if __name__ == "__main__":

    SUBSET_SIZE = -1 
    main()

