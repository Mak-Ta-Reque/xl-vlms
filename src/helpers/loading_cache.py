import os
import torch
import psutil
import multiprocessing
from typing import Dict, Any
from functools import partial

# Function to update dictionary of lists
def update_dict_of_list(item: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges a dictionary into another, appending values to existing lists or creating new lists for missing keys.

    Args:
        item (Dict[str, Any]): The dictionary whose items should be added to `data`.
        data (Dict[str, Any]): The target dictionary to update.

    Returns:
        Dict[str, Any]: The updated dictionary.
    """
    for k, v in item.items():
        if k in data:
            data[k].append(v)  # Append to existing list
        else:
            data[k] = [v]  # Create a new list if key doesn't exist
    return data

# Check the current RAM usage
def check_ram_usage() -> bool:
    """
    Checks the system's RAM usage. If usage exceeds 90%, return True to indicate that no more files should be loaded.

    Returns:
        bool: True if RAM usage is high, False otherwise.
    """
    memory = psutil.virtual_memory()
    # If memory usage is over 90%, return True (stop loading more files)
    return memory.percent > 10

# Worker function to load pickle file
def load_pickle_file(file_path: str, combined_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loads a pickle file and merges its content into the combined data.

    Args:
        file_path (str): The path of the pickle file to load.
        combined_data (Dict[str, Any]): The dictionary to append data to.

    Returns:
        Dict[str, Any]: The updated combined data.
    """
    if check_ram_usage():
        print(f"RAM usage is high, skipping file: {file_path}")
        return combined_data  # Return the combined_data without changes
    
    # Load the pickle file data
    loaded_data = torch.load(file_path)
    return update_dict_of_list(loaded_data, combined_data)

def load_all_pickles(directory: str) -> Dict[str, Any]:
    """
    Loads all pickle files from a directory and merges them into a single dictionary.
    Utilizes multiprocessing for parallel loading, and stops if RAM usage exceeds 90%.

    Args:
        directory (str): Path to the directory containing .pkl files.

    Returns:
        Dict[str, Any]: Merged dictionary containing all loaded data.
    """
    combined_data = {}  # Initialize empty dictionary

    # List all .pkl files in the directory
    pickle_files = [f for f in os.listdir(directory) if f.endswith(".pkl")]

    # Create a multiprocessing pool
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Use partial to bind the combined_data argument to each worker
        load_func = partial(load_pickle_file, combined_data=combined_data)

        # Map the files to the load function
        results = pool.map(load_func, [os.path.join(directory, file) for file in pickle_files])

    # Merge all results
    for result in results:
        combined_data = result  # Results are already merged inside the worker function

    return combined_data

# Example usage
if __name__ == "__main__":
    directory = "/path/to/pickle/files"  # Replace with your actual directory path
    combined_data = load_all_pickles(directory)
    print(f"Total combined data: {len(combined_data)} keys")
