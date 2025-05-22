from pathlib import Path
import scipy.io

def create_image_label_file(dir_images, output_path, path_labels, path_synset_words, path_meta):
    """Create a text file with image names and their class names"""
    
    # Load meta.mat file
    meta = scipy.io.loadmat(str(path_meta))
    
    # Create mappings from meta.mat
    original_idx_to_synset = {}
    synset_to_name = {}
    for i in range(1000):
        ilsvrc2012_id = int(meta["synsets"][i,0][0][0][0])
        synset = meta["synsets"][i,0][1][0]
        name = meta["synsets"][i,0][2][0]
        original_idx_to_synset[ilsvrc2012_id] = synset
        synset_to_name[synset] = name
    
    # Create mappings from synset_words.txt
    synset_to_keras_idx = {}
    keras_idx_to_name = {}
    with open(str(path_synset_words), "r") as f:
        for idx, line in enumerate(f):
            parts = line.split(" ", 1)
            synset = parts[0]
            synset_to_keras_idx[synset] = idx
            keras_idx_to_name[idx] = parts[1].strip()
    
    # Define conversion function
    def convert_original_idx_to_keras_idx(idx):
        return synset_to_keras_idx[original_idx_to_synset[idx]]
    
    # Read validation ground truth file
    with open(path_labels, 'r') as f:
        val_labels = [int(line.strip()) for line in f.readlines()]
    
    # Create the output file
    with open(output_path, 'w') as f:
        # For each validation image (assuming they're numbered from 1 to 50000)
        for i in range(1, 50001):
            # Format image filename with leading zeros
            image_file = f"ILSVRC2012_val_{i:08d}.JPEG"
            
            # Get the original class ID for this image (i-1 because val_labels is 0-indexed)
            original_class_id = val_labels[i-1]
            
            # Convert to keras index
            keras_idx = convert_original_idx_to_keras_idx(original_class_id)
            
            # Get keras name
            keras_name = keras_idx_to_name[keras_idx]
            
            # Write image name and keras name to file
            f.write(f"{image_file}\t{keras_name}\n")
    
    print(f"Created image-label file at {output_path}")

# Your file paths
dir_images = Path("/mnt/abka03/raw_data_download/imagent/val")
path_labels = Path("/mnt/abka03/Projects/xl-vlms/preprocessing/data/ILSVRC2012_validation_ground_truth.txt")
path_synset_words = Path("/mnt/abka03/Projects/xl-vlms/preprocessing/data/synset_words.txt")
path_meta = Path("/mnt/abka03/Projects/xl-vlms/preprocessing/data/meta.mat")

# Create the output file
output_file = Path("/mnt/abka03/Projects/xl-vlms/preprocessing/data/image_labels.txt")
create_image_label_file(dir_images, output_file, path_labels, path_synset_words, path_meta)