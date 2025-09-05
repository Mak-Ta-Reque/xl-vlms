#!/usr/bin/env python3
"""
Interactive script to query the concept-to-images mapping.
"""

import json
import sys
from pathlib import Path

def load_concept_mapping(file_path):
    """Load the concept mapping from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Mapping file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        return None

def search_by_concept(concept_mapping, search_term, exact_match=False):
    """Search for concepts by term."""
    search_term = search_term.lower()
    results = {}
    
    for concept, images in concept_mapping.items():
        if exact_match:
            if concept == search_term:
                results[concept] = images
        else:
            if search_term in concept:
                results[concept] = images
    
    return results

def find_images_with_concept(concept_mapping, concept_name):
    """Find all images that contain a specific concept."""
    concept_name = concept_name.lower()
    return concept_mapping.get(concept_name, [])

def find_concepts_in_image(concept_mapping, image_name):
    """Find all concepts that appear in a specific image."""
    image_name = image_name.lower()
    concepts = []
    
    for concept, images in concept_mapping.items():
        if any(image_name in img.lower() for img in images):
            concepts.append(concept)
    
    return concepts

def get_statistics(concept_mapping):
    """Get basic statistics about the mapping."""
    total_concepts = len(concept_mapping)
    total_associations = sum(len(images) for images in concept_mapping.values())
    
    # Get unique images
    all_images = set()
    for images in concept_mapping.values():
        all_images.update(images)
    total_images = len(all_images)
    
    # Concept frequency distribution
    frequencies = [len(images) for images in concept_mapping.values()]
    max_freq = max(frequencies) if frequencies else 0
    min_freq = min(frequencies) if frequencies else 0
    avg_freq = sum(frequencies) / len(frequencies) if frequencies else 0
    
    return {
        'total_concepts': total_concepts,
        'total_associations': total_associations,
        'total_unique_images': total_images,
        'max_frequency': max_freq,
        'min_frequency': min_freq,
        'average_frequency': avg_freq
    }

def display_results(results, title, max_results=20):
    """Display search results in a formatted way."""
    print(f"\n{title}")
    print("=" * len(title))
    
    if not results:
        print("No results found.")
        return
    
    if isinstance(results, dict):
        # Results is a concept->images mapping
        sorted_results = sorted(results.items(), key=lambda x: len(x[1]), reverse=True)
        
        for i, (concept, images) in enumerate(sorted_results[:max_results], 1):
            print(f"{i:2d}. {concept} ({len(images)} images)")
            for img in images[:3]:  # Show first 3 images
                print(f"    - {img}")
            if len(images) > 3:
                print(f"    ... and {len(images) - 3} more")
            print()
        
        if len(sorted_results) > max_results:
            print(f"... and {len(sorted_results) - max_results} more results")
    
    elif isinstance(results, list):
        # Results is a list of items
        for i, item in enumerate(results[:max_results], 1):
            print(f"{i:2d}. {item}")
        
        if len(results) > max_results:
            print(f"... and {len(results) - max_results} more results")

def main():
    """Main interactive function."""
    mapping_file = "/mnt/abka03/Projects/xl-vlms/concept_to_images_mapping.json"
    
    # Load the mapping
    concept_mapping = load_concept_mapping(mapping_file)
    if concept_mapping is None:
        return
    
    # Get statistics
    stats = get_statistics(concept_mapping)
    
    print("CONCEPT-TO-IMAGES MAPPING QUERY TOOL")
    print("=" * 40)
    print(f"Loaded mapping with {stats['total_concepts']} concepts")
    print(f"Total associations: {stats['total_associations']}")
    print(f"Unique images: {stats['total_unique_images']}")
    print(f"Average concepts per image: {stats['total_associations'] / stats['total_unique_images']:.1f}")
    
    if len(sys.argv) > 1:
        # Command line query
        query = " ".join(sys.argv[1:])
        
        if query.startswith("image:"):
            # Query for concepts in a specific image
            image_name = query[6:].strip()
            concepts = find_concepts_in_image(concept_mapping, image_name)
            display_results(concepts, f"Concepts in image '{image_name}'")
        
        elif query.startswith("exact:"):
            # Exact concept match
            concept = query[6:].strip()
            images = find_images_with_concept(concept_mapping, concept)
            if images:
                print(f"\nImages with concept '{concept}':")
                display_results(images, f"Images with concept '{concept}'")
            else:
                print(f"No images found with exact concept '{concept}'")
        
        else:
            # Partial concept search
            results = search_by_concept(concept_mapping, query)
            display_results(results, f"Concepts containing '{query}'")
    
    else:
        # Interactive mode
        print("\nUsage examples:")
        print("  python query_mapping.py apple          # Search for concepts containing 'apple'")
        print("  python query_mapping.py exact:red      # Find images with exactly 'red' concept")
        print("  python query_mapping.py image:403459   # Find concepts in specific image")
        
        # Show some sample queries
        sample_queries = ["apple", "kitchen", "wood", "red"]
        
        for query in sample_queries:
            results = search_by_concept(concept_mapping, query)
            display_results(results, f"Sample: Concepts containing '{query}'", max_results=5)

if __name__ == "__main__":
    main()
