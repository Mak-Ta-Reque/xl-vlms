#!/usr/bin/env python3
"""
Script to process debug_results.csv and create a mapping of concepts to images.
For each concept (string) found in predicted_text, maps it to the images where it appears.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
import re
import argparse
import inflect
import re
p = inflect.engine()


def clean_concept(concept: str) -> str:
    concept = concept.strip().lower().strip('"\'')
    concept = re.sub(r'[^a-z\s]', '', concept)
    concept = re.sub(r'\s+', ' ', concept).strip()
    # Singularize each word safely
    words = [p.singular_noun(w) or w for w in concept.split()]
    return ' '.join(words)


def process_csv_file(csv_file_path, output_file_path=None):
    """
    Process the CSV file and create concept-to-images mapping.
    
    Args:
        csv_file_path: Path to the CSV file
        output_file_path: Optional path to save the mapping as JSON
    
    Returns:
        dict: Mapping of concepts to lists of images
    """
    concept_to_images = defaultdict(list)
    
    print(f"Processing CSV file: {csv_file_path}")
    
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        total_rows = 0
        processed_rows = 0
        
        for row in reader:
            total_rows += 1
            
            # Extract image information
            image_name = row.get('image_name', '')
            root_path = row.get('root_path', '')
            subfolder = row.get('subfolder', '')
            predicted_text = row.get('predicted_text', '')
            
            if not predicted_text or not image_name:
                continue
                
            # Create image identifier
            image_id = f"{Path(root_path).name}/{subfolder}/{image_name}"

            # Split predicted_text by comma and process each concept
            concepts = predicted_text.split(',')
            
            for concept in concepts:
                cleaned_concept = clean_concept(concept)
                
                # Skip empty concepts
                if not cleaned_concept:
                    continue
                    
                # Add image to concept mapping
                if image_id not in concept_to_images[cleaned_concept]:
                    concept_to_images[cleaned_concept].append(image_id)
            
            processed_rows += 1
            
            # Progress indicator
            if processed_rows % 10 == 0:
                print(f"Processed {processed_rows}/{total_rows} rows...")
    
    print(f"Finished processing {processed_rows} rows")
    print(f"Found {len(concept_to_images)} unique concepts")
    
    # Convert defaultdict to regular dict for JSON serialization
    concept_mapping = dict(concept_to_images)
    
    # Save to JSON file if output path is provided
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(concept_mapping, f, indent=2, ensure_ascii=False)
        print(f"Concept mapping saved to: {output_file_path}")
    
    return concept_mapping

def analyze_concept_mapping(concept_mapping):
    """Analyze the concept mapping and provide statistics."""
    print("\n" + "="*50)
    print("CONCEPT MAPPING ANALYSIS")
    print("="*50)
    
    # Sort concepts by number of images (descending)
    sorted_concepts = sorted(concept_mapping.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"Total unique concepts: {len(concept_mapping)}")
    print(f"Total concept-image associations: {sum(len(images) for images in concept_mapping.values())}")
    
    print("\nTop 20 most frequent concepts:")
    print("-" * 40)
    for i, (concept, images) in enumerate(sorted_concepts[:20], 1):
        print(f"{i:2d}. {concept:<25} ({len(images)} images)")
    
    print("\nConcepts appearing in only 1 image:")
    print("-" * 40)
    single_image_concepts = [concept for concept, images in concept_mapping.items() if len(images) == 1]
    print(f"Count: {len(single_image_concepts)}")
    
    if single_image_concepts:
        print("Examples:")
        for concept in single_image_concepts[:10]:
            images = concept_mapping[concept]
            print(f"  - {concept} -> {images[0]}")
    
    return sorted_concepts

def search_concept(concept_mapping, search_term):
    """Search for concepts containing a specific term."""
    search_term = search_term.lower()
    matching_concepts = {}
    
    for concept, images in concept_mapping.items():
        if search_term in concept:
            matching_concepts[concept] = images
    
    return matching_concepts

def main():
    """Main function to process the CSV and create concept mapping."""
    parser = argparse.ArgumentParser(description="Process a CSV file to create a concept-to-images mapping.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save the output JSON mapping.")
    args = parser.parse_args()

    csv_file = args.input
    output_file = args.output

    # Check if CSV file exists
    if not Path(csv_file).exists():
        print(f"Error: CSV file not found at {csv_file}")
        return

    # Process the CSV file
    concept_mapping = process_csv_file(csv_file, output_file)

    # Analyze the mapping
    sorted_concepts = analyze_concept_mapping(concept_mapping)

    # Example searches
    print("\n" + "="*50)
    print("EXAMPLE SEARCHES")
    print("="*50)

    search_terms = ["apple", "red", "kitchen", "wood"]

    for term in search_terms:
        matches = search_concept(concept_mapping, term)
        print(f"\nConcepts containing '{term}': {len(matches)}")
        if matches:
            for concept, images in list(matches.items())[:5]:  # Show first 5 matches
                print(f"  - {concept} ({len(images)} images)")

    print(f"\nComplete mapping saved to: {output_file}")

if __name__ == "__main__":
    main()
