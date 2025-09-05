#!/usr/bin/env python3
"""
Generate a comprehensive report of the concept-to-images mapping.
"""

import json
from collections import Counter, defaultdict
import csv

def load_concept_mapping(file_path):
    """Load the concept mapping from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_comprehensive_report(mapping_file, csv_file, output_file):
    """Generate a comprehensive analysis report."""
    
    # Load data
    concept_mapping = load_concept_mapping(mapping_file)
    
    # Get all unique images
    all_images = set()
    for images in concept_mapping.values():
        all_images.update(images)
    
    # Statistics
    total_concepts = len(concept_mapping)
    total_associations = sum(len(images) for images in concept_mapping.values())
    total_images = len(all_images)
    avg_concepts_per_image = total_associations / total_images if total_images > 0 else 0
    
    # Frequency analysis
    concept_frequencies = [(concept, len(images)) for concept, images in concept_mapping.items()]
    concept_frequencies.sort(key=lambda x: x[1], reverse=True)
    
    # Image analysis - concepts per image
    image_concept_count = defaultdict(int)
    for concept, images in concept_mapping.items():
        for image in images:
            image_concept_count[image] += 1
    
    # Category analysis
    color_concepts = []
    object_concepts = []
    material_concepts = []
    texture_concepts = []
    spatial_concepts = []
    
    color_words = ['red', 'blue', 'green', 'yellow', 'brown', 'white', 'black', 'gray', 'orange', 'purple', 'pink', 'silver', 'gold']
    material_words = ['wood', 'metal', 'plastic', 'glass', 'ceramic', 'fabric', 'leather', 'stone']
    texture_words = ['smooth', 'rough', 'textured', 'glossy', 'matte', 'shiny', 'patterned']
    spatial_words = ['horizontal', 'vertical', 'round', 'oval', 'rectangular', 'square', 'curved']
    
    for concept, count in concept_frequencies:
        if any(color in concept for color in color_words):
            color_concepts.append((concept, count))
        elif any(material in concept for material in material_words):
            material_concepts.append((concept, count))
        elif any(texture in concept for texture in texture_words):
            texture_concepts.append((concept, count))
        elif any(spatial in concept for spatial in spatial_words):
            spatial_concepts.append((concept, count))
        else:
            object_concepts.append((concept, count))
    
    # Generate report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CONCEPT-TO-IMAGES MAPPING ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall Statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total unique concepts: {total_concepts}\n")
        f.write(f"Total unique images: {total_images}\n")
        f.write(f"Total concept-image associations: {total_associations}\n")
        f.write(f"Average concepts per image: {avg_concepts_per_image:.1f}\n\n")
        
        # Top concepts by frequency
        f.write("TOP 30 MOST FREQUENT CONCEPTS\n")
        f.write("-" * 35 + "\n")
        for i, (concept, count) in enumerate(concept_frequencies[:30], 1):
            f.write(f"{i:2d}. {concept:<30} ({count:2d} images)\n")
        f.write("\n")
        
        # Images with most concepts
        sorted_images = sorted(image_concept_count.items(), key=lambda x: x[1], reverse=True)
        f.write("IMAGES WITH MOST CONCEPTS\n")
        f.write("-" * 25 + "\n")
        for i, (image, count) in enumerate(sorted_images[:15], 1):
            f.write(f"{i:2d}. {image:<25} ({count:2d} concepts)\n")
        f.write("\n")
        
        # Category analysis
        f.write("CONCEPT CATEGORIES ANALYSIS\n")
        f.write("-" * 27 + "\n")
        
        f.write(f"Color-related concepts: {len(color_concepts)}\n")
        f.write("Top color concepts:\n")
        for concept, count in color_concepts[:10]:
            f.write(f"  - {concept} ({count} images)\n")
        f.write("\n")
        
        f.write(f"Material-related concepts: {len(material_concepts)}\n")
        f.write("Top material concepts:\n")
        for concept, count in material_concepts[:10]:
            f.write(f"  - {concept} ({count} images)\n")
        f.write("\n")
        
        f.write(f"Texture-related concepts: {len(texture_concepts)}\n")
        f.write("Top texture concepts:\n")
        for concept, count in texture_concepts[:10]:
            f.write(f"  - {concept} ({count} images)\n")
        f.write("\n")
        
        f.write(f"Object-related concepts: {len(object_concepts)}\n")
        f.write("Top object concepts:\n")
        for concept, count in object_concepts[:15]:
            f.write(f"  - {concept} ({count} images)\n")
        f.write("\n")
        
        # Rare concepts (appearing in only 1 image)
        rare_concepts = [(concept, images) for concept, images in concept_mapping.items() if len(images) == 1]
        f.write(f"RARE CONCEPTS (appearing in only 1 image): {len(rare_concepts)}\n")
        f.write("-" * 45 + "\n")
        
        # Group rare concepts by image
        rare_by_image = defaultdict(list)
        for concept, images in rare_concepts:
            rare_by_image[images[0]].append(concept)
        
        for image, concepts in sorted(rare_by_image.items()):
            f.write(f"\n{image}:\n")
            for concept in sorted(concepts):
                f.write(f"  - {concept}\n")
        
        # Detailed mapping for specific concepts of interest
        interesting_concepts = ['apple', 'apples', 'kitchen', 'fruit', 'food']
        f.write(f"\nDETAILED MAPPING FOR KEY CONCEPTS\n")
        f.write("-" * 35 + "\n")
        
        for concept in interesting_concepts:
            if concept in concept_mapping:
                images = concept_mapping[concept]
                f.write(f"\n'{concept}' appears in {len(images)} images:\n")
                for image in images:
                    f.write(f"  - {image}\n")

def main():
    """Main function."""
    mapping_file = "/mnt/abka03/Projects/xl-vlms/concept_to_images_mapping.json"
    csv_file = "/mnt/abka03/Projects/xl-vlms/debug_results.csv"
    output_file = "/mnt/abka03/Projects/xl-vlms/concept_analysis_report.txt"
    
    print("Generating comprehensive analysis report...")
    generate_comprehensive_report(mapping_file, csv_file, output_file)
    print(f"Report saved to: {output_file}")

if __name__ == "__main__":
    main()
