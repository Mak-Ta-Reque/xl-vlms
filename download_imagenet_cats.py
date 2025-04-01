import os
import requests
from PIL import Image
from io import BytesIO
import json
from tqdm import tqdm

def download_imagenet_cats(num_images=300, output_dir='cat_images'):
    """
    Download cat images from ImageNet.
    
    Args:
        num_images (int): Number of images to download
        output_dir (str): Directory to save the images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # ImageNet cat class ID (n02123045)
    cat_class_id = 'n02123045'
    
    # URL for ImageNet class info
    class_info_url = f'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={cat_class_id}'
    
    try:
        # Get list of image URLs
        response = requests.get(class_info_url)
        image_urls = response.text.split('\n')
        
        # Filter out empty URLs and limit to requested number
        image_urls = [url.strip() for url in image_urls if url.strip()][:num_images]
        
        print(f"Downloading {len(image_urls)} cat images...")
        
        # Download images with progress bar
        for i, url in enumerate(tqdm(image_urls)):
            try:
                # Download image
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Try to open as image to verify it's valid
                    img = Image.open(BytesIO(response.content))
                    
                    # Save image
                    img_path = os.path.join(output_dir, f'cat_{i:04d}.jpg')
                    img.save(img_path, 'JPEG')
                    
            except Exception as e:
                print(f"Error downloading image {url}: {str(e)}")
                continue
                
        print(f"Download complete! Images saved in {output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    download_imagenet_cats(num_images=300) 