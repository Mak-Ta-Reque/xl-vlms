import os
import torch
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from torchvision import transforms

def download_and_transform_cats(num_images=300, image_size=224, output_dir='cat_images'):
    """
    Download cat images from ImageNet and transform them to a specific size.
    
    Args:
        num_images (int): Number of images to download
        image_size (int): Size to resize images to (will be square)
        output_dir (str): Directory to save the images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    # ImageNet cat class ID
    cat_class_id = 'n02123045'
    
    # URL for ImageNet class info
    class_info_url = f'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={cat_class_id}'
    
    try:
        # Get list of image URLs
        response = requests.get(class_info_url)
        image_urls = response.text.split('\n')
        
        # Filter out empty URLs and limit to requested number
        image_urls = [url.strip() for url in image_urls if url.strip()][:num_images]
        
        print(f"Downloading and transforming {len(image_urls)} cat images...")
        
        transformed_images = []
        saved_paths = []
        
        # Download and transform images with progress bar
        for i, url in enumerate(tqdm(image_urls)):
            try:
                # Download image
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Open image and transform
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    transformed_img = transform(img)
                    
                    # Save transformed image
                    img_path = os.path.join(output_dir, f'cat_{i:04d}.jpg')
                    transformed_images.append(transformed_img)
                    saved_paths.append(img_path)
                    
                    # Save original image
                    img.save(img_path, 'JPEG')
                    
            except Exception as e:
                print(f"Error processing image {url}: {str(e)}")
                continue
        
        # Stack all transformed images into a single tensor
        if transformed_images:
            all_images = torch.stack(transformed_images)
            print(f"Successfully created tensor of shape: {all_images.shape}")
            return all_images, saved_paths
        else:
            print("No images were successfully processed")
            return None, []
                
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, []

if __name__ == "__main__":
    # Download and transform images to 224x224 (standard ImageNet size)
    images_tensor, image_paths = download_and_transform_cats(num_images=300, image_size=224)
    
    if images_tensor is not None:
        print(f"Final tensor shape: {images_tensor.shape}")
        print(f"Images saved in: {os.path.abspath('cat_images')}") 