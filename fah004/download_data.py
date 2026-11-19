import os
import shutil
from datasets import load_dataset

def download_high_quality_fashion():
    print("📥 Downloading high-quality color fashion images from Hugging Face...")
    
    # Load the dataset (This is the standard Fashion Product Images dataset)
    # We will grab 500 images for a fast, high-quality demo
    dataset = load_dataset("ashraq/fashion-product-images-small", split="train[:10000]")
    
    target_dir = "dataset/images"
    
    # 1. Delete the old blurry images to avoid confusion
    if os.path.exists(target_dir):
        print("🧹 Removing old blurry images...")
        shutil.rmtree(target_dir)
        
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"💾 Saving 10000 high-resolution images to {target_dir}...")
    
    for i, item in enumerate(dataset):
        img = item['image'] # These are already full-color PIL images
        # Get the product name to use as a filename (e.g., "Red_Dress_1.jpg")
        product_name = item['productDisplayName'].replace(" ", "_").replace("/", "")
        filename = f"{product_name}_{i}.jpg"
        
        filepath = os.path.join(target_dir, filename)
        
        # Save the image
        img.convert('RGB').save(filepath, "JPEG")
        
    print(f"✅ Success! Extracted 500 high-resolution images to {target_dir}")

if __name__ == "__main__":
    download_high_quality_fashion()