import os
import random
from PIL import Image, ImageChops, ImageEnhance
from pathlib import Path
import concurrent.futures

def convert_to_ela(image_path, save_path, quality=90):
    """
    Applies Error Level Analysis to an image and saves the heatmap.
    """
    try:
        # Open original image
        original = Image.open(image_path).convert('RGB')
        
        # Re-save at a known JPEG quality
        temp_filename = f"temp_ela_{os.getpid()}_{os.path.basename(image_path)}.jpg"
        original.save(temp_filename, 'JPEG', quality=quality)
        compressed = Image.open(temp_filename)
        
        # Calculate absolute difference
        ela_image = ImageChops.difference(original, compressed)
        
        # Enhance brightness
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        # Save output
        ela_image.save(save_path, 'JPEG')
        
        # Cleanup
        compressed.close()
        original.close()
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_file(args):
    src_path, dest_path = args
    convert_to_ela(src_path, dest_path)

def setup_and_process():
    base_dir = Path("1GB_dataset")
    raw_dir = base_dir / "raw"
    ela_dir = base_dir / "ela"
    
    # Create splits
    train_ratio = 0.8
    tasks = []
    
    for category in ["real", "fake"]:
        category_dir = raw_dir / category
        if not category_dir.exists():
            print(f"Warning: {category_dir} does not exist.")
            continue
            
        images = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
        random.shuffle(images)
        
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        print(f"Category {category}: {len(train_images)} train, {len(val_images)} val")
        
        for split_name, img_list in [("train", train_images), ("val", val_images)]:
            out_dir = ela_dir / split_name / category
            out_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in img_list:
                dest_path = out_dir / f"{img_path.stem}.jpg"
                tasks.append((str(img_path), str(dest_path)))
    
    print(f"Processing {len(tasks)} images with Error Level Analysis...")
    
    # Process in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(process_file, tasks))
        
    print("Done generating ELA dataset.")

if __name__ == "__main__":
    setup_and_process()
