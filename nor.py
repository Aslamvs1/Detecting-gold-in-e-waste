from PIL import Image
import numpy as np
import os

# Define input and output folders
input_folder = "C:/main pro/nogold"  # Folder containing images
output_folder = "C:/main pro/noresgold"  # Folder to save normalized images

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    
    try:
        # Step 1: Load the image
        with Image.open(img_path) as img:
            # Step 2: Convert to NumPy array
            image_array = np.array(img).astype("float32")
            
            # Step 3: Normalize pixel values
            image_normalized = image_array / 255.0

            # Step 4: Convert back to Image format and save (optional)
            img_normalized = Image.fromarray((image_normalized * 255).astype("uint8"))
            img_normalized.save(os.path.join(output_folder, filename))
            
            print(f"Normalized and saved: {filename}")
    
    except Exception as e:
        print(f"Skipping {filename}: {e}")
