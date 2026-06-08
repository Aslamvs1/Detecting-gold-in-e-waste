from PIL import Image
import os

def resize_images(input_folder, output_folder, size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            with Image.open(img_path) as img:
                # Convert RGBA images to RGB to remove transparency
                if img.mode in ("RGBA", "P"):  
                    img = img.convert("RGB")
                
                img_resized = img.resize(size)
                img_resized.save(output_path)
                
                print(f"Resized and saved: {filename}")
        except Exception as e:
            print(f"Skipping {filename}: {e}")

input_folder = "C:/main pro/nogold"  # Change this to your folder
output_folder = "C:/main pro/noresgold"
resize_images(input_folder, output_folder)
