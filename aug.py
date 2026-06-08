from PIL import Image, ImageEnhance
import os

def augment_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            with Image.open(img_path) as img:
                # Convert RGBA to RGB if needed
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                # Apply Augmentations
                img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)  # Flip Horizontally
                img_rotated = img.rotate(30)  # Rotate 30 degrees
                enhancer = ImageEnhance.Brightness(img)
                img_bright = enhancer.enhance(1.5)  # Increase Brightness

                # Save augmented images
                img.save(output_path.replace(".jpg", "_original.jpg"))  # Save original
                img_flipped.save(output_path.replace(".jpg", "_flipped.jpg"))
                img_rotated.save(output_path.replace(".jpg", "_rotated.jpg"))
                img_bright.save(output_path.replace(".jpg", "_bright.jpg"))

                print(f"Augmented and saved: {filename}")
        
        except Exception as e:
            print(f"Skipping {filename}: {e}")

# Change folder paths as needed
input_folder = "C:/main pro/noresgold"
output_folder = "C:/main pro/nogold"
augment_images(input_folder, output_folder)
