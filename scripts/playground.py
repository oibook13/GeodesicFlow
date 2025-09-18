import os
from PIL import Image

def get_png_file_sizes(folder_path):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print("The specified folder does not exist.")
        return

    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        # Construct the complete file path
        file_path = os.path.join(folder_path, file_name)

        # Check if the file is a PNG
        if file_name.lower().endswith('.png') and os.path.isfile(file_path):
            # Open the PNG file using PIL (Pillow)
            with Image.open(file_path) as img:
                width, height = img.size
                # if width == 512:
                print(f"{file_name}: {width}x{height} pixels")
                assert width == 768
            # try:
            #     # Open the PNG file using PIL (Pillow)
            #     with Image.open(file_path) as img:
            #         width, height = img.size
            #         print(f"{file_name}: {width}x{height} pixels")
            # except Exception as e:
            #     print(f"Could not open {file_name}: {e}")

folder_path = "/PHShome/yl535/project/python/datasets/richhf_18k_dataset/richhf_18k/test_for_simpletuner"
get_png_file_sizes(folder_path)