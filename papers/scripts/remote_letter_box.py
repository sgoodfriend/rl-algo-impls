from PIL import Image


def remove_letter_box(image_path, upper_letter_box_height, bottom_letter_box_height):
    img = Image.open(image_path)
    width, height = img.size

    # Assuming the black bar is about 5% of the height
    crop_area = (
        0,
        height * upper_letter_box_height,
        width,
        height * (1 - bottom_letter_box_height),
    )
    cropped_img = img.crop(crop_area)
    cropped_img.save(image_path)


# List of image file paths to process
image_paths = [
    "papers/cog2024/figures/basesWorkers8x8A.png",
    "papers/cog2024/figures/FourBasesWorkers8x8.png",
    "papers/cog2024/figures/NoWhereToRun9x8.png",
    "papers/cog2024/figures/basesWorkers16x16A.png",
    "papers/cog2024/figures/TwoBasesBarracks16x16.png",
    "papers/cog2024/figures/DoubleGame24x24.png",
    "papers/cog2024/figures/BWDistantResources32x32.png",
    "papers/cog2024/figures/(4)BloodBath.png",
]

for path in image_paths:
    remove_letter_box(path, upper_letter_box_height=0.07, bottom_letter_box_height=0.04)
