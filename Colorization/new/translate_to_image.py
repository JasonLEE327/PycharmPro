import numpy as np
from PIL import Image

image_size = (image_width, image_height) = (641, 361)

def save_image(filename, image_data):
    image_data = np.mat(image_data, dtype=np.int)
    def color_at_position(row, column):
        index = row * image_width + column
        return image_data[index, 0], image_data[index, 1], image_data[index,2]
    image = Image.new('RGB', image_size)
    for i in range(image_width):
        for j in range(image_height):
            image.putpixel((i, j), color_at_position(j, i))
    image.save(filename)

