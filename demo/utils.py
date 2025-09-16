import numpy as np
from PIL import Image

def make_sprite(images, image_size=(64,64)):
    n = int(np.ceil(np.sqrt(len(images))))
    sprite = np.ones((image_size[0]*n, image_size[1]*n, 3), dtype=np.uint8)*255
    for idx, img in enumerate(images):
        img = img.resize(image_size)
        x = (idx % n)*image_size[0]
        y = (idx // n)*image_size[1]
        sprite[y:y+image_size[1], x:x+image_size[0]] = np.array(img)
    return Image.fromarray(sprite)
