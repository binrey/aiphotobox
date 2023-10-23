from time import time
from pathlib import Path
import numpy as np
from loguru import logger
from PIL import Image, ImageOps
from bot import inpaint


dir = list(Path("/Users/andrybin/Pictures/129_FUJI/фотозона/out").glob("*.jpg"))
np.random.shuffle(dir)
for f in dir:
    fnum = int(time())
    img = Image.open(f)
    img = ImageOps.exif_transpose(img)
    img_res = inpaint(img, fnum)
    if img_res:
        img_res.save(f"output/{fnum}.png")