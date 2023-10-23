import io
from time import time

import numpy as np
import requests
import telebot
import webuiapi
from loguru import logger
from PIL import Image, ImageOps
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x-seg.pt')#.to("mps")  # load an official model
api = webuiapi.WebUIApi()


def crop(img):
    new_width, new_height = [min(*img.size)]*2
    width, height = img.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    return img

def inpaint(img, fnum):
    # img = crop(img)
    img = img.resize((960, 640)) if img.width > img.height else img.resize((640, 960))
    # bot.send_message(message.chat.id, "Run model...")
    results = model(img)
    if results[0].masks is None:
        # bot.send_message(message.chat.id, "Никого не нашли на фото, попробуй сфоткаться по-другому :)")
        return
    # img.save(f"output/{fnum}-inp.png")
    mask = results[0].masks.data.cpu().numpy()
    
    cls = results[0].boxes.cls
    mask = mask[cls == 0] if len(cls) > 1 else mask
    if len(mask.shape) < 3:
        # bot.send_message(message.chat.id, "Не нашёл людей :( попробуй сфоткаться по-другому!")
        return
    # bot.send_message(message.chat.id, f"Найдено человеков: {mask.shape[0]}...")
    mask = mask.sum(axis=0).clip(0, 1)*255
    mask = mask.astype(np.uint8)
    
    k = (mask == 0).sum()/img.size[0]/img.size[1]
    print(k)
    if k < 0.3:
        return
    
    # kernel = np.ones((5,5),np.uint8)
    # mask = cv2.dilate(np.array(mask),kernel,iterations = 10)
    mask = Image.fromarray(mask).convert("RGB")#.resize((512, 512))
    mask = ImageOps.invert(mask).resize(results[0].orig_shape[::-1])
    # bot.send_message(message.chat.id, "Вжух...вжух...")
    inpainting_result = api.img2img(images=[img],
                                    mask_image=mask,
                                    inpainting_fill=1,
                                    prompt="ancient dark castle with walls made of black stone and brics",
                                    negative_prompt="person face captions",
                                    # seed=0,
                                    steps=100,
                                    mask_blur=4,
                                    cfg_scale=7,
                                    resize_mode=3,
                                    denoising_strength=0.4)
    img_res = inpainting_result.image
    return img_res








if __name__ == "__main__":
    token = "6545009319:AAG1YRp7Cr_FOg2jwFBSXYrDR39dgpHoazM"
    bot = telebot.TeleBot(token)
    @bot.message_handler(content_types=["photo"])
    def get_photo(message):
        try:
            document_id = message.photo[2].file_id
            file_info = bot.get_file(document_id)
            url = f"http://api.telegram.org/file/bot{token}/{file_info.file_path}"
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                fnum = int(time())
                bot.send_message(message.chat.id, "Open photo...")
                img = Image.open(io.BytesIO(r.content))
                img_res = inpaint(img, fnum)
                bot.send_photo(message.chat.id, img_res)
                img_res.save(f"output/{fnum}-out.png")
        except Exception as ex:
            bot.send_message(message.chat.id, ex)
    bot.polling(none_stop=True, interval=0, timeout=180)