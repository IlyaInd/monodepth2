import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import io
import telebot

import PIL.Image as pil

import torch
from torchvision import transforms
from networks.resnet_encoder import VAN_encoder
from networks.depth_decoder import HRDepthDecoder

BOT_TOKEN = os.environ.get('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)
parser = argparse.ArgumentParser(description="Bot options")
parser.add_argument("--resolution", type=int, help="resolution of the processed image, choose of (640, 1024)",
                    default=640, choices=[640, 1024])

START_MSG = """This is the bot for monocular depth estimation from RGB images. \
The underlying model utilized by this bot is outlined in the paper "MonoVAN: Visual Attention \
for Self-Supervised Monocular Depth Estimation" authored by I. Indyk and I. Makarov. 

It is important to note that the model has been trained exclusively on the outdoor KITTI dataset. \
However, it also demonstrates commendable performance on indoor images, providing results of decent quality.

To obtain a depth map, simply send a landscape-oriented image. \
The bot will promptly process the image and return a depth map where hotter colors correspond to closer objects.

Please note that the inference process runs on a CPU, which may require a brief waiting period of \
approximately 2-5 seconds before the results are available. We appreciate your patience during this time.
"""

def load_my_model(weights_path):
    encoder_path = f'{weights_path}/encoder.pth'
    decoder_path = f'{weights_path}/depth.pth'

    encoder_dict = torch.load(encoder_path, map_location='cpu')
    encoder = VAN_encoder(zero_layer_mlp_ratio=4, zero_layer_depths=2,  pretrained=False)
    depth_decoder = HRDepthDecoder(num_ch_enc=[64, 64, 128, 320, 512], use_super_res=True, convnext=False)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))

    encoder.eval()
    depth_decoder.eval()
    return encoder, depth_decoder


@bot.message_handler(commands=['info'])
def get_info(message):
    bot.reply_to(message, START_MSG)


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, '- Use command /info to know more about the model \n\n' +
                     '- Send image to the bot to get depth map')


@bot.message_handler(content_types=['text'])
def text_handler(message):
    bot.reply_to(message, 'This is text, not an image')


@bot.message_handler(content_types=['photo'])
def handle_message(message):
    # Get the photo file ID
    file_id = message.photo[-1].file_id
    print(file_id)

    # Get the file object using the file ID
    file_info = bot.get_file(file_id)
    print(file_info)

    # Download the photo
    file = bot.download_file(file_info.file_path)
    curr_image_id = len(list(filter(lambda x: x[-9:-4] == 'image', os.listdir('bot_images'))))  # count only images
    print(f"Current image id = {curr_image_id:0>3}")

    with open(f"bot_images/{curr_image_id:0>3}_image.jpg", "wb") as new_file:
        new_file.write(file)

    img = pil.open(f'bot_images/{curr_image_id:0>3}_image.jpg')
    inference(message, img, curr_image_id)


def inference(message, img, curr_image_id: int):
    """
    :param message: message_id to reply on
    :param curr_image_id: index of current image in database
    :param img: PIL image
    :return: disparity image in PIL format
    """
    source_w, source_h = img.size
    if source_w < source_h:
        bot.reply_to(message, 'Please, provide image in landscape orientation, not in portrait one')
        return None

    aspect_ratio = source_w / source_h
    final_w = min(1279, source_w)  # limit the maximum image size to prevent severe interpolation artifacts
    final_h = int(source_w / aspect_ratio)

    # img_tensor = transforms.ToTensor()(img.resize((640, 256)))  # inherited from KITTI size, adjusted on NYUv2
    height = 416 if options.resolution == 1024 else 256
    img_tensor = transforms.ToTensor()(img.resize((options.resolution, height)))
    with torch.no_grad():
        disp = decoder(encoder(img_tensor))[('disp', 0)]

    disp = transforms.Resize((final_h, final_w))(disp)
    print(f'Final disparity shape = {disp.shape}')
    disp = disp.squeeze()

    fig, ax = plt.subplots(1, 1, figsize=(final_w/100, final_h/100), dpi=130)
    ax.imshow(disp, cmap='magma', vmax=np.percentile(disp, 97))
    ax.axis('off')
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])
    # fig.tight_layout()
    fig.savefig(f'bot_images/{curr_image_id:0>3}_depth.jpg', bbox_inches='tight', pad_inches=0.0, dpi='figure')
    print('-> Saved disparity image')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.0, dpi='figure')
    buf.seek(0)
    im = pil.open(buf)

    bot.send_photo(message.chat.id, im,
                   # f'Depth map resolution = {im.size}'
                   )
    print('-> Sent disparity image')
    buf.close()


if __name__ == '__main__':
    if not os.path.isdir('bot_images'):
        os.mkdir('bot_images')
    options = parser.parse_args()
    encoder, decoder = load_my_model(f'../trained_model/BestConfig_Mish_{options.resolution}')
    print('-> Load model state successfully')
    bot.infinity_polling()
