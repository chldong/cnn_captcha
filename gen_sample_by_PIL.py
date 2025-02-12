# -*- coding: UTF-8 -*-
"""
使用PIL lib生成验证码（前提：pip install PIL）
"""
from PIL import Image,ImageFont,ImageDraw,ImageFilter
import os
import random
import time
import json


def gen_special_img(text, file_path, width, height):
    # 生成img文件
    fontSize = 16
    backGroundColor = (102,142,107)
    fontColor = (112,66,60)
    font = ImageFont.truetype('./simhei.ttf', fontSize)
    img = Image.new('RGBA',(width,height), backGroundColor)
    textWidth, textHeight = font.getsize(text)
    textLeft = (width-textWidth)/2
    textTop = (height-textHeight)/2-2
    draw = ImageDraw.Draw(img)
    draw.text(xy=(textLeft,textTop),text=text,fill=fontColor,font=font)
    rot = img.rotate(0,expand=0)
    img.rotate
    fff = Image.new('RGBA', rot.size,backGroundColor)
    img = Image.composite(rot, fff, rot)
    img.save(file_path)  # 保存图片


def gen_ima_by_batch(root_dir, image_suffix, characters, count, char_count, width, height):
    # 判断文件夹是否存在
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # for index, i in enumerate(range(count)):
    #     text = ""
    #     for j in range(char_count):
    #         text += random.choice(characters)

    for index, i in enumerate(range(count)):
        text = ""
        add_al = chr(random.randrange(65, 91))  # chr转换为A-Z大写。print(chr(90))#65-90任意生成A-Z
        for j in range(char_count):
            text += random.choice(characters)

        text = "".join([str(add_al),text])

        timec = str(time.time()).replace(".", "")
        p = os.path.join(root_dir, "{}_{}.{}".format(text, timec, image_suffix))
        gen_special_img(text, p, width, height)

        print("Generate captcha image => {}".format(index + 1))



def main():
    with open("conf/captcha_config.json", "r") as f:
        config = json.load(f)
    # 配置参数
    root_dir = config["root_dir"]  # 图片储存路径
    image_suffix = config["image_suffix"]  # 图片储存后缀
    characters = config["characters"]  # 图片上显示的字符集 # characters = "0123456789abcdefghijklmnopqrstuvwxyz"
    count = config["count"]  # 生成多少张样本
    char_count = config["char_count"]  # 图片上的字符数量

    # 设置图片高度和宽度
    width = config["width"]
    height = config["height"]

    gen_ima_by_batch(root_dir, image_suffix, characters, count, char_count, width, height)


if __name__ == '__main__':
    main()
