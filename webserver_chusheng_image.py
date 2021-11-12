# -*- coding: UTF-8 -*-
"""
    出生证明图片接口，访问`/chusheng/1`获得图片
"""
from captcha.image import ImageCaptcha
from PIL import Image,ImageFont,ImageDraw,ImageFilter
import os
import random
from flask import Flask, request, jsonify, Response, make_response
import json
import io


# Flask对象
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))


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


def response_headers(content):
    resp = Response(content)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


def gen_special_img():
    # 随机文字
    text = ""
    add_al = chr(random.randrange(65, 91))  # chr转换为A-Z大写。print(chr(90))#65-90任意生成A-Z
    for j in range(char_count):
        text += random.choice(characters)
    text = "".join([str(add_al),text])
    print(text)
    
    # 生成验证码文件
    # generator = ImageCaptcha(width=width, height=height)  # 指定大小
    # img = generator.generate_image(text)  # 生成图片
    # imgByteArr = io.BytesIO()
    # img.save(imgByteArr, format='PNG')
    # imgByteArr = imgByteArr.getvalue()
    # return imgByteArr

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
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


@app.route('/chusheng/', methods=['GET'])
def show_photo():
    if request.method == 'GET':
        image_data = gen_special_img()
        response = make_response(image_data)
        response.headers['Content-Type'] = 'image/png'
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    else:
        pass


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=6100,
        debug=True
    )
