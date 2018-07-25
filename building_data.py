from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os

np.random.seed(1337)


number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

MAX_CAPTCHA = 3

WIDTH=200
HEIGHT=50


def get_char_set():
	return number+alphabet

def get_char_set_len():
	return len(get_char_set())

def get_captcha_size():
	return MAX_CAPTCHA

def get_y_len():
	return MAX_CAPTCHA*get_char_set_len()

def get_width():
    return WIDTH

def get_height():
    return HEIGHT





height_p = 60
width_p = 70

def random_captcha_text(char_set=number+alphabet, captcha_size=MAX_CAPTCHA):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)

    return captcha_text
 
def gen_captcha_text_and_image(i):
    image = ImageCaptcha(width=70, height=60, font_sizes=[30])
 
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)

    path = './data/'
    if os.path.exists(path) == False: # if the folder is not existed, create it
        os.mkdir(path)
                
    captcha = image.generate(captcha_text)

    image.write(captcha_text, path+str(i)+'_'+captcha_text + '.png') 
 
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image
 
if __name__ == '__main__':

        
        for i in range(20000):  #Number of data   
                text, image = gen_captcha_text_and_image(i)
















