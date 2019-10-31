import os
import sys
import lcd
import math
import time
import image
import audio
import sensor
import KPU as kpu


from Maix import I2S
from Maix import GPIO
from machine import I2C
from fpioa_manager import *
from board import board_info
from fpioa_manager import fm
from machine import Timer,PWM

lcd.init()
lcd.rotation(2)

img_w = 224
img_h = 224

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((img_w, img_h))
sensor.set_vflip(2)
sensor.run(1)

lcd.clear()
lcd.draw_string(100,96,"MobileNet Demo")
lcd.draw_string(100,112,"Loading labels...")


#os.remove("/flash/boot.py")

print(os.getcwd())

print(os.listdir("/"))
print(os.listdir("/flash"))
#print(os.listdir("/sd"))

print(board_info.LED_R)
print(board_info.LED_G)
print(board_info.LED_B)

task = kpu.load(0x300000)
anchor = (1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437, 6.92275, 6.718375, 9.01025)
sensor.set_vflip(0)
clock = time.clock()

# AF = afraid
# AN = angry
# DI = disgusted
# HA = happy
# NE = neutral
# SA = sad
# SU = surprised

'''fm.register(board_info.LED_R, fm.fpioa.GPIO4)
fm.register(board_info.LED_G, fm.fpioa.GPIO5)
fm.register(board_info.LED_B, fm.fpioa.GPIO6)'''

'''led_r = GPIO(GPIO.GPIO4, GPIO.OUT)
led_g = GPIO(GPIO.GPIO5, GPIO.OUT)
led_b = GPIO(GPIO.GPIO6, GPIO.OUT)'''

duty=0
dir = True

#tim = Timer(Timer.TIMER0, Timer.CHANNEL0, mode=Timer.MODE_PWM)
#ch = PWM(tim, freq=500000, duty=2.5, pin=board_info.LED_G)
#ch_g = PWM(tim, freq=500000, duty=2.5, pin=board_info.LED_G)
#ch_b = PWM(tim, freq=500000, duty=2.5, pin=board_info.LED_B)

labels = ['afraid', 'angry', 'disgusted', 'happy', 'neutral', 'sad', 'surprised']
emotion_colors = []

def emotion_color():
    global dir, duty

    if dir:
        duty += 5
    else:
        duty -= 5

    if duty>100:
        duty = 100
        dir = False
    elif duty<0:
        duty = 0
        dir = True

    time.sleep(0.05)
    ch.duty(duty)

while True:
    img = sensor.snapshot()
    clock.tick()
    fmap = kpu.forward(task, img)
    fps=clock.fps()
    plist=fmap[:]
    pmax=max(plist)
    max_index=plist.index(pmax)
    a = lcd.display(img) ##, oft=(0,0))

    #print(str(pmax) +  " " + str(labels[max_index]))
    lcd.draw_string(10, 10, str(labels[max_index]))

    #emotion_color()

kpu.fmap_free(fmap)
a = kpu.deinit(task)



