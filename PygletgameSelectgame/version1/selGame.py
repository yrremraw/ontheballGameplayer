from __future__ import division
from subprocess import call, Popen
import os
import numpy as np
import cv2
import cv2.cv as cv
from pyglet.gl import *
import sys
import pyglet
from pyglet.window import mouse
from math import sqrt
from collections import Counter
import threading
from Queue import Queue
import time

WIDTH = 1024
HEIGHT = 768

stage = 0
select = False

musicT = True
effectT = True

pyglet.clock.set_fps_limit(45)


class Background(object):
    """Background of window"""

    def __init__(self, img):
        super(Background, self).__init__()
        self.img = img

    def draw(self):
        self.img.blit(0, 0, width=WIDTH, height=HEIGHT)
# --------------------------------------------------------------------------
# Create window
# --------------------------------------------------------------------------

platform = pyglet.window.get_platform()
display = platform.get_default_display()
try:
    screens = display.get_screens()[1]
    print "Screen: 1"
except (IndexError, ValueError):
    screens = display.get_screens()[0]
    print "Screen: 0"

game_window = pyglet.window.Window(
    fullscreen=True, screen=screens, width=WIDTH, height=HEIGHT)

# --------------------------------------------------------------------------
# click menu
# --------------------------------------------------------------------------


def right_click_or_laser(x, y):
    global stage
    global select
    global musicT
    global effectT

    if 627 < y < 686 and 829 < x < 1003:
        if effectT:
            button_eff.play()
        stage = 1
    if 566 < y < 625 and 829 < x < 1003:
        if effectT:
            button_eff.play()
        stage = 2
    if 505 < y < 564 and 829 < x < 1003:
        if effectT:
            button_eff.play()
        stage = 3
# uncomment this
    if 122 < y < 184 and 847 < x < 1008:
        if effectT:
            button_eff.play()
        effectT = False
        musicT = False
        if stage == 1:
            call(["../forTest/forTest.exe"], shell=False)
        if stage == 2:
            call(["../dartGame/dartGame.exe"], shell=False)
        if stage == 3:
            call(["../fireThatFlies/fireThatFlies.exe"], shell=False)

    # setting
    if 850 < x < 870 and 228 < y < 248:
        effectT = not effectT
    if 850 < x < 870 and 289 < y < 309:
        musicT = not musicT

    # exit
    if 33 < y < 95 and 847 < x < 1008:
        if effectT:
            button_eff.play()
        pyglet.app.exit()


@game_window.event
def on_mouse_press(x, y, button, modifiers):
    # clear & quit button
    if button == pyglet.window.mouse.RIGHT:
        right_click_or_laser(x, y)


#-----------------------------------------------
# Draw
#-----------------------------------------------

@game_window.event
def on_draw():
    game_window.clear()
    if stage == 0:
        stage0.draw()
    elif stage == 1:
        stage1.draw()
        set_sound.draw()
    elif stage == 2:
        stage2.draw()
        set_sound.draw()
    elif stage == 3:
        stage3.draw()
        set_sound.draw()
    if musicT:
        player.play()
        if stage > 0:
            tick_music.draw()
    elif not musicT:
        player.pause()
    if effectT and stage > 0:
        tick_effect.draw()

# --------------------------------------------------------------------------
# Load resources
# --------------------------------------------------------------------------

pyglet.resource.path = ['../resources']
pyglet.resource.reindex()

setSound = pyglet.resource.image("setsound.png")
set_sound = pyglet.sprite.Sprite(setSound, 851, 224)

stage_0 = pyglet.resource.image("stage0.jpg")
stage0 = pyglet.sprite.Sprite(stage_0, 0, 0)

stage_1 = pyglet.resource.image("stage1.jpg")
stage1 = pyglet.sprite.Sprite(stage_1, 0, 0)

stage_2 = pyglet.resource.image("stage2.jpg")
stage2 = pyglet.sprite.Sprite(stage_2, 0, 0)

stage_3 = pyglet.resource.image("stage3.jpg")
stage3 = pyglet.sprite.Sprite(stage_3, 0, 0)

tick = pyglet.resource.image("tick.jpg")
tick_effect = pyglet.sprite.Sprite(tick, 856, 230)
tick_music = pyglet.sprite.Sprite(tick, 856, 291)

selgame_sound = pyglet.resource.media('sound.mp3', streaming=False)
player = pyglet.media.Player()
for a in range(100):
    player.queue(selgame_sound)

button_eff = pyglet.resource.media('CLICK.mp3', streaming=False)

if __name__ == '__main__':
    pyglet.app.run()
    cv2.destroyAllWindows()
