from __future__ import division
import numpy as np
import cv2
import cv2.cv as cv
import pyglet
from pyglet.gl import *
from pyglet.window import mouse
from common import draw_str
from math import sqrt
from collections import Counter
import threading
from Queue import Queue
from collections import deque
import time

# Game variables
WIDTH = 1024
cWIDTH = WIDTH / 2
HEIGHT = 768
cHEIGHT = HEIGHT / 2
BUTTONw = 150
BUTTONh = 70
click_time = 0
sum_score = 0
hitPscore = 0
size = 1  # 10 Level
score = 1  # 12 Level
sizeStar = 95  # min 95    max 580
scoreStar = 313  # min 313   max 907
levelSize = 54
temp = 0
pause = False

ballT = True
laserT = True
musicT = True
effectT = True
setting_stage = False

# Screen detection variables
screen_check = False

SCREEN_X_TOP = 0
SCREEN_Y_TOP = 0
SCREEN_X_BOT = 720
SCREEN_Y_BOT = 720

screen_x_long = 0
screen_y_long = 0

# Ball & Laser detector variables
ball_before = 0
ball_count = 0
new_ball = False
bounce = 0
channels = {
    'hue': None,
    'saturation': None,
    'value': None,
    'laser': None,
}

isCamera = True
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        isCamera = False
    print "camera: 0"
else:
    print "camera: 1"
cap.set(3, 720)
cap.set(4, 720)
frame = None
crop_img = None

q_ball_x = Queue()
q_ball_y = Queue()
q_laser_x = Queue()
q_laser_y = Queue()

balls = []


def is_ball_increase(before, after):
    if after > before:
        print("increase")
        print("ball before = %d" % before)
        print("ball count = %d" % after)
        return True
    else:
        return False


def is_ball_decrease(before, after):
    if before > after:
        print("ball decrease")
        return True
    else:
        return False


def distance(p0, p1):
    return sqrt((int(p0[0]) - int(p1[0])) ** 2 + (int(p0[1]) - int(p1[1])) ** 2)


def closest(pt, others):
    return min(others, key=lambda p: distance(pt, p[0]))


class Ball(object):
    """docstring for Ball"""

    def __init__(self, x, y, ball_size, number=0):
        super(Ball, self).__init__()
        self.ball_size = ball_size
        self.number = number
        self.center = (int(x), int(y))
        self.tracker = deque(maxlen=32)
        self.tracker.appendleft(self.center)
        self.bounced = False
        self.direction = ""

    # Update ball position from nearest ball position
    def update(self, x, y, ball_size):
        self.center = (x, y)
        self.ball_size = ball_size
        self.tracker.appendleft(self.center)

    # show position, size, number of ball
    def show(self):
        # draw the outer circle
        cv2.circle(crop_img, (self.center[0], self.center[1]), self.ball_size, (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(crop_img, (self.center[0], self.center[1]), 2, (0, 0, 255), 3)
        # type ball status
        draw_str(crop_img, (self.center[0], self.center[1]), 'Pos {},{}'.format(self.center[0], self.center[1]))
        draw_str(crop_img, (self.center[0], self.center[1] + 15), 'R {}'.format(self.ball_size))
        draw_str(crop_img, (self.center[0], self.center[1] + 30), 'Number {}'.format(self.number))
        draw_str(crop_img, (self.center[0], self.center[1] + 45), 'Bounced? {}'.format(self.bounced))

    def check_size(self, out_q_x, out_q_y):
        global screen_x_long
        global screen_y_long
        # print self.ball_size
        if self.ball_size <= 15 and self.bounced is False:
            self.bounced = True
            print "center = ", self.center
            x_o = ((self.center[0]) / screen_x_long) * WIDTH
            y_o = HEIGHT - (((self.center[1]) / screen_y_long) * HEIGHT)
            print "height = ", HEIGHT, " width = ", WIDTH
            if ballT:
                out_q_x.put(x_o)
                out_q_y.put(y_o)
                print "send:" + str(x_o) + "," + str(y_o)

    # remove ball object when bounced on the wall
    def delete(self):
        self.tracker = None

    def destroy(self):
        global balls

        self.delete()
        balls.remove(self)


def threshold_image(channel):
    global channels
    minimum = 0
    maximum = 0
    if channel == "hue":
        minimum = 46
        maximum = 69
    elif channel == "saturation":
        minimum = 0
        maximum = 15
    elif channel == "value":
        minimum = 208
        maximum = 255

    (t, tmp) = cv2.threshold(
        channels[channel],  # src
        maximum,  # threshold value
        0,  # we dont care because of the selected type
        cv2.THRESH_TOZERO_INV  # t type
    )

    (t, channels[channel]) = cv2.threshold(
        tmp,  # src
        minimum,  # threshold value
        255,  # maxvalue
        cv2.THRESH_BINARY  # type
    )

    if channel == 'saturation':
        # only works for filtering red color because the range for the hue
        # is split
        channels['saturation'] = cv2.bitwise_not(
            channels['saturation'])


def laser(img):
    global channels
    # split the video frame into color channels
    h, s, v = cv2.split(img)
    channels['hue'] = h
    channels['saturation'] = s
    channels['value'] = v

    threshold_image("hue")
    threshold_image("saturation")
    threshold_image("value")

    # Perform an AND on HSV components to identify the laser!
    channels['laser'] = cv2.bitwise_and(
        channels['hue'],
        channels['value']
    )
    channels['laser'] = cv2.bitwise_and(
        channels['saturation'],
        channels['laser']
    )

    return channels['laser']


def ball_detector(out_q_x, out_q_y):
    """Thread of ball detector"""
    global ball_before
    global ball_count
    global new_ball
    global bounce
    global frame
    global crop_img

    green_color_hsv_min = cv.Scalar(43, 54, 84)
    green_color_hsv_max = cv.Scalar(78, 179, 256)
    orange_color_hsv_min = cv.Scalar(0, 105, 200)
    orange_color_hsv_max = cv.Scalar(11, 256, 256)
    blue_color_hsv_min = cv.Scalar(103, 90, 87)
    blue_color_hsv_max = cv.Scalar(112, 256, 181)
    yellow_color_hsv_min = cv.Scalar(17, 70, 163)
    yellow_color_hsv_max = cv.Scalar(27, 256, 256)
    ball_number = 0

    while True:
        ret, frame = cap.read()
        if ret:
            crop_img = frame[SCREEN_Y_TOP:SCREEN_Y_BOT, SCREEN_X_TOP:SCREEN_X_BOT]
            hsv_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            hsv_blue = cv2.inRange(hsv_image,
                                   blue_color_hsv_min,
                                   blue_color_hsv_max
                                   )
            hsv_orange = cv2.inRange(hsv_image,
                                     orange_color_hsv_min,
                                     orange_color_hsv_max
                                     )
            hsv_green = cv2.inRange(hsv_image,
                                    green_color_hsv_min,
                                    green_color_hsv_max
                                    )
            hsv_yellow = cv2.inRange(hsv_image,
                                     yellow_color_hsv_min,
                                     yellow_color_hsv_max
                                     )
            green_yellow_hsv = cv2.bitwise_or(hsv_green, hsv_yellow)
            green_yellow_blue_hsv = cv2.bitwise_or(green_yellow_hsv, hsv_blue)
            green_yellow_blue_orange_hsv = cv2.bitwise_or(green_yellow_blue_hsv,
                                                          hsv_orange)

            str_el = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            open_morphed = cv2.morphologyEx(green_yellow_blue_orange_hsv,
                                            cv2.MORPH_OPEN,
                                            str_el
                                            )
            morphed = cv2.morphologyEx(open_morphed, cv2.MORPH_CLOSE, str_el)

            hsv_blur = cv2.GaussianBlur(morphed, (11, 11), 4, 4)
            circles = cv2.HoughCircles(hsv_blur, cv.CV_HOUGH_GRADIENT, 2, 720 / 9,
                                       param1=75, param2=45, minRadius=5,
                                       maxRadius=70)
            ball_count = 0

            if circles is not None:
                circles = np.uint16(np.around(circles))
                ball_count = len(circles)
                for i in circles[0, :]:

                    # create ball object if ball's number is increase
                    if is_ball_increase(ball_before, ball_count) and i[2] > 16:
                        ball_number += 1
                        print "new ball"
                        balls.append(Ball(i[0], i[1], i[2], ball_number))

                    # update ball position in ball object
                    pos_list = []
                    for b in balls:
                        pos_list.append([(b.center[0], b.center[1]), b])

                    if len(pos_list) == 0:
                        pass
                    elif len(pos_list) == 1:
                        pos_list[0][1].update(i[0], i[1], i[2])
                        if (0 <= i[0] <= SCREEN_X_BOT) and (0 <= i[1] <= SCREEN_Y_BOT):
                            pos_list[0][1].check_size(out_q_x, out_q_y)

                        pos_list[0][1].show()

                    else:
                        close = closest((i[0], i[1]), pos_list)
                        close[1].update(i[0], i[1], i[2])
                        if (0 <= i[0] <= SCREEN_X_BOT) and (0 <= i[1] <= SCREEN_Y_BOT):
                            close[1].check_size(out_q_x, out_q_y)
                        print pos_list
                        close[1].show()

            if is_ball_decrease(ball_before, ball_count):
                for b in balls:
                    if b.bounced is True:
                        b.destroy()
            cv2.rectangle(frame, (SCREEN_X_TOP, SCREEN_Y_TOP), (SCREEN_X_BOT, SCREEN_Y_BOT), (0, 255, 0), 3)
            cv2.imshow('crop', crop_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ball_before = ball_count
    print "Exiting Background Thread: Ball detector"


def laser_detector(out_l_x, out_l_y):
    """Thread of laser detector"""
    global screen_x_long
    global screen_y_long
    while True:
        ret, frame2 = cap.read()
        time.sleep(0.5)
        crop_img2 = frame2[SCREEN_Y_TOP:SCREEN_Y_BOT, SCREEN_X_TOP:SCREEN_X_BOT]
        hsv_image2 = cv2.cvtColor(crop_img2, cv2.COLOR_BGR2HSV)
        laser(hsv_image2)
        laser_str_el = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        laser_str_el_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        laser_close_morphed = cv2.morphologyEx(channels['laser'],
                                               cv2.MORPH_CLOSE,
                                               laser_str_el
                                               )
        laser_morphed = cv2.morphologyEx(laser_close_morphed,
                                         cv2.MORPH_OPEN,
                                         laser_str_el_2
                                         )

        blur = cv2.GaussianBlur(laser_morphed, (7, 7), 4, 4)

        lasers = cv2.HoughCircles(blur, cv.CV_HOUGH_GRADIENT, 2.5, 720 / 2,
                                  param1=10, param2=4, minRadius=4,
                                  maxRadius=10
                                  )
        if lasers is not None:
            lasers = np.uint16(np.around(lasers))
            for i in lasers[0, :]:
                print "lasers!"
                # draw the outer circle
                cv2.circle(crop_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(crop_img, (i[0], i[1]), 2, (0, 0, 255), 3)
                x_l = ((i[0]) / screen_x_long) * WIDTH
                y_l = HEIGHT - (((i[1]) / screen_y_long) * HEIGHT)
                if laserT:
                    out_l_x.put(x_l)
                    out_l_y.put(y_l)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print "Exiting Background Thread: Laser detector"


# --------------------------------------------------------------------------
# Game objects
# --------------------------------------------------------------------------

score_label = pyglet.text.Label('Score :',
                                font_name='Times New Roman',
                                font_size=15,
                                x=20, y=HEIGHT - 15,
                                anchor_x='left', anchor_y='top')

sumScore_label = pyglet.text.Label('',
                                   font_name='Times New Roman',
                                   font_size=40,
                                   x=195, y=HEIGHT - 75,
                                   anchor_x='right', anchor_y='bottom')

countscore_text = pyglet.text.Label('',
                                    font_name='Times New Roman',
                                    font_size=50, bold=True,
                                    x=WIDTH / 2, y=HEIGHT / 2,
                                    anchor_x='center', anchor_y='center')

pauseButton_label = pyglet.text.Label('Pause',
                                      font_name='Times New Roman',
                                      font_size=25,
                                      x=WIDTH - BUTTONw + 45, y=95 + 2 * BUTTONh,
                                      anchor_x='center', anchor_y='center')

resumeButton_label = pyglet.text.Label('Resume',
                                       font_name='Times New Roman',
                                       font_size=25,
                                       x=WIDTH - BUTTONw + 45, y=95 + 2 * BUTTONh,
                                       anchor_x='center', anchor_y='center')

resetButton_label = pyglet.text.Label('Reset',
                                      font_name='Times New Roman',
                                      font_size=25,
                                      x=WIDTH - BUTTONw + 45, y=80 + BUTTONh,
                                      anchor_x='center', anchor_y='center')

quitButton_label = pyglet.text.Label('Quit',
                                     font_name='Times New Roman',
                                     font_size=25,
                                     x=WIDTH - BUTTONw + 45, y=65,
                                     anchor_x='center', anchor_y='center')

sizeAdj_text = pyglet.text.Label('',
                                 font_name='Times New Roman', color=(255, 230, 0, 255),
                                 font_size=30, bold=True,
                                 x=375, y=45,
                                 anchor_x='left', anchor_y='bottom')

scoreAdj_text = pyglet.text.Label('',
                                  font_name='Times New Roman', color=(255, 230, 0, 255),
                                  font_size=30, bold=True,
                                  x=880, y=705,
                                  anchor_x='left', anchor_y='top')


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
    fullscreen=True, screen=screens, width=WIDTH, height=HEIGHT, caption="On the ball")


# --------------------------------------------------------------------------
# Check if click in circle
# --------------------------------------------------------------------------

def is_within(x, y):
    global click_time
    global sum_score
    global score
    global hitPscore
    global temp
    radius = 0
    distance_circle = sqrt(((x - WIDTH / 2) ** 2) +
                    ((y - HEIGHT / 2) ** 2))
    if size == 1:
        radius = dartBoard1.width / 2
    elif size == 2:
        radius = dartBoard2.width / 2
    elif size == 3:
        radius = dartBoard3.width / 2
    elif size == 4:
        radius = dartBoard4.width / 2
    elif size == 5:
        radius = dartBoard5.width / 2
    elif size == 6:
        radius = dartBoard6.width / 2
    elif size == 7:
        radius = dartBoard7.width / 2
    elif size == 8:
        radius = dartBoard8.width / 2
    elif size == 9:
        radius = dartBoard9.width / 2
    elif size == 10:
        radius = dartBoard10.width / 2

    if (distance_circle < radius) and not pause:
        click_time += 1
        hitPscore += 1

        if effectT:
            dart_eff.play()

        if hitPscore > 1:
            sum_score += (hitPscore // score) - sum_score + temp
        else:
            temp = sum_score
            sum_score += hitPscore // score
    else:
        if effectT:
            miss_eff.play()


def is_right_click(x, y):
    global size
    global sizeStar
    global score
    global scoreStar

    global hitPscore
    global click_time
    global sum_score
    global temp
    global pause

    global ballT
    global laserT
    global musicT
    global effectT
    global setting_stage

    if not setting_stage:
        # pause
        if 200 < y < 270 and 844 < x < 994:
            if effectT:
                button_eff.play()
            if pause:
                pause = False
            elif not pause:
                pause = True
        # reset screen
        if 115 < y < 185 and 844 < x < 994:
            if effectT:
                button_eff.play()
            click_time = 0
            sum_score = 0
            hitPscore = 0
            temp = 0
        # quit
        elif 30 < y < 100 and 844 < x < 994:
            if effectT:
                button_eff.play()
            pyglet.app.exit()
        # score -
        elif 275 < x < 302 and 712 < y < 739:
            if effectT:
                button_eff.play()
            if score > 1:
                score -= 1
                scoreStar -= levelSize
                hitPscore = 0
        # score +
        elif 938 < x < 965 and 712 < y < 739:
            if effectT:
                button_eff.play()
            if score < 12:
                score += 1
                scoreStar += levelSize
                hitPscore = 0
        # size -
        elif 54 < x < 81 and 55 < y < 82:
            if effectT:
                button_eff.play()
            if size > 1:
                size -= 1
                sizeStar -= levelSize
        # size +
        elif 54 < x < 81 and 618 < y < 645:
            if effectT:
                button_eff.play()
            if size < 10:
                size += 1
                sizeStar += levelSize

        if 924 < x < 994 and 619 < y < 689:
            if effectT:
                button_eff.play()
            setting_stage = True

    # setting
    if setting_stage:
        if 464 < x < 566 and 453 < y < 479:
            if effectT:
                button_eff.play()
            ballT = not ballT
        if 376 < x < 655 and 391 < y < 417:
            if effectT:
                button_eff.play()
            laserT = not laserT
        if 443 < x < 585 and 236 < y < 262:
            if effectT:
                button_eff.play()
            musicT = not musicT
        if 436 < x < 594 and 175 < y < 201:
            if effectT:
                button_eff.play()
            effectT = not effectT
        if 407 < x < 619 and 49 < y < 131:
            if effectT:
                button_eff.play()
            setting_stage = False


# Press spacebar twice to calibrate projected screen
@game_window.event
def on_key_press(symbol, modifiers):
    global screen_check
    global SCREEN_X_TOP
    global SCREEN_Y_TOP
    global SCREEN_X_BOT
    global SCREEN_Y_BOT
    global screen_x_long
    global screen_y_long
    global frame

    screen_color_hsv_min = cv.Scalar(103, 90, 202)
    screen_color_hsv_max = cv.Scalar(112, 230, 256)

    if symbol == pyglet.window.key.SPACE and not screen_check:
        screen_check = True

    elif symbol == pyglet.window.key.SPACE and screen_check:
        # list of top left and bottom right corner position of screen
        x_top = []
        y_top = []
        x_bot = []
        y_bot = []

        for x in range(50):
            time.sleep(0.02)
            print "round {}".format(x)
            ret, framex = cap.read()
            if x == 49:
                screen_check = False
                print "------------------------ End --------------------------"
                print x_top
                print y_top
                print x_bot
                print y_bot
                data_x_top = Counter(x_top)
                # Returns the highest occurring item from each list
                SCREEN_X_TOP = data_x_top.most_common(1)[0][0]
                data_y_top = Counter(y_top)
                SCREEN_Y_TOP = data_y_top.most_common(1)[0][0]
                data_x_bot = Counter(x_bot)
                SCREEN_X_BOT = data_x_bot.most_common(1)[0][0]
                data_y_bot = Counter(y_bot)
                SCREEN_Y_BOT = data_y_bot.most_common(1)[0][0]

                screen_x_long = SCREEN_X_BOT - SCREEN_X_TOP
                screen_y_long = SCREEN_Y_BOT - SCREEN_Y_TOP

                print SCREEN_X_TOP, SCREEN_X_BOT, SCREEN_Y_TOP, SCREEN_Y_BOT
                del x_top, y_top, x_bot, y_bot

            elif ret:
                hsv_image = cv2.cvtColor(framex, cv2.COLOR_BGR2HSV)
                hsv_blue = cv2.inRange(hsv_image,
                                       screen_color_hsv_min,
                                       screen_color_hsv_max
                                       )
                gray = cv2.GaussianBlur(hsv_blue, (3, 3), 0)
                edged = cv2.Canny(gray, 10, 30)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
                (cnts, _) = cv2.findContours(
                    closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                total = 0

                for c in cnts:
                    # approximate the contour
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                    if len(approx) == 4:
                        cv2.drawContours(framex, [approx], -1, (0, 255, 0), 4)
                        total += 1

                        # Get Screen corner position from each frame
                        x_min = min(approx[0][0][0],
                                    approx[1][0][0],
                                    approx[2][0][0],
                                    approx[3][0][0])

                        y_min = min(approx[0][0][1],
                                    approx[1][0][1],
                                    approx[2][0][1],
                                    approx[3][0][1])
                        print x_min, y_min
                        x_max = max(approx[0][0][0],
                                    approx[1][0][0],
                                    approx[2][0][0],
                                    approx[3][0][0])

                        y_max = max(approx[0][0][1],
                                    approx[1][0][1],
                                    approx[2][0][1],
                                    approx[3][0][1])
                        # print x_max, y_max
                        # print ("Width: " + str(x_max - x_min) +
                        # " Height: " + str(y_max - y_min))

                        if x_max - x_min > 30 and y_max - y_min > 30:
                            x_top.append(x_min)
                            y_top.append(y_min)
                            x_bot.append(x_max)
                            y_bot.append(y_max)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass
            else:
                pass


# --------------------------------------------------------------------------
# Count score when click
# --------------------------------------------------------------------------

@game_window.event
def on_mouse_press(x, y, button, modifiers):
    if button == pyglet.window.mouse.LEFT:
        if not setting_stage:
            is_within(x, y)
            # pause & quit button
    if button == pyglet.window.mouse.RIGHT:
        is_right_click(x, y)


# --------------------------------------------------------------------------
# Draw
# --------------------------------------------------------------------------

@game_window.event
def on_draw():
    game_window.clear()
    if screen_check is True:
        background_blue.draw()
    else:
        # board
        if size == 1:
            dart_Board = pyglet.sprite.Sprite(dartBoard1, (WIDTH / 2) - (dartBoard1.width / 2),
                                              (HEIGHT / 2) - (dartBoard1.height / 2))
        elif size == 2:
            dart_Board = pyglet.sprite.Sprite(dartBoard2, (WIDTH / 2) - (dartBoard2.width / 2),
                                              (HEIGHT / 2) - (dartBoard2.height / 2))
        elif size == 3:
            dart_Board = pyglet.sprite.Sprite(dartBoard3, (WIDTH / 2) - (dartBoard3.width / 2),
                                              (HEIGHT / 2) - (dartBoard3.height / 2))
        elif size == 4:
            dart_Board = pyglet.sprite.Sprite(dartBoard4, (WIDTH / 2) - (dartBoard4.width / 2),
                                              (HEIGHT / 2) - (dartBoard4.height / 2))
        elif size == 5:
            dart_Board = pyglet.sprite.Sprite(dartBoard5, (WIDTH / 2) - (dartBoard5.width / 2),
                                              (HEIGHT / 2) - (dartBoard5.height / 2))
        elif size == 6:
            dart_Board = pyglet.sprite.Sprite(dartBoard6, (WIDTH / 2) - (dartBoard6.width / 2),
                                              (HEIGHT / 2) - (dartBoard6.height / 2))
        elif size == 7:
            dart_Board = pyglet.sprite.Sprite(dartBoard7, (WIDTH / 2) - (dartBoard7.width / 2),
                                              (HEIGHT / 2) - (dartBoard7.height / 2))
        elif size == 8:
            dart_Board = pyglet.sprite.Sprite(dartBoard8, (WIDTH / 2) - (dartBoard8.width / 2),
                                              (HEIGHT / 2) - (dartBoard8.height / 2))
        elif size == 9:
            dart_Board = pyglet.sprite.Sprite(dartBoard9, (WIDTH / 2) - (dartBoard9.width / 2),
                                              (HEIGHT / 2) - (dartBoard9.height / 2))
        elif size == 10:
            dart_Board = pyglet.sprite.Sprite(dartBoard10, (WIDTH / 2) - (dartBoard10.width / 2),
                                              (HEIGHT / 2) - (dartBoard10.height / 2))
        dart_Board.draw()
        # adjust bar
        score_bar.draw()
        size_bar.draw()
        star_size = pyglet.sprite.Sprite(starSize, 55, sizeStar)
        star_size.draw()
        star_score = pyglet.sprite.Sprite(starSize, scoreStar, HEIGHT - 55)
        star_score.draw()
        # BG
        score_BG.draw()
        quit_BG.draw()
        pause_BG.draw()
        reset_BG.draw()
        # text
        countscore_text.text = str(click_time)
        countscore_text.draw()
        score_label.draw()
        resetButton_label.draw()

        if pause is True:
            pause_screen.draw()
            resumeButton_label.draw()
        else:
            pauseButton_label.draw()

        quitButton_label.draw()
        # print size & score
        sizeAdj_text.text = str(size)
        sizeAdj_text.draw()
        scoreAdj_text.text = str(score)
        scoreAdj_text.draw()
        # Calculate score
        sumScore_label.text = str(sum_score)
        sumScore_label.draw()
        set_but.draw()
        # setting
        if setting_stage:
            setting_bg.draw()
            if ballT:
                tick_ball.draw()
            if laserT:
                tick_lp.draw()
            if musicT:
                tick_music.draw()
            if effectT:
                tick_eff.draw()
        if musicT:
            player.play()
        if not musicT:
            player.pause()

        border_bg.draw()


# --------------------------------------------------------------------------
# Detect
# --------------------------------------------------------------------------
def ball_check(_, q_x, q_y):
    if not q_x.empty() and not q_y.empty():
        is_within(q_x.get(), q_y.get())
        print "Ball Hit!!"


def laser_check(_, q_x, q_y):
    if not q_x.empty() and not q_y.empty():
        print "Laser Get!!"
        is_right_click(q_x.get(), q_y.get())


pyglet.clock.schedule_interval(ball_check, 1 / 20., q_ball_x, q_ball_y)
pyglet.clock.schedule_interval(laser_check, 1 / 10., q_laser_x, q_laser_y)

# --------------------------------------------------------------------------
# Load resources
# --------------------------------------------------------------------------

pyglet.resource.path = ['../resources']
pyglet.resource.reindex()

background_white = Background(pyglet.resource.image("background_white.jpg"))
background_blue = Background(pyglet.resource.image("background_blue.jpg"))

dartBoard10 = pyglet.resource.image("DB574.png")
dartBoard9 = pyglet.resource.image("DB517.png")
dartBoard8 = pyglet.resource.image("DB465.png")
dartBoard7 = pyglet.resource.image("DB419.png")
dartBoard6 = pyglet.resource.image("DB377.png")
dartBoard5 = pyglet.resource.image("DB339.png")
dartBoard4 = pyglet.resource.image("DB305.png")
dartBoard3 = pyglet.resource.image("DB275.png")
dartBoard2 = pyglet.resource.image("DB248.png")
dartBoard1 = pyglet.resource.image("DB223.png")

pauseScreen = pyglet.resource.image("pauseScreen.png")
pause_screen = pyglet.sprite.Sprite(pauseScreen, WIDTH / 2 - pauseScreen.width / 2, HEIGHT / 2 - pauseScreen.height / 2)

buttonBG = pyglet.resource.image("buttonBG.png")
pause_BG = pyglet.sprite.Sprite(buttonBG, WIDTH - BUTTONw - 30, 60 + 2 * BUTTONh)
reset_BG = pyglet.sprite.Sprite(buttonBG, WIDTH - BUTTONw - 30, 45 + BUTTONh)
quit_BG = pyglet.sprite.Sprite(buttonBG, WIDTH - BUTTONw - 30, 30)
BG = pyglet.resource.image("scoreBG.png")
score_BG = pyglet.sprite.Sprite(BG, 10, HEIGHT - BUTTONh - 10)

scoreBar = pyglet.resource.image("scoreBar.png")
score_bar = pyglet.sprite.Sprite(scoreBar, BUTTONw + 120, HEIGHT - 115)
sizeBar = pyglet.resource.image("sizeBar.png")
size_bar = pyglet.sprite.Sprite(sizeBar, 50, 50)
starSize = pyglet.resource.image("star.png")

setBut = pyglet.resource.image("setBut01.jpg")
set_but = pyglet.sprite.Sprite(setBut, WIDTH - setBut.width - 30, HEIGHT - 80 - setBut.height)

borderBg = pyglet.resource.image("border.png")
border_bg = pyglet.sprite.Sprite(borderBg, 0, 0)

# settings
settingBG = pyglet.resource.image("setting.jpg")
setting_bg = pyglet.sprite.Sprite(settingBG, 0, 0)

tick = pyglet.resource.image("tick.jpg")
tick_ball = pyglet.sprite.Sprite(tick, 467, 473 - tick.height)
tick_lp = pyglet.sprite.Sprite(tick, 379, 411 - tick.height)
tick_music = pyglet.sprite.Sprite(tick, 446, 256 - tick.height)
tick_eff = pyglet.sprite.Sprite(tick, 439, 195 - tick.height)

# sound&music
dart_sound = pyglet.resource.media('Ponies_and_Balloons.mp3', streaming=False)
player = pyglet.media.Player()
for a in range(100):
    player.queue(dart_sound)

dart_eff = pyglet.resource.media('dart.mp3', streaming=False)
miss_eff = pyglet.resource.media('miss.mp3', streaming=False)
button_eff = pyglet.resource.media('CLICK.mp3', streaming=False)

if __name__ == '__main__':
    # create queues for keep data from detector

    # create daemon thread of ball and laser detector run in background
    ball_thread = threading.Thread(target=ball_detector, args=(q_ball_x, q_ball_y))
    laser_thread = threading.Thread(target=laser_detector, args=(q_laser_x, q_laser_y))
    ball_thread.daemon = True
    laser_thread.daemon = True

    if isCamera is True:
        print "Start Background (Daemon) Thread"
        # start ball and laser thread
        ball_thread.start()
        laser_thread.start()

    print "Start Main Thread"
    # get start time to check runtime
    start = time.time()

    # run game
    pyglet.app.run()

    cv2.destroyAllWindows()

    print "Exiting Main Thread (Game)"
    print(time.time() - start)
