import os
import sys
import pathlib
import enum
import re
import time
import datetime
import json
import collections
import cv2
import numpy as np
import tesserocr
from tesserocr import PSM

from PIL import Image

class PokemonUnite:
    class GameState:
        LOADING = enum.auto()
        BATTLE = enum.auto()
        RESULT = enum.auto()
        OTHER = enum.auto()

    def __init__(self):
        self.VS_IMAGE = cv2.imread("C:\\src\\unite\\matcher\\vs.png", cv2.IMREAD_GRAYSCALE)
        self.CREST_IMAGE = cv2.imread("C:\\src\\unite\\matcher\\crest.png", cv2.IMREAD_GRAYSCALE)
        self.state = self.GameState.OTHER
        self.remaining_time = 600
        self.mypoint = 0
        self.fixed_count = 3
        self.prev_states = collections.deque([], self.fixed_count)
        self.onload = []
        self.onload_fired_at = 0
        tessdata_path = str(pathlib.Path(os.__file__).parents[1] / 'tessdata') # ex.C:\Users\user\AppData\Local\Programs\Python\Python38\tessdata
        self.ocr = tesserocr.PyTessBaseAPI(lang='eng', psm=PSM.SINGLE_BLOCK, path=tessdata_path)

    def __str__(self):
        ret = {}
        if self.state == self.GameState.LOADING:
            ret["state"] = "LOADING"
        elif self.state == self.GameState.BATTLE:
            ret["state"] = "BATTLE"
        elif self.state == self.GameState.RESULT:
            ret["state"] = "RESULT"
        elif self.state == self.GameState.OTHER:
            ret["state"] = "OTHER"

        ret["remaining_time"] = self.remaining_time
        ret["remaining_time_string"] = self.remaining_time_string
        ret["mypoint"] = self.mypoint
        return json.dumps(ret)

    @property
    def remaining_time_string(self):
        return str(datetime.timedelta(seconds=self.remaining_time))[2:7]

    def eval(self, img):
        self.prev_states.append(self.state)
        self.state_scan(img)

        state_counter = collections.Counter(self.prev_states)

        # 状態がしばらく変わらなかったら多分認識に誤りはないだろう
        if self.fixed_count <= state_counter[self.GameState.LOADING]:
            # 前回発火してから5分経ったら多分次の試合だろう
            now = int(time.time())
            if 300 <= (now - self.onload_fired_at):
                for func in self.onload:
                    func()
                self.onload_fired_at = now

        elif self.fixed_count <= state_counter[self.GameState.BATTLE]:
            mypoint_img = img[686:716, 1865:1920]
            mypoint_img = cv2.copyMakeBorder(mypoint_img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            mypoint_img = cv2.cvtColor(mypoint_img, cv2.COLOR_RGB2HSV)
            mypoint_mask = self.process_hsv_extract(mypoint_img, [0, 0, 110], [180, 255, 255], 1, 5, False)
            mypoint_img = cv2.bitwise_and(mypoint_img, mypoint_mask)
            # mypoint_img = cv2.cvtColor(mypoint_img, cv2.COLOR_BGR2GRAY)
            # mypoint_img = cv2.bitwise_not(mypoint_img)
            # mypoint_img = cv2.resize(mypoint_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            self.ocr.SetImage(Image.fromarray(mypoint_img))
            mypoint_string = self.ocr.GetUTF8Text()
            mypoint_string = re.sub(r"\D", "", mypoint_string)
            if len(mypoint_string) != 0:
                self.mypoint = int(mypoint_string)

        # elif self.fixed_count <= state_counter[self.GameState.RESULT]:
        # elif self.fixed_count <= state_counter[self.GameState.OTHER]:

    def state_scan(self, img):
        # 残り時間らしきものが得られるならバトル画面として扱う
        # heuristic process\
        time_image = img[12:43, 1154:1234];  # cv2.imwrite("time_image.png", time_image)
        time_image = cv2.resize(time_image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_AREA)
        time_image = cv2.cvtColor(time_image, cv2.COLOR_RGB2GRAY)
        self.ocr.SetImage(Image.fromarray(time_image))
        time_string = self.ocr.GetUTF8Text()
        time_string = re.sub(r"\D", "", time_string)
        if len(time_string) == 4:
            # minutes = time_string[0:2]
            # seconds = time_string[2:2]
            self.remaining_time = int(time_string[0:2]) * 60 + int(time_string[2:4])
            self.state = self.GameState.BATTLE
            return self.state

        # バーサス表示らしきものが得られるならロード画面として扱う
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        seems_vs = cv2.matchTemplate(gray_img, self.VS_IMAGE, cv2.TM_CCOEFF_NORMED)
        if 0.925 < seems_vs.max():
            self.remaining_time = 600
            self.state = self.GameState.LOADING
            return self.state

        # # ユナイト紋章?らしきものが得られるならリザルト画面として扱う
        # seems_crest = cv2.matchTemplate(img, self.CREST_IMAGE, cv2.TM_CCOEFF_NORMED)
        # if 0.925 < seems_crest.max():
        #     self.remaining_time = 0
        #     self.state = self.GameState.RESULT
        #     return self.state

        # 知らん…
        self.state = self.GameState.OTHER
        return self.state


    # ref. https://github.com/Kazuhito00/hsv-mask-extracter
    def process_hsv_extract(self, hsv_frame, lower_hsv, upper_hsv, closing_kernel_size, top_area_number, is_reverse):
        # HSVマスク画像生成
        mask_hsv = cv2.inRange(hsv_frame, np.array(lower_hsv), np.array(upper_hsv))

        # クロージング処理による粒ノイズ除去
        kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
        mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)

        # 大きい領域の上位のみマスク画像として描画する
        mask = np.zeros(hsv_frame.shape, np.uint8)
        contours = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        for i, controur in enumerate(contours):
            if i < top_area_number:
                mask = cv2.drawContours(mask, [controur],
                                       -1,
                                       color=(255, 255, 255),
                                       thickness=-1)

        # マスク反転
        if is_reverse:
            mask = cv2.bitwise_not(mask)

        return mask
