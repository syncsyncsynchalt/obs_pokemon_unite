import enum
import re
import os
import time
import pathlib
import threading
import datetime
import json
import collections
import cv2
import pytesseract
import obspython as S

SCREENSHOTS_PATH = os.getenv("USERPROFILE") + "\\Videos\\"


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
        self.fixed_count = 4
        self.prev_states = collections.deque([], self.fixed_count)
        self.onload = []
        self.onload_fired_at = 0

    def __str__(self):
        ret = {}
        if self.state == self.GameState.LOADING:
            ret["state"] = "LOADING"
        elif self.state == self.GameState.BATTLE:
            ret["state"] = "BATTLE"
        elif self.state == self.GameState.RESULT:
            ret["state"] = "RESULT"
        elif self.state == self.GameState.OTHER:
            ret["state"] = "RESULT"

        ret["remaining_time"] = self.remaining_time
        ret["remaining_time_string"] = self.remaining_time_string
        return json.dumps(ret)

    @property
    def remaining_time_string(self):
        return str(datetime.timedelta(seconds=self.remaining_time))[2:7]

    def eval(self, img):
        self.prev_states.append(self.state)
        self.scan(img)

        state_counter = collections.Counter(self.prev_states)

        # 状態がしばらく変わらなかったら多分認識に誤りはないだろう
        if self.fixed_count <= state_counter[self.GameState.LOADING]:
            # 前回発火してから5分経ったら多分次の試合だろう
            now = int(time.time())
            if 300 <= (now - self.onload_fired_at):
                for func in self.onload:
                    func()
                self.onload_fired_at = now

        # elif self.fixed_count <= state_counter[self.GameState.BATTLE]:
        # elif self.fixed_count <= state_counter[self.GameState.RESULT]:
        # elif self.fixed_count <= state_counter[self.GameState.OTHER]:

    def scan(self, img):
        # 残り時間らしきものが得られるならバトル画面として扱う
        time_image = img[5: 22, 580: 615]
        # cv2.imwrite("time.png", time_image)
        time_string = pytesseract.image_to_string(time_image, lang="eng")
        time_string = re.sub(r"\D", "", time_string)
        if len(time_string) == 4:
            # minutes = time_string[0:2]
            # seconds = time_string[2:2]
            self.remaining_time = int(time_string[0:2]) * 60 + int(time_string[2:4])
            self.state = self.GameState.BATTLE
            return self.state

        # バーサス表示らしきものが得られるならロード画面として扱う
        seems_vs = cv2.matchTemplate(img, self.VS_IMAGE, cv2.TM_CCOEFF_NORMED)
        if 0.925 < seems_vs.max():
            self.remaining_time = 600
            self.state = self.GameState.LOADING
            return self.state

        # ユナイト紋章?らしきものが得られるならリザルト画面として扱う
        seems_crest = cv2.matchTemplate(img, self.CREST_IMAGE, cv2.TM_CCOEFF_NORMED)
        if 0.925 < seems_crest.max():
            self.remaining_time = 0
            self.state = self.GameState.RESULT
            return self.state

        # 知らん…
        self.state = self.GameState.OTHER
        return self.state


unite = PokemonUnite()
unite.onload.append(lambda: S.obs_frontend_replay_buffer_save())
source = S.obs_get_source_by_name("unite")


def _tick():
    # XXX: Pythonの世界からオンメモでレンダリング結果を取得する術が分からない
    # obs-studio/UI/window-basic-main-screenshot.cppを移植しようとしたが
    # Pythonからポインタを渡せない|辿れないのと、任意サイズのメモリを確保する術が分からず頓挫
    S.obs_frontend_take_source_screenshot(source)
    try:
        last_file = max(pathlib.Path(SCREENSHOTS_PATH).glob("Screenshot_*.png"), key=os.path.getctime)
        screenshot = cv2.imread(str(last_file), cv2.IMREAD_GRAYSCALE)
        last_file.unlink()
    except:
        print("file delete failure")

    unite.eval(screenshot)


def tick():
    th = threading.Thread(target=_tick)
    th.start()


def script_update(settings):
    # print("script_update")
    S.timer_remove(tick)
    S.timer_add(tick, 1 * 1000)


# def test(self, event):
#     print("oops")
#
# def script_properties():
#     # print("script_properties")
#     props = S.obs_properties_create()
#     S.obs_properties_add_button(props, "test", "TEST:", test)
#     return props
