import obspython as S
from multiprocessing import shared_memory
from PIL import Image
import numpy as np
import cv2
import threading
from pokemon_unite import PokemonUnite

unite = PokemonUnite()
unite.onload.append(lambda: S.obs_frontend_replay_buffer_save())


def take_screenshot():
    try:
        shm = shared_memory.SharedMemory("POKEMON_UNITE")
        # skip header
        # ref. https://github.com/synap5e/obs-screenshot-plugin
        shm.buf.obj.seek(4 * 4)
        img = Image.frombytes("RGBA", (2400, 800), shm.buf.obj.read())
    except:
        print("WTF")
    finally:
        shm.close()

    ret = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    return ret


def _tick():
    screenshot = take_screenshot()
    unite.eval(screenshot)
    print(unite)


def tick():
    print("tick")
    th = threading.Thread(target=_tick)
    th.start()


def script_update(settings):
    print("script_update")
    S.timer_remove(tick)
    S.timer_add(tick, 1 * 1000)


def show_ss(self, evnt):
    ss = take_screenshot()
    unite.eval(ss)
    print(unite)
    cv2.imshow("Push any key", ss)
    cv2.waitKey(0)


def script_properties():
    print("script_properties")
    props = S.obs_properties_create()
    S.obs_properties_add_button(props, "show_ss", "Show Screenshot:", show_ss)
    return props
