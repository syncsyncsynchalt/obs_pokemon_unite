import unittest
import random
import shutil

from datetime import time

from pokemon_unite import *
import cv2
import math
from xml.dom import minidom
from xml.dom.minidom import getDOMImplementation
from xml.etree.ElementTree import Element, SubElement, Comment, tostring


class TestPokemonUnite(unittest.TestCase):

    # @classmethod
    # def setUpClass(cls):
    #     cls.runner = PokemonUnite()
    #     cls.battle_png = cv2.imread("samples\\battle.png")
    #     cls.battle2_png = cv2.imread("samples\\battle2.png")
    #     cls.battle_death_png = cv2.imread("samples\\battle_death.png")
    #     cls.battle_death2_png = cv2.imread("samples\\battle_death2.png")
    #     cls.loading_png = cv2.imread("samples\\loading.png")
    #
    # def setUp(self):
    #     self.r = __class__.runner
    #
    # # def tearDownClass(cls):
    # # def tearDown(self):
    #
    # def test_default_constructor(self):
    #     self.assertTrue(self.r)
    #
    # def test_battle(self):  # バトル中 非ラストスパート 生きてる
    #     for i in range(self.r.fixed_count + 1):
    #         self.r.eval(__class__.battle_png)
    #     print(self.r)
    #     self.assertEqual(self.r.state, PokemonUnite.GameState.BATTLE)
    #     self.assertEqual(self.r.remaining_time, 519)
    #     self.assertEqual(self.r.remaining_time_string, '08:39')
    #     self.assertEqual(self.r.mypoint, 19)
    #
    # def test_battle2(self):  # バトル中 ラストスパート 生きてる
    #     for i in range(self.r.fixed_count + 1):
    #         self.r.eval(__class__.battle2_png)
    #     print(self.r)
    #     self.assertEqual(self.r.state, PokemonUnite.GameState.BATTLE)
    #     self.assertEqual(self.r.remaining_time, 39)
    #     self.assertEqual(self.r.remaining_time_string, '00:39')
    #     self.assertEqual(self.r.mypoint, 2)
    #
    #
    # def test_battle3(self):  # バトル中 自得点満タン
    #     self.assertTrue(False, "未実装")
    #
    #
    # def test_battle_death(self):  # バトル中 非ラストスパート 死んでる
    #     for i in range(self.r.fixed_count + 1):
    #         self.r.eval(__class__.battle_death_png)
    #     print(self.r)
    #     self.assertEqual(self.r.state, PokemonUnite.GameState.BATTLE)
    #     self.assertEqual(self.r.remaining_time, 126)
    #     self.assertEqual(self.r.remaining_time_string, '02:06')
    #     self.assertEqual(self.r.mypoint, 16)
    #
    # def test_battle_death2(self):  # バトル中 ラストスパート 死んでる
    #     for i in range(self.r.fixed_count + 1):
    #         self.r.eval(__class__.battle_death2_png)
    #     print(self.r)
    #     self.assertEqual(self.r.state, PokemonUnite.GameState.BATTLE)
    #     self.assertEqual(self.r.remaining_time, 62)
    #     self.assertEqual(self.r.remaining_time_string, '01:02')
    #     # self.assertEqual(self.r.mypoint, 0) FIXME: スキャン間違い、死んで画面が暗転しているとき用の補正を考えないといけない

    # def test_loading(self):  # ロード中
    #     self.r.state_scan(__class__.loading_png)
    #     print(self.r)
    #     self.assertEqual(self.r.state, PokemonUnite.GameState.LOADING)

    # def test_resize(self):
    #     pokemon_names = [
    #         "talonflame1" # ,  "talonflame2", "talonflame3",
    #         # "wigglytuff1", "wigglytuff2",
    #         # "greninja1", "greninja2", "greninja3",
    #         # "cinderace1", "cinderace2", "cinderace3"
    #     ]
    #
    #     for pokemon_name  in pokemon_names:
    #         for f in pathlib.Path(".\\train_orig").glob(pokemon_name + "-*.jpg"):
    #             filename_base = f.stem
    #             img = cv2.imread(str(f))  # RGB
    #             cv2.imwrite(".\\train_output\\" + f.name, img[80:720, 880:1520])

    # def test_xywh(self):
    #     # xywh = self.xywh(640, 213, pokemon["xmin"], pokemon["xmax"], pokemon["ymin"], pokemon["ymax"])
    #
    #     xmin = 310
    #     xmax = 329
    #     ymin = 91
    #     ymax = 110
    #     xywh = self.xywh(640, 213, xmin, xmax, ymin, ymax)
    #     print(xywh)
    #
    #     # 0 0.499219 0.471831 0.029687 0.089202
    #     # {'x_center': 0.499219, 'y_center': 0.471831, 'width': 0.029687, 'height': 0.089202}

    def test_create_yolov5data(self):
        # labelImg data
        pokemons = [
            {"name": "talonflame1", "type": "talonflame1", "type_id": 0, "xmin": 302, "ymin": 81, "xmax": 333, "ymax": 102},
            {"name": "talonflame2", "type": "talonflame2", "type_id": 1, "xmin": 303, "ymin": 78, "xmax": 333, "ymax": 100},
            {"name": "talonflame3", "type": "talonflame3", "type_id": 2, "xmin": 300, "ymin": 76, "xmax": 336, "ymax": 103},
            {"name": "wigglytuff1", "type": "wigglytuff1", "type_id": 3, "xmin": 303, "ymin": 87, "xmax": 335, "ymax": 117},
            {"name": "wigglytuff2", "type": "wigglytuff2", "type_id": 4, "xmin": 303, "ymin": 83, "xmax": 337, "ymax": 117},
            {"name": "greninja1", "type": "greninja1", "type_id": 5, "xmin": 301, "ymin": 87, "xmax": 336, "ymax": 114},
            {"name": "greninja2", "type": "greninja2", "type_id": 6, "xmin": 304, "ymin": 87, "xmax": 336, "ymax": 111},
            {"name": "greninja3", "type": "greninja3", "type_id": 7, "xmin": 304, "ymin": 87, "xmax": 334, "ymax": 115},
            {"name": "cinderace1", "type": "cinderace1", "type_id": 8, "xmin": 306, "ymin": 90, "xmax": 328, "ymax": 110},
            {"name": "cinderace2", "type": "cinderace2", "type_id": 9, "xmin": 306, "ymin": 87, "xmax": 331, "ymax": 115},
            {"name": "cinderace3", "type": "cinderace3", "type_id": 10, "xmin": 306, "ymin": 78, "xmax": 331, "ymax": 115},
            {"name": "lucario1", "type": "lucario1", "type_id": 11, "xmin": 303, "ymin": 83, "xmax": 337, "ymax": 117},
            {"name": "snorlax1", "type": "snorlax1", "type_id": 12, "xmin": 302, "ymin": 76, "xmax": 336, "ymax": 117},
            {"name": "zeraora1", "type": "zeraora1", "type_id": 13, "xmin": 304, "ymin": 86, "xmax": 335, "ymax": 116},
            {"name": "cramorant1", "type": "cramorant1", "type_id": 14, "xmin": 303, "ymin": 90, "xmax": 332, "ymax": 117},
        ]

        cnt = 100

        [file.unlink() for file in pathlib.Path(".\\train_output\\train").glob("*.jpg")]
        [file.unlink() for file in pathlib.Path(".\\train_output\\train").glob("*.xml")]
        [file.unlink() for file in pathlib.Path(".\\train_output\\train").glob("*.txt")]

        [file.unlink() for file in pathlib.Path(".\\train_output\\valid").glob("*.jpg")]
        [file.unlink() for file in pathlib.Path(".\\train_output\\valid").glob("*.xml")]
        [file.unlink() for file in pathlib.Path(".\\train_output\\valid").glob("*.txt")]

        map_video = cv2.VideoCapture(".\\train_orig\\remoat_stadium1.mp4")
        map_frame_count = int(map_video.get(cv2.CAP_PROP_FRAME_COUNT))
        map_frames = random.sample(range(map_frame_count), cnt)
        for map_frame_num in map_frames:
            map_video.set(cv2.CAP_PROP_POS_FRAMES, map_frame_num)
            _map_video_read_ret, map_img = map_video.read()
            map_img = cv2.resize(map_img, (640, 213), interpolation=cv2.INTER_AREA)
            map_type_name = random.choices(["train", "valid"], k=1, weights=[0.8, 0.2])[0]
            map_output_dir = pathlib.Path(".\\train_output") / map_type_name
            map_output_image_path = map_output_dir / ("remoat_stadium1-" + str(map_frame_num) + ".jpg")
            cv2.imwrite(str(map_output_image_path), map_img)
            (map_output_dir / ("remoat_stadium1-" + str(map_frame_num) + ".txt")).touch()

        for pokemon in pokemons:
            video = cv2.VideoCapture(".\\train_orig\\" + pokemon["name"] + ".mp4")
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = random.sample(range(frame_count), 100)
            frames.sort()

            for frame_num in frames:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                _video_read_ret, img = video.read()
                img = cv2.resize(img, (640, 213), interpolation=cv2.INTER_AREA)

                type_name = random.choices(["train", "valid"], k=1, weights=[0.8, 0.2])[0]
                output_dir = pathlib.Path(".\\train_output") / type_name
                output_image_path = output_dir / (pokemon["name"] + "-" + str(frame_num) + ".jpg")
                cv2.imwrite(str(output_image_path), img)
                self.create_train_file(type_name, output_image_path, pokemon)

    # ref. https://stackoverflow.com/questions/12581437/python-random-sample-with-a-generator-iterable-iterator/53948071#53948071
    def iter_sample_fast(self, iterable, samplesize):
        results = []
        iterator = iter(iterable)
        # Fill in the first samplesize elements:
        try:
            for _ in range(samplesize):
                results.append(iterator.__next__())
        except StopIteration:
            raise ValueError("Sample larger than population.")
        random.shuffle(results)  # Randomize their positions
        for i, v in enumerate(iterator, samplesize):
            r = random.randint(0, i)
            if r < samplesize:
                results[r] = v  # at a decreasing rate, replace random items
        return results

    def create_train_file(self, folder, file, pokemon):
        database = "Unknown"
        width = 640
        height = 213
        depth = 3
        segmented = 0
        pose = "Unspecified"
        truncated = 0
        difficult = 0

        xywh = self.xywh(width, height, pokemon["xmin"], pokemon["xmax"], pokemon["ymin"], pokemon["ymax"])

        root = Element("annotation")
        elm_folder = SubElement(root, "folder")
        elm_folder.text = folder

        elm_filename = SubElement(root, "filename")
        elm_path = SubElement(root, "path")

        elm_source = SubElement(root, "source")
        elm_database = SubElement(elm_source, "width")
        elm_database.text = database

        elm_size = SubElement(root, "size")
        elm_size_width = SubElement(elm_size, "width")
        elm_size_width.text = str(width)
        elm_size_height = SubElement(elm_size, "height")
        elm_size_height.text = str(height)
        elm_size_depth = SubElement(elm_size, "depth")
        elm_size_depth.text = str(depth)

        elm_segmented = SubElement(root, "segmented")
        elm_segmented.text = str(segmented)

        elm_object = SubElement(root, "object")
        elm_object_name = SubElement(elm_object, "name")
        elm_object_name.text = str(pokemon["type"]) #pokemon["name"]
        elm_object_pose = SubElement(elm_object, "pose")
        elm_object_pose.text = str(pose)
        elm_object_truncated = SubElement(elm_object, "truncated")
        elm_object_truncated.text = str(truncated)
        elm_object_difficult = SubElement(elm_object, "difficult")
        elm_object_difficult.text = str(difficult)
        elm_object_bndbox = SubElement(elm_object, "bndbox")
        elm_object_bndbox_xmin = SubElement(elm_object_bndbox, "xmin")
        elm_object_bndbox_xmin.text = str(pokemon["xmin"])
        elm_object_bndbox_ymin = SubElement(elm_object_bndbox, "ymin")
        elm_object_bndbox_ymin.text = str(pokemon["ymin"])
        elm_object_bndbox_xmax = SubElement(elm_object_bndbox, "xmax")
        elm_object_bndbox_xmax.text = str(pokemon["xmax"])
        elm_object_bndbox_ymax = SubElement(elm_object_bndbox, "ymax")
        elm_object_bndbox_ymax.text = str(pokemon["ymax"])

        filename_base = file.stem
        elm_filename.text = file.name
        elm_path.text = str(file.resolve())

        xml_out_path = (file.parent / filename_base).with_suffix(".xml")
        with open(xml_out_path, 'w', encoding='utf-8', newline='\n') as f_xml:
            f_xml.write(tostring(root, "unicode"))

        txt_out_path = (file.parent / filename_base).with_suffix(".txt")
        txt = " ".join(map(str, [pokemon["type_id"], xywh["x_center"], xywh["y_center"], xywh["width"], xywh["height"]]))
        # print(txt)
        with open(txt_out_path, 'w', encoding='utf-8', newline='\n') as f_txt:
            f_txt.write(txt)

    def check_contours(self, frame):
        orig_cnts = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        mask = np.ones(frame.shape[:2], dtype="uint8") * 255
        cnts = []
        for c in orig_cnts:
            x, y, w, h = cv2.boundingRect(c)
            if all([cv2.contourArea(c) >= self.screen.value['contour_area_min'],
                    h <= self.screen.value['contour_height_max'],
                    h >= self.screen.value['contour_height_min'],
                    w <= self.screen.value['contour_width_max'],
                    w >= self.screen.value['contour_width_min'],
                    x > self.border_size * 1.2, x + w < (frame.shape[1] - (self.border_size * 1.2)),
                    y > self.border_size * 1.2, y + h < (frame.shape[0] - (
                        self.border_size * 1.2))]):  # only keep contours with good forms and not at border
                cnts.append(c)
            else:
                cv2.drawContours(mask, [c], -1, 0, -1)
        if len(cnts) == 0:
            return None
        i = 0
        for c in cnts:
            i += 1
            x, y, w, h = cv2.boundingRect(c)
            middle_x = x + (w / 2)
            middle_y = y + (h / 2)
            distance_to_middle = math.sqrt(
                ((frame.shape[1] / 2) - middle_x) ** 2 + ((frame.shape[0] / 2) - middle_y) ** 2)
            min_dist = 9999
            max_dist_to_middle = 0
            min_height_diff = 9999
            min_y_diff = 9999
            j = 0
            for c2 in cnts:  # Check distance and other relations to all other contours
                j += 1
                if i == j:
                    continue
                x2, y2, w2, h2 = cv2.boundingRect(c2)
                middle_x2 = x2 + (w2 / 2)
                middle_y2 = y2 + (h2 / 2)
                dist = math.sqrt((middle_x - middle_x2) ** 2 + (middle_y - middle_y2) ** 2)
                other_distance_to_middle = math.sqrt(
                    ((frame.shape[1] / 2) - middle_x2) ** 2 + ((frame.shape[0] / 2) - middle_y2) ** 2)
                if dist < min_dist:
                    min_dist = dist
                if other_distance_to_middle > max_dist_to_middle:
                    max_dist_to_middle = other_distance_to_middle
                if abs(h - h2) < min_height_diff:
                    min_height_diff = abs(h - h2)
                if abs(y - y2) < min_y_diff:
                    min_y_diff = abs(y - y2)
            if min_dist > self.screen.value[
                'contour_diff_dist_max']:  # Two contours far from each other: Take the one closer to middle; three or more: take the best group
                if len(cnts) == 2 and distance_to_middle == max_dist_to_middle:
                    cv2.drawContours(mask, [c], -1, 0, -1)
                if len(cnts) > 2:  # Problem: If there are 3 single contours all get deleted
                    cv2.drawContours(mask, [c], -1, 0, -1)
            if len(cnts) > 2 and (
                    min_height_diff > self.screen.value['contour_diff_height_max'] or min_y_diff > self.screen.value[
                'contour_diff_y_max']):
                cv2.drawContours(mask, [c], -1, 0, -1)
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        return frame

    def xywh(self, width, height, x_min, x_max, y_min, y_max):
        w = x_max - x_min
        h = y_max - y_min
        x_center = x_min + w / 2.
        y_center = y_min + h / 2.

        yolo_x = round(x_center / width, 6)
        yolo_y = round(y_center / height, 6)
        yolo_w = round(w / width, 6)
        yolo_h = round(h / height, 6)
        return {"x_center": yolo_x,
                "y_center": yolo_y,
                "width": yolo_w,
                "height": yolo_h
                }

if __name__ == '__main__':
    unittest.main()
