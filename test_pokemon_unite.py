import unittest
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
    #
    # def test_loading(self):  # ロード中
    #     self.r.state_scan(__class__.loading_png)
    #     print(self.r)
    #     self.assertEqual(self.r.state, PokemonUnite.GameState.LOADING)

    # labelImg
    def test_oops(self):
        print("oops!")

        # pokemon_name = "talonflame"
        # yolo_object_class = 0
        # yolo_x_center = 0.497500
        # yolo_y_center = 0.436875
        # yolo_width = 0.067500
        # yolo_height = 0.186250
        # xmin = 1113
        # ymin = 275
        # xmax = 1275
        # ymax = 424
        # self.create_train_files(pokemon_name, yolo_object_class, yolo_x_center, yolo_y_center, yolo_width, yolo_height, xmin, ymin, xmax, ymax)

        # pokemon_name = "wigglytuff"
        # yolo_object_class = 1
        # yolo_x_center = 0.495417
        # yolo_y_center = 0.481250
        # yolo_width = 0.065833
        # yolo_height = 0.170000
        # xmin = 1110
        # ymin = 317
        # xmax = 1268
        # ymax = 453
        # self.create_train_files(pokemon_name, yolo_object_class, yolo_x_center, yolo_y_center, yolo_width, yolo_height, xmin, ymin, xmax, ymax)

        # pokemon_name = "greninja"
        # yolo_object_class = 2
        # yolo_x_center = 0.497917
        # yolo_y_center = 0.481875
        # yolo_width = 0.067500
        # yolo_height = 0.161250
        # xmin = 1113
        # ymin = 319
        # xmax = 1275
        # ymax = 448
        # self.create_train_files(pokemon_name, yolo_object_class, yolo_x_center, yolo_y_center, yolo_width, yolo_height, xmin, ymin, xmax, ymax)

        pokemon_name = "cinderace"
        yolo_object_class = 3
        yolo_x_center = 0.496667
        yolo_y_center = 0.480625
        yolo_width = 0.065833
        yolo_height = 0.158750
        xmin = 1113
        ymin = 321
        xmax = 1271
        ymax = 448
        self.create_train_files(pokemon_name, yolo_object_class, yolo_x_center, yolo_y_center, yolo_width, yolo_height, xmin, ymin, xmax, ymax)



    def create_train_files(self, pokemon_name, yolo_object_class, yolo_x_center, yolo_y_center, yolo_width, yolo_height, xmin, ymin, xmax, ymax):
        folder = "train"
        database = "Unknown"
        width = 2400
        height = 800
        depth = 3
        segmented = 0
        pose = "Unspecified"
        truncated = 0
        difficult = 0

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
        elm_object_name.text = str(pokemon_name)
        elm_object_pose = SubElement(elm_object, "pose")
        elm_object_pose.text = str(pose)
        elm_object_truncated = SubElement(elm_object, "truncated")
        elm_object_truncated.text = str(truncated)
        elm_object_difficult = SubElement(elm_object, "difficult")
        elm_object_difficult.text = str(difficult)
        elm_object_bndbox = SubElement(elm_object, "bndbox")
        elm_object_bndbox_xmin = SubElement(elm_object_bndbox, "xmin")
        elm_object_bndbox_xmin.text = str(xmin)
        elm_object_bndbox_ymin = SubElement(elm_object_bndbox, "ymin")
        elm_object_bndbox_ymin.text = str(ymin)
        elm_object_bndbox_xmax = SubElement(elm_object_bndbox, "xmax")
        elm_object_bndbox_xmax.text = str(xmax)
        elm_object_bndbox_ymax = SubElement(elm_object_bndbox, "ymax")
        elm_object_bndbox_ymax.text = str(ymax)

        for f in pathlib.Path(".\\train").glob(pokemon_name + "-*.jpg"):
            filename_base = f.stem
            elm_filename.text = f.name
            elm_path.text = str(f.resolve())

            xml_out_path = (f.parent / filename_base).with_suffix(".xml")
            with open(xml_out_path, 'w', encoding='utf-8', newline='\n') as f_xml:
                f_xml.write(tostring(root, "unicode"))

            txt_out_path = (f.parent / filename_base).with_suffix(".txt")
            txt = " ".join(map(str, [yolo_object_class,  yolo_x_center, yolo_y_center, yolo_width, yolo_height]))
            # print(txt)
            with open(txt_out_path, 'w', encoding='utf-8', newline='\n') as f_txt:
                f_txt.write(txt)


    # def test_test(self):
    #     print("oops")
    #     img = cv2.imread("samples\\time1.png") #RGB
    #
    #     # heuristic process
    #     img = cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_AREA)
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #
    #     time_string = pytesseract.image_to_string(img, lang="eng")
    #     print(time_string)
    #
    #     cv2.imshow("TEST", img)
    #     cv2.waitKey(0)

    # def test_mypoint(self):
    #     # img = __class__.battle_png[686:716, 1865:1920]
    #     img = __class__.battle2_png[686:716, 1865:1920]
    #     img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    #
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #     mask = self.process_hsv_extract(img, [0, 0, 110], [180, 255, 255], 1, 5, False)
    #     img = cv2.bitwise_and(img, mask)
    #
    #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     # img = cv2.bitwise_not(img)
    #     # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    #
    #     self.r.ocr.SetImage(Image.fromarray(img))
    #     time_string = self.r.ocr.GetUTF8Text()
    #     print(time_string)
    #
    #     cv2.imshow("TEST", img)
    #     cv2.waitKey(0)


    # def test_mypoint(self):
    #     # img = __class__.battle_png[686:716, 1865:1920]
    #     img = __class__.battle2_png[686:716, 1865:1920]
    #     cv2.imwrite("mypoint_image.png", img)
    #
    #     img = cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_AREA)
    #
    #     frame_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #     # frame_cleaned = self.process_hsv_extract(frame_hsv, [0,0,80], [156,255,255], 1, 2, False)
    #     frame_cleaned = self.process_hsv_extract(frame_hsv, [0,0,100], [180,255,255], 1, 2, False)
    #
    #     # frame_cleaned = self.check_contours(frame_cleaned)
    #
    #     # frame_cleaned = cv2.erode(frame_cleaned, np.ones((2, 2), np.uint8))
    #     # frame_cleaned = cv2.dilate(frame_cleaned, np.ones((2, 2), np.uint8))
    #
    #     # frame_final = cv2.bitwise_not(frame_cleaned)  # Swap Black/White
    #     # frame_final = cv2.copyMakeBorder(frame_final, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255,255,255))
    #     frame_final = cv2.copyMakeBorder(frame_cleaned, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    #
    #     contours, hierarchy = cv2.findContours(cv2.cvtColor(frame_final, cv2.COLOR_RGB2GRAY),
    #                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    #     frame_final = cv2.bitwise_not(frame_final)  # Swap Black/White
    #     frame_final = cv2.drawContours(frame_final, contours, -1, color=(0, 0, 255), thickness=2)
    #
    #     self.r.ocr.SetImage(Image.fromarray(frame_final))
    #     time_string = self.r.ocr.GetUTF8Text()
    #     print(time_string)
    #
    #     cv2.imshow("TEST", frame_final)
    #     cv2.waitKey(0)
    #

    def check_contours(self, frame):
        orig_cnts = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        mask = np.ones(frame.shape[:2], dtype="uint8") * 255
        cnts = []
        for c in orig_cnts:
            x,y,w,h = cv2.boundingRect(c)
            if all([cv2.contourArea(c) >= self.screen.value['contour_area_min'],
                    h <= self.screen.value['contour_height_max'],
                    h >= self.screen.value['contour_height_min'],
                    w <= self.screen.value['contour_width_max'],
                    w >= self.screen.value['contour_width_min'],
                    x > self.border_size * 1.2, x+w < (frame.shape[1] - (self.border_size * 1.2)),
                    y > self.border_size * 1.2, y+h < (frame.shape[0] - (self.border_size * 1.2))]): # only keep contours with good forms and not at border
                cnts.append(c)
            else:
                cv2.drawContours(mask, [c], -1, 0, -1)
        if len(cnts) == 0:
            return None
        i = 0
        for c in cnts:
            i += 1
            x,y,w,h = cv2.boundingRect(c)
            middle_x = x + (w / 2)
            middle_y = y + (h / 2)
            distance_to_middle = math.sqrt(((frame.shape[1]/2) - middle_x)**2 + ((frame.shape[0]/2) - middle_y)**2)
            min_dist = 9999
            max_dist_to_middle = 0
            min_height_diff = 9999
            min_y_diff = 9999
            j = 0
            for c2 in cnts: # Check distance and other relations to all other contours
                j += 1
                if i == j:
                    continue
                x2,y2,w2,h2 = cv2.boundingRect(c2)
                middle_x2 = x2 + (w2 / 2)
                middle_y2 = y2 + (h2 / 2)
                dist = math.sqrt((middle_x - middle_x2)**2 + (middle_y - middle_y2)**2)
                other_distance_to_middle = math.sqrt(((frame.shape[1]/2) - middle_x2)**2 + ((frame.shape[0]/2) - middle_y2)**2)
                if dist < min_dist:
                    min_dist = dist
                if other_distance_to_middle > max_dist_to_middle:
                    max_dist_to_middle = other_distance_to_middle
                if abs(h - h2) < min_height_diff:
                    min_height_diff = abs(h - h2)
                if abs(y - y2) < min_y_diff:
                    min_y_diff = abs(y - y2)
            if min_dist > self.screen.value['contour_diff_dist_max']: # Two contours far from each other: Take the one closer to middle; three or more: take the best group
                if len(cnts) == 2 and distance_to_middle == max_dist_to_middle:
                    cv2.drawContours(mask, [c], -1, 0, -1)
                if len(cnts) > 2: # Problem: If there are 3 single contours all get deleted
                    cv2.drawContours(mask, [c], -1, 0, -1)
            if len(cnts) > 2 and (min_height_diff > self.screen.value['contour_diff_height_max'] or min_y_diff > self.screen.value['contour_diff_y_max']):
                cv2.drawContours(mask, [c], -1, 0, -1)
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        return frame



if __name__ == '__main__':
    unittest.main()
