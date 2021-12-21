pokemon_unite.py


---
機能


ポケモンユナイト。

試合ごとに動画を小分けにしたい。学び。

OBSの操作するのが面倒。忘れる。

このスクリプトをOBSに仕込む。

ローディング画面が出るたびに、リプレイバッファ保存が自動実行される。

これは便利

---
使い方

1. NoxとOBSを起動する
2. OBSで[リプレイバッファ開始]押下
3. ポケモンユナイトで遊ぶ
4. 自動でリプレイが区切られて保存されていく
5. 遊び終わったらOBSで[リプレイバッファ停止]押下


---
事前インストール

* Nox Player
* Pokémon UNITE
* OBS 27.1.3 (64-bit, windows)
* Python 3.6.8 x64 入れてパス通す
* opencv-4.5.4-vc14_vc15.exe 入れてパス通す
* tesseract-ocr-w64-setup-v5.0.0.20211201.exe 入れてパス通す
* Pythonパッケージ

````
>pip list
Package         Version
--------------- --------
cycler          0.11.0
Cython          0.29.26
debugpy         1.5.1
kiwisolver      1.3.1
matplotlib      3.3.4
numpy           1.19.5
opencv-python   4.5.4.60
Pillow          8.4.0
pip             18.1
pyocr           0.8.1
pyparsing       3.0.6
pytesseract     0.3.8
python-dateutil 2.8.2
setuptools      40.6.2
six             1.16.0
````

---
Nox設定

解像度をカスタムで 2400x800 にする

---
OBS設定

1. OBSを起動し [設定] 押下
2. 左ペイン [映像] 押下
3. 基本(キャンバス)解像度 "2400x800" を指定
4. 出力(スケーリング)解像度 "1200x400" を指定
5. 左ペイン [出力] 押下
6. [リプレイバッファ]タブの最大リプレイ時間に "1800s" を指定
7. ソースにウインドウキャプチャを追加
   1. [ウインドウ] にNoxを指定
   2. 名前に "unite" を指定
8. OBSのプレビュー画面にソースのNox画面がピッタリ貼り付くよう位置合わせする

---
スクリプトインストール

1. c:\src\unite に本スクリプト一式を置く
2. OBSを起動し [ツール] - [スクリプト] 押下
3. [Pythonの設定] タブを開き、 [Python インストールパス (64bit)] を指定
4. [スクリプトタブ] を開き 、[＋]押下し、c:\src\unite\pokemon_unite.py を指定


