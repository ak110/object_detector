"""全般的なコードとかを適当に。"""
import os
import pathlib

import better_exceptions
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

better_exceptions.MAX_LENGTH = 128

BASE_DIR = pathlib.Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR.joinpath('results')

# 念のため(?)resultsフォルダを作っちゃう
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 念のため(?)カレントディレクトリを設定しちゃう
os.chdir(str(BASE_DIR))
