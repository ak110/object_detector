"""全般的なコードとかを適当に。"""
import os
import pathlib

import better_exceptions
import matplotlib as mpl

mpl.use('Agg')
better_exceptions.MAX_LENGTH = 128

SCRIPT_PATH = pathlib.Path(__file__).resolve()
BASE_DIR = SCRIPT_PATH.parent
RESULT_DIR = BASE_DIR.joinpath('results')
LOG_PATH = RESULT_DIR.joinpath(SCRIPT_PATH.stem + '.log')

# 念のため(?)resultsフォルダを作っちゃう
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 念のため(?)カレントディレクトリを設定しちゃう
os.chdir(str(BASE_DIR))
