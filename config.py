"""全般的なコードとかを適当に。"""
import os
import pathlib

import better_exceptions

better_exceptions.MAX_LENGTH = 100

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULT_DIR = BASE_DIR / 'results'

# 念のため(?)resultsフォルダを作っちゃう
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 念のため(?)カレントディレクトリを設定しちゃう
os.chdir(str(BASE_DIR))
