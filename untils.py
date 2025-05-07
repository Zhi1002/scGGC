import os
import pandas as pd

def check_and_create_paths():
    # 创建必要的目录
    dirs = [
        os.path.join(os.path.dirname(__file__), 'data/'),
        os.path.join(os.path.dirname(__file__), 'data/raw/'),
        os.path.join(os.path.dirname(__file__), 'data/processed/'),
        os.path.join(os.path.dirname(__file__), 'models/'),
        os.path.join(os.path.dirname(__file__), 'utils/'),
        os.path.join(os.path.dirname(__file__), 'configs/')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)