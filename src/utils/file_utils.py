import os
import pandas as pd

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_dataframe(df, filename, directory='data/processed'):
    """保存DataFrame到指定目录"""
    ensure_dir(directory)
    full_path = os.path.join(directory, filename)
    df.to_csv(full_path, index=False)
    return full_path 