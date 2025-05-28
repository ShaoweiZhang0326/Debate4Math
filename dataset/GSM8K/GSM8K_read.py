import pandas as pd

# 读取单个 Parquet 文件
# df = pd.read_parquet('/home/swzhang/debate/dataset/GSM8K/train-00000-of-00001.parquet')

import pandas as pd
import json

def parquet_to_jsonl(parquet_file: str, jsonl_file: str):
    """
    将 Parquet 文件转换为 JSONL 文件。
    
    :param parquet_file: 输入的 Parquet 文件路径。
    :param jsonl_file: 输出的 JSONL 文件路径。
    """
    try:
        # 读取 Parquet 文件到 Pandas DataFrame
        df = pd.read_parquet(parquet_file)
        
        # 打开 JSONL 文件写入模式
        with open(jsonl_file, 'w', encoding='utf-8') as jsonl:
            for _, row in df.iterrows():
                # 将每行转换为 JSON 格式并写入 JSONL 文件
                jsonl.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
        
        print(f"Successfully converted {parquet_file} to {jsonl_file}")
    except Exception as e:
        print(f"Error occurred: {e}")

# 示例用法
parquet_file = "ataset/GSM8K/test-00000-of-00001.parquet"  # 替换为你的 Parquet 文件路径
jsonl_file = "GSM8K_test.jsonl"    # 替换为目标 JSONL 文件路径

parquet_to_jsonl(parquet_file, jsonl_file)