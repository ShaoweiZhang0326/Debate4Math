import os
import json

def merge_json_to_jsonl(input_dir, output_file):
    """
    将指定文件夹下所有子文件夹中的 JSON 文件合并到一个 JSONL 文件中。
    
    :param input_dir: 包含多个子文件夹的主目录。
    :param output_file: 输出的 JSONL 文件路径。
    """
    jsonl_data = []

    # 遍历主目录下的所有子文件夹
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)

        # 检查是否为目录
        if os.path.isdir(subdir_path):
            # 查找子文件夹中的 JSON 文件
            for file_name in os.listdir(subdir_path):
                if file_name.endswith(".json"):
                    json_path = os.path.join(subdir_path, file_name)

                    # 读取 JSON 文件
                    with open(json_path, "r", encoding="utf-8") as json_file:
                        try:
                            json_data = json.load(json_file)
                            jsonl_data.append(json_data)  # 将 JSON 数据添加到列表中
                        except json.JSONDecodeError as e:
                            print(f"Error decoding {json_path}: {e}")

    # 将数据写入 JSONL 文件
    with open(output_file, "w", encoding="utf-8") as jsonl_file:
        for entry in jsonl_data:
            jsonl_file.write(json.dumps(entry) + "\n")
    print(f"Successfully merged into {output_file}")

# 设置输入目录和输出文件路径
input_dir = "MATH/test"  # 主目录路径
output_file = "MATH_test.jsonl"  # 输出 JSONL 文件名

# 合并 JSON 文件
merge_json_to_jsonl(input_dir, output_file)
