import json

def validate_jsonl(file_path: str):
    """
    读取 JSONL 文件并检查每一行是否符合 JSON 格式。
    
    :param file_path: JSONL 文件的路径。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            print(f"Validating JSONL file: {file_path}")
            line_number = 0
            valid_lines = 0
            invalid_lines = 0

            for line in f:
                line_number += 1
                line = line.strip()  # 去除多余的空格
                if not line:  # 跳过空行
                    continue
                
                try:
                    json.loads(line)  # 尝试解析为 JSON
                    valid_lines += 1
                except json.JSONDecodeError as e:
                    print(f"Line {line_number}: Invalid JSON - {e}")
                    invalid_lines += 1

            print(f"\nValidation completed:")
            print(f"  Total lines: {line_number}")
            print(f"  Valid lines: {valid_lines}")
            print(f"  Invalid lines: {invalid_lines}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# 示例用法
jsonl_file = "dataset/GSM8K/GSM8K_train.jsonl"  # 替换为你的 JSONL 文件路径
validate_jsonl(jsonl_file)
