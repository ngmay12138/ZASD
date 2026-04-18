import json

def convert_sentence_pairs_to_format(input_file, output_file):
    # 读取原始数据，支持 JSON 数组 或 JSON Lines
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        # 尝试一次性解析整个文件（JSON 数组）
        try:
            data = json.load(f)
            # 如果解析成功但结果是单个字典，转为列表
            if isinstance(data, dict):
                data = [data]
        except json.JSONDecodeError as e:
            # 如果出错且错误信息包含 "Extra data"，则按行读取
            if "Extra data" in str(e):
                f.seek(0)  # 重置文件指针
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        data.append(obj)
                    except json.JSONDecodeError as line_error:
                        print(f"Warning: Skipping invalid JSON at line {line_num}: {line_error}")
            else:
                # 其他 JSON 错误直接抛出
                raise
    
    # 构建目标格式的列表
    output_data = []
    for item in data:
        text1 = item.get('text1', '')
        text2 = item.get('text2', '')
        label = str(item.get('label', 0))  # 确保 label 为字符串
        
        # 构造 user 的 content
        user_content = (
            "You are an industrial anomaly detection expert. Please determine, based on the provided 'standard answer,' whether the 'predicted answer' is semantically the same as the 'standard answer.' If they are the same, output 1; otherwise, output 0.\n"
            f'            Standard answer:"{text1}"\n'
            f'            Predicted answer:"{text2}"'
        )
        
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": label}
        ]
        
        output_data.append({"messages": messages})
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

# 使用示例
convert_sentence_pairs_to_format('data4judger/sentence_pairs.json', 'data4judger/judger.json')