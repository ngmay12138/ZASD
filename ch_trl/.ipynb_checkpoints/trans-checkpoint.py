import json

def convert_format(input_file_path, output_file_path):
    # 读取原始JSON文件
    with open(input_file_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # 初始化结果列表
    converted_data = []
    
    for item in original_data:
        # 提取图像路径（假设每个item只有一个图像）
        image_path = item["images"][0]
        
        # 提取用户消息内容，移除<image>标签
        user_content = item["messages"][0]["content"].replace("<image>\n", "")
        
        # 提取助手响应并转换为1/0（保持原样）
        assistant_response = item["messages"][1]["content"]
        solution = assistant_response  # 直接使用"0"或"1"
        
        # 构建转换后的item
        converted_item = {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": user_content}
                    ]
                }
            ],
            "solution": solution  # 直接使用"0"或"1"
        }
        
        converted_data.append(converted_item)
    
    # 写入新文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！共处理{len(converted_data)}个样本")
    print(f"结果已保存到: {output_file_path}")
    
    # 统计异常和正常样本数量
    normal_count = sum(1 for item in converted_data if item["solution"] == "0")
    anomaly_count = sum(1 for item in converted_data if item["solution"] == "1")
    print(f"正常样本数: {normal_count}")
    print(f"异常样本数: {anomaly_count}")

# 使用示例
if __name__ == "__main__":
    # 调用转换函数
    convert_format("sam4sft_mix.json", "mix_train.json")

# 在grpo_train.py中添加
# def test_reward_function():
#     from reward import simple_accuracy_reward
    
#     # 模拟completions数据
#     test_completions = [
#         [{"content": "这张图片中的物体是正常的。<answer>0</answer>"}],
#         [{"content": "这张图片中的物体存在异常。<answer>1</answer>"}],
#         [{"content": "图片正常"}],
#     ]
    
#     test_solutions = [
#         "<answer>0</answer>",
#         "<answer>1</answer>",
#         "<answer>0</answer>",
#     ]
    
#     rewards = simple_accuracy_reward(test_completions, test_solutions)
#     print(f"Test rewards: {rewards}")

# # 在训练前调用
# test_reward_function()

# import json
# import re


# # 更简洁的版本，直接处理文件
# def simple_fix():
#     with open("mvtec_ad_grpo_modified.json", 'r', encoding='utf-8') as f:
#         data = f.read()
    
#     # 直接替换所有模式
#     # 匹配 <answer> "0" </answer> 或 <answer> "1" </answer>
#     fixed_data = data.replace('"<answer>“0”</answer>"', '"<answer>0</answer>"')
#     fixed_data = fixed_data.replace('"<answer>“1”</answer>"', '"<answer>1</answer>"')
    
#     with open("mvtec_ad_grpo_modified_fixed.json", 'w', encoding='utf-8') as f:
#         f.write(fixed_data)
    
#     print("修复完成！")


# simple_fix()  # 方法2：直接文本替换