# def simple_accuracy_reward(completions, solution, **kwargs):
#     """只检查answer是否正确的奖励函数"""
#     import re
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
    
#     for content, sol in zip(contents, solution):
#         reward = 0.0
#         try:
#             # 从solution中提取真实答案，移除可能的引号
#             sol_match = re.search(r'<answer>(.*?)</answer>', sol)
#             if sol_match:
#                 # 移除引号和空格
#                 ground_truth = sol_match.group(1).strip().strip('"').strip("'").lower()
#             else:
#                 ground_truth = sol.strip().strip('"').strip("'").lower()
            
#             # 从生成内容中提取答案
#             content_match = re.search(r'<answer>(.*?)</answer>', content)
#             if content_match:
#                 # 同样移除引号和空格
#                 model_answer = content_match.group(1).strip().strip('"').strip("'").lower()
                
#                 # 简单的字符串匹配
#                 if model_answer == ground_truth:
#                     reward = 1.0
#             else:
#                 # 如果没有找到<answer>标签，尝试在整个内容中查找
#                 # 移除ground_truth的引号后再比较
#                 clean_ground_truth = ground_truth.strip('"').strip("'")
#                 if clean_ground_truth in content.lower():
#                     reward = 1.0
                    
#         except Exception as e:
#             # 出现异常时给予0奖励
#             reward = 0.0
            
#         rewards.append(reward)
    
#     return rewards
########################################################################################
# import re
# import math
# from typing import List, Dict, Any
# import numpy as np
# from collections import Counter

# class EnhancedRewardSystem:
#     """多维度奖励系统，避免奖励稀疏问题"""
    
#     def __init__(self, weights: Dict[str, float] = None):
#         """
#         初始化奖励系统
        
#         Args:
#             weights: 各奖励组件的权重配置
#         """
#         self.default_weights = {
#             'accuracy': 0.9,      # 答案准确性
#             'format': 0.1,       # 格式规范性
#             'confidence': 0,    # 置信度
#             'completeness': 0, # 回答完整性
#             'relevance': 0,     # 相关性
#             'consistency': 0,  # 一致性
#             'creativity': 0,   # 创造性（如果适用）
#         }
#         self.weights = weights or self.default_weights
        
#         # 验证权重和为1
#         total_weight = sum(self.weights.values())
#         if abs(total_weight - 1.0) > 1e-6:
#             raise ValueError(f"权重总和必须为1，当前为{total_weight}")
    
#     def compute_comprehensive_reward(self, completions: List[List[Dict]], 
#                                      solution: List[str], 
#                                      questions: List[str] = None,
#                                      **kwargs) -> List[float]:
#         """
#         计算综合奖励值
        
#         Args:
#             completions: 模型生成内容列表
#             solution: 标准答案列表
#             questions: 对应的问题列表（可选，用于相关性评估）
            
#         Returns:
#             奖励值列表
#         """
#         contents = [completion[0]["content"] for completion in completions]
#         rewards = []
        
#         for i, (content, sol) in enumerate(zip(contents, solution)):
#             question = questions[i] if questions else None
            
#             # 计算各个维度的奖励
#             reward_components = {
#                 'accuracy': self._accuracy_reward(content, sol),
#                 'format': self._format_reward(content, sol),
#                 'confidence': self._confidence_reward(content),
#                 'completeness': self._completeness_reward(content, sol),
#                 'relevance': self._relevance_reward(content, question) if question else 0.5,
#                 'consistency': self._consistency_reward(content),
#                 'creativity': self._creativity_reward(content, sol) if kwargs.get('allow_creativity', False) else 0.0,
#             }
            
#             # 加权求和得到最终奖励
#             total_reward = sum(
#                 reward_components[component] * self.weights[component]
#                 for component in self.weights
#                 if component in reward_components
#             )
            
#             # 添加可选的正则化
#             if kwargs.get('normalize', False):
#                 total_reward = self._normalize_reward(total_reward)
            
#             # 添加可选的基础奖励（避免零奖励）
#             base_reward = kwargs.get('base_reward', 0.1)
#             total_reward = max(total_reward, base_reward)
            
#             rewards.append(total_reward)
        
#         return rewards
    
#     def _accuracy_reward(self, content: str, solution: str) -> float:
#         """答案准确性奖励"""
#         try:
#             # 提取真实答案
#             sol_match = re.search(r'<answer>(.*?)</answer>', solution)
#             ground_truth = sol_match.group(1).strip().lower() if sol_match else solution.strip().lower()
            
#             # 提取模型答案
#             content_match = re.search(r'<answer>(.*?)</answer>', content)
#             if content_match:
#                 model_answer = content_match.group(1).strip().lower()
                
#                 # 1. 精确匹配
#                 if model_answer == ground_truth:
#                     return 1.0
                
#                 # 2. 模糊匹配（处理大小写、空格、标点差异）
#                 if self._fuzzy_match(model_answer, ground_truth):
#                     return 0.5              
#                 return 0.0
#             else:
#                 return 0.0
                
#         except Exception:
#             return 0.0
    
#     def _format_reward(self, content: str, solution: str) -> float:
#         """格式规范性奖励"""
#         score = 0.0
        
#         # 1. 检查是否有<answer>标签
#         if re.search(r'<answer>.*?</answer>', content, re.DOTALL):
#             score += 0.3        
#         # 2. 检查是否有推理过程
#         if re.search(r'<description>.*?</description>', content, re.DOTALL):
#             score += 0.3 
#         if re.search(r'<think>.*?</think>', content, re.DOTALL):
#             score += 0.3       
#         # 3. 检查格式整洁性（没有多余的空行、奇怪的符号等）
#         if not re.search(r'\n\s*\n\s*\n', content):  # 没有连续多个空行
#             score += 0.1
        
#         return min(score, 1.0)
    
#     def _confidence_reward(self, content: str) -> float:
#         """置信度奖励（基于语言确定性）"""
#         confidence_indicators = [
#             'certainly', 'definitely', 'surely', 'clearly', 'obviously',
#             'without doubt', 'undoubtedly', 'absolutely'
#         ]
        
#         hesitation_indicators = [
#             'maybe', 'perhaps', 'possibly', 'might', 'could',
#             'I think', 'I believe', 'probably', 'likely'
#         ]
        
#         content_lower = content.lower()
        
#         # 统计确定性词汇
#         confidence_count = sum(1 for word in confidence_indicators if word in content_lower)
#         hesitation_count = sum(1 for word in hesitation_indicators if word in content_lower)
        
#         total = confidence_count + hesitation_count
        
#         if total == 0:
#             return 0.5  # 中性
        
#         confidence_score = confidence_count / total
        
#         # 调整分数，避免极端情况
#         return 0.3 + 0.4 * confidence_score  # 映射到0.3-0.7之间
    
#     def _completeness_reward(self, content: str, solution: str) -> float:
#         """回答完整性奖励"""
#         score = 0.0
        
#         # 1. 答案长度合理性
#         content_length = len(content.strip())
#         if 500 <= content_length <= 700:  # 合理长度范围
#             score += 1.0
#         elif content_length < 500:
#             score += 0.5
#         else:
#             score += 0
        
#         # # 2. 是否回答了所有部分（对于多部分问题）
#         # # 提取solution中的所有关键部分
#         # solution_parts = re.findall(r'<answer>(.*?)</answer>', solution, re.DOTALL)
#         # if solution_parts:
#         #     solution_text = ' '.join(solution_parts).lower()
#         #     # 简单的关键词覆盖率
#         #     important_words = set(re.findall(r'\b\w{4,}\b', solution_text))
#         #     content_words = set(re.findall(r'\b\w{4,}\b', content.lower()))
            
#         #     if important_words:
#         #         coverage = len(important_words.intersection(content_words)) / len(important_words)
#         #         score += 0.4 * min(coverage, 1.0)
#         # else:
#         #     # 如果没有明确的多个部分，给基础分
#         #     score += 0.3    
        
#         return min(score, 1.0)
    
#     def _relevance_reward(self, content: str, question: str) -> float:
#         """相关性奖励（回答是否切题）"""
#         if not question:
#             return 0.5
        
#         # 简单方法：计算共同词汇比例
#         question_words = set(re.findall(r'\b\w+\b', question.lower()))
#         content_words = set(re.findall(r'\b\w+\b', content.lower()))
        
#         # 移除常见停用词
#         stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
#         question_words = question_words - stop_words
#         content_words = content_words - stop_words
        
#         if not question_words:
#             return 0.5
        
#         overlap = len(question_words.intersection(content_words))
#         relevance = overlap / len(question_words)
        
#         # 使用sigmoid函数平滑
#         return 1 / (1 + math.exp(-10 * (relevance - 0.5)))
    
#     def _consistency_reward(self, content: str) -> float:
#         """一致性奖励（回答内部是否一致）"""
#         sentences = re.split(r'[.!?]+', content)
#         sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
#         if len(sentences) < 2:
#             return 0.5
        
#         # 检查关键词的一致性
#         keyword_counter = Counter()
#         for sentence in sentences:
#             words = re.findall(r'\b\w{4,}\b', sentence.lower())
#             keyword_counter.update(words)
        
#         # 计算关键词的重复率（适度重复是好的，但过度重复可能有问题）
#         total_keywords = sum(keyword_counter.values())
#         unique_keywords = len(keyword_counter)
        
#         if total_keywords == 0:
#             return 0.5
        
#         repetition_rate = total_keywords / unique_keywords
        
#         # 理想重复率在1.2-2.0之间
#         if 1.2 <= repetition_rate <= 2.0:
#             return 0.8
#         elif repetition_rate < 1.1:
#             return 0.4  # 几乎没有重复，可能不够连贯
#         else:
#             return 0.6  # 有些过度重复
    
#     def _creativity_reward(self, content: str, solution: str) -> float:
#         """创造性奖励（如果允许创造性回答）"""
#         # 检查是否提供了标准答案之外的额外信息
#         solution_keywords = set(re.findall(r'\b\w{4,}\b', solution.lower()))
#         content_keywords = set(re.findall(r'\b\w{4,}\b', content.lower()))
        
#         extra_keywords = content_keywords - solution_keywords
        
#         if not extra_keywords:
#             return 0.0
        
#         # 额外的相关关键词越多，创造性得分越高
#         creativity_score = min(len(extra_keywords) / 5, 1.0)
        
#         # 但也要检查这些额外信息是否相关（简单检查）
#         # 这里可以更复杂，比如使用词向量计算相关性
        
#         return creativity_score * 0.8  # 创造性奖励不应过高，避免偏离准确性
    
#     def _fuzzy_match(self, str1: str, str2: str) -> bool:
#         """模糊字符串匹配"""
#         # 移除空格、标点，转为小写
#         def clean(s):
#             return re.sub(r'[^\w]', '', s).lower()
        
#         return clean(str1) == clean(str2)
    
#     def _partial_match(self, model_answer: str, ground_truth: str) -> bool:
#         """部分匹配（对于多部分答案）"""
#         # 如果真实答案由多个部分组成
#         if ';' in ground_truth or ',' in ground_truth:
#             truth_parts = re.split(r'[;,]\s*', ground_truth)
#             model_parts = re.split(r'[;,]\s*', model_answer)
            
#             # 检查是否有共同部分
#             common = set(p.strip() for p in truth_parts) & set(p.strip() for p in model_parts)
#             return len(common) > 0
        
#         return False
    
#     def _normalize_reward(self, reward: float) -> float:
#         """奖励值正则化"""
#         # 使用sigmoid函数将奖励值压缩到(0,1)区间
#         return 1 / (1 + math.exp(-8 * (reward - 0.5)))


# # 提供简化接口，保持向后兼容性
# def comprehensive_accuracy_reward(completions, solution, **kwargs):
#     """综合奖励函数（推荐使用）"""
#     reward_system = EnhancedRewardSystem(
#         weights=kwargs.get('weights', None)
#     )
    
#     questions = kwargs.get('questions', None)
#     allow_creativity = kwargs.get('allow_creativity', False)
    
#     return reward_system.compute_comprehensive_reward(
#         completions, solution, questions, 
#         allow_creativity=allow_creativity,
#         normalize=kwargs.get('normalize', True),
#         base_reward=kwargs.get('base_reward', 0.1)
#     )


# def tiered_reward_system(completions, solution, **kwargs):
#     """
#     分层奖励系统
    
#     根据答案质量分为不同层级：
#     1. 优秀答案：完全正确且格式良好 (0.8-1.0)
#     2. 良好答案：基本正确但有微小问题 (0.6-0.8)
#     3. 中等答案：部分正确 (0.4-0.6)
#     4. 较差答案：有少量正确信息 (0.2-0.4)
#     5. 错误答案：完全错误 (0.0-0.2)
#     """
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
    
#     for content, sol in zip(contents, solution):
#         # 提取答案
#         sol_match = re.search(r'<answer>(.*?)</answer>', sol)
#         ground_truth = sol_match.group(1).strip().lower() if sol_match else sol.strip().lower()
        
#         content_match = re.search(r'<answer>(.*?)</answer>', content)
#         model_answer = content_match.group(1).strip().lower() if content_match else ""
        
#         # 层级1：完全正确且格式完美
#         if model_answer and model_answer == ground_truth:
#             # 检查格式
#             if re.search(r'<reasoning>.*?</reasoning>', content, re.DOTALL):
#                 rewards.append(1.0)  # 完美答案
#             else:
#                 rewards.append(0.9)  # 答案正确但格式不完整
        
#         # 层级2：模糊匹配基本正确
#         elif model_answer and re.sub(r'[^\w]', '', model_answer).lower() == re.sub(r'[^\w]', '', ground_truth).lower():
#             rewards.append(0.7)
        
#         # 层级3：部分匹配
#         elif model_answer and any(part in ground_truth for part in model_answer.split() if len(part) > 3):
#             rewards.append(0.5)
        
#         # 层级4：包含正确答案但未在<answer>标签中
#         elif ground_truth in content.lower():
#             rewards.append(0.3)
        
#         # 层级5：完全错误但有相关内容
#         elif any(word in content.lower() for word in ground_truth.split() if len(word) > 3):
#             rewards.append(0.1)
        
#         # 层级6：完全错误
#         else:
#             rewards.append(0.0)
    
#     return rewards


# def adaptive_reward_with_memory(completions, solution, **kwargs):
#     """
#     自适应奖励函数，带有历史记忆
    
#     可以根据历史表现调整奖励标准
#     """
#     # 获取历史表现（如果有）
#     history = kwargs.get('history', [])
    
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
    
#     for i, (content, sol) in enumerate(zip(contents, solution)):
#         # 基本准确性检查
#         sol_match = re.search(r'<answer>(.*?)</answer>', sol)
#         ground_truth = sol_match.group(1).strip().lower() if sol_match else sol.strip().lower()
        
#         content_match = re.search(r'<answer>(.*?)</answer>', content)
#         if content_match:
#             model_answer = content_match.group(1).strip().lower()
            
#             # 根据历史表现调整阈值
#             avg_history = np.mean(history) if history else 0.5
            
#             if model_answer == ground_truth:
#                 base_reward = 1.0
#             elif self._fuzzy_match(model_answer, ground_truth):
#                 base_reward = 0.8
#             else:
#                 base_reward = 0.0
#         else:
#             base_reward = 0.0
        
#         # 自适应调整：如果历史表现好，提高标准；如果历史表现差，降低标准
#         adjustment = 1.0
#         if history:
#             if avg_history > 0.7:  # 表现好，提高标准
#                 adjustment = 0.9
#             elif avg_history < 0.3:  # 表现差，降低标准
#                 adjustment = 1.1
        
#         rewards.append(min(base_reward * adjustment, 1.0))
    
#     return rewards


# # 保持原有简单函数的兼容性
# def simple_accuracy_reward(completions, solution, **kwargs):
#     """只检查answer是否正确的奖励函数（保持向后兼容）"""
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
    
#     for content, sol in zip(contents, solution):
#         reward = 0.0
#         try:
#             # 从solution中提取真实答案
#             sol_match = re.search(r'<answer>(.*?)</answer>', sol)
#             ground_truth = sol_match.group(1).strip().lower() if sol_match else sol.strip().lower()
            
#             # 从生成内容中提取答案
#             content_match = re.search(r'<answer>(.*?)</answer>', content)
#             if content_match:
#                 model_answer = content_match.group(1).strip().lower()
                
#                 # 简单的字符串匹配
#                 if model_answer == ground_truth:
#                     reward = 1.0
#                 else:
#                     reward = 0.0
#             else:
#                 # 如果没有找到<answer>标签，尝试在整个内容中查找
#                 if ground_truth in content.lower():
#                     reward = 1.0
#                 else:
#                     reward = 0.0
                    
#         except Exception as e:
#             # 出现异常时给予0奖励
#             reward = 0.0
            
#         rewards.append(reward)
    
#     return rewards
###########################################################################
import re
import math
from typing import List, Dict, Any
import numpy as np
from collections import Counter

class EnhancedRewardSystem:
    """多维度奖励系统（精简版，仅保留准确性和格式两个维度）"""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        初始化奖励系统
        
        Args:
            weights: 各奖励组件的权重配置，默认为 {'accuracy': 0.9, 'format': 0.1}
        """
        self.default_weights = {
            'accuracy': 0.9,      # 答案准确性
            'format': 0.1,        # 格式规范性
        }
        self.weights = weights or self.default_weights
        
        # 验证权重和为1
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"权重总和必须为1，当前为{total_weight}")
    
    def compute_comprehensive_reward(self, completions: List[List[Dict]], 
                                     solution: List[str], 
                                     questions: List[str] = None,
                                     **kwargs) -> List[float]:
        """
        计算综合奖励值（仅使用准确性和格式）
        
        Args:
            completions: 模型生成内容列表
            solution: 标准答案列表
            
        Returns:
            奖励值列表
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for i, (content, sol) in enumerate(zip(contents, solution)):
            # 仅计算准确性和格式两个维度的奖励
            reward_components = {
                'accuracy': self._accuracy_reward(content, sol),
                'format': self._format_reward(content, sol),
            }
            
            # 加权求和得到最终奖励
            total_reward = sum(
                reward_components[component] * self.weights.get(component, 0.0)
                for component in reward_components
            )
            
            # 添加可选的正则化
            if kwargs.get('normalize', False):
                total_reward = self._normalize_reward(total_reward)
            
            # 添加可选的基础奖励（避免零奖励）
            base_reward = kwargs.get('base_reward', 0.1)
            total_reward = max(total_reward, base_reward)
            
            rewards.append(total_reward)
        
        return rewards
    
    def _accuracy_reward(self, content: str, solution: str) -> float:
        """答案准确性奖励"""
        try:
            # 提取真实答案
            sol_match = re.search(r'<answer>(.*?)</answer>', solution)
            ground_truth = sol_match.group(1).strip().lower() if sol_match else solution.strip().lower()
            
            # 提取模型答案
            content_match = re.search(r'<answer>(.*?)</answer>', content)
            if content_match:
                model_answer = content_match.group(1).strip().lower()
                
                # 1. 精确匹配
                if model_answer == ground_truth:
                    return 1.0
                
                # 2. 模糊匹配（处理大小写、空格、标点差异）
                if self._fuzzy_match(model_answer, ground_truth):
                    return 0.5              
                return 0.0
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _format_reward(self, content: str, solution: str) -> float:
        """格式规范性奖励"""
        score = 0.0
        
        # 1. 检查是否有<answer>标签
        if re.search(r'<answer>.*?</answer>', content, re.DOTALL):
            score += 0.3        
        # 2. 检查是否有推理过程
        if re.search(r'<description>.*?</description>', content, re.DOTALL):
            score += 0.3 
        if re.search(r'<think>.*?</think>', content, re.DOTALL):
            score += 0.3       
        # 3. 检查格式整洁性（没有多余的空行、奇怪的符号等）
        if not re.search(r'\n\s*\n\s*\n', content):  # 没有连续多个空行
            score += 0.1
        
        return min(score, 1.0)
    
    # 辅助方法
    def _fuzzy_match(self, str1: str, str2: str) -> bool:
        """模糊字符串匹配"""
        # 移除空格、标点，转为小写
        def clean(s):
            return re.sub(r'[^\w]', '', s).lower()
        
        return clean(str1) == clean(str2)
    
    def _normalize_reward(self, reward: float) -> float:
        """奖励值正则化"""
        # 使用sigmoid函数将奖励值压缩到(0,1)区间
        return 1 / (1 + math.exp(-8 * (reward - 0.5)))