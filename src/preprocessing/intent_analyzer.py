import pandas as pd
import json
import os
import requests
from tqdm import tqdm
import time

class IntentAnalyzer:
    def __init__(self):
        # 定义意图及其对应的描述
        self.intent_labels = {
            # 1. 提供个人信息
            '提供姓名': '用户提供自己的姓氏或全名',
            '说明身份关系': '用户说明与债务人的关系或身份',
            
            # 2. 联系方式
            '提供联系方式': '用户提供电话、微信等联系方式',
            '询问联系方式': '用户询问对方的联系方式',
            
            # 3. 债务金额与类型
            '说明具体金额': '用户提供具体的欠款金额',
            '说明债务类型': '用户说明债务类型（如信用卡、网贷等）',
            '说明总欠款': '用户说明所有债务的总金额',
            
            # 4. 协商还款
            '表达还款意愿': '用户表达还款或暂时无法还款的意愿',
            '讨论还款方案': '用户讨论分期、延期等还款方案',
            
            # 5. 时间安排
            '约定具体时间': '用户约定具体的联系或处理时间',
            '说明时间限制': '用户说明自己的时间限制或不便时段',
            
            # 6. 信任与安全
            '表达安全疑虑': '用户表达对服务真实性的怀疑',
            '要求身份验证': '用户要求验证对方身份',
            
            # 7. 操作指导
            '询问自主操作': '用户询问如何自己操作处理',
            '寻求操作指导': '用户寻求具体的操作指导',
            
            # 8. 费用问题
            '询问费用': '用户询问服务费用相关问题',
            '要求免费服务': '用户表示只考虑免费服务',
            
            # 9. 催收与骚扰
            '投诉催收': '用户投诉催收骚扰问题',
            '要求拦截': '用户询问如何拦截催收电话',
            
            # 10. 法律问题
            '法律咨询': '用户咨询法律相关问题',
            '询问刑责': '用户询问是否会承担刑事责任',
            
            # 11. 信用与征信
            '询问征信影响': '用户询问对征信的影响',
            '咨询信用修复': '用户咨询如何修复信用',
            
            # 12. 平台与银行
            '提及具体平台': '用户提到具体的借贷平台',
            '提及银行': '用户提到具体的银行机构',
            
            # 其他高频意图
            '拒绝服务': '用户明确表示不需要服务',
            '表达情绪': '用户表达压力、焦虑等情绪',
            '表达模糊需求': '用户表达不确定或模糊的需求',
            '反馈技术问题': '用户反馈系统操作等技术问题',
            
            # 其他
            '其他': '不属于上述任何类别的内容'
        }
        
        self.api_url = "https://xiaoai.plus/v1/chat/completions"
        self.processed_dir = 'data/processed'
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": ""
        }
        
        # 调整请求限制参数
        self.retry_count = 5
        self.retry_delay = 5  # 增加重试间隔到5秒
        self.request_interval = 1.0  # 增加请求间隔到1秒
        self.last_request_time = 0
        self.timeout = 60  # 增加超时时间到60秒
    
    def _wait_for_rate_limit(self):
        """确保请求间隔符合限制"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_interval:
            time.sleep(self.request_interval - time_since_last_request)
        self.last_request_time = time.time()
    
    def analyze_text(self, text):
        """使用ChatGPT分析文本意图"""
        if not text or str(text).strip() == '':
            return '其他'
            
        prompt = f"""请仔细分析以下文本属于哪种意图类型。请只返回最匹配的一个意图类型，不要解释。

文本：{text}

可选的意图类型及其描述：
{chr(10).join([f'- {k}：{v}' for k, v in self.intent_labels.items()])}

请只返回上述意图类型中的一个具体类型名称（不要返回描述），如果都不符合就返回"其他"。"""

        data = {
            "model": "gpt-3.5-turbo",  # 使用 GPT-3.5-turbo 模型
            "messages": [
                {"role": "system", "content": "你是一个专业的文本意图分析助手，请准确分析用户文本的意图类型。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        for attempt in range(self.retry_count):
            try:
                self._wait_for_rate_limit()
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json=data,
                    timeout=self.timeout  # 使用更长的超时时间
                )
                response.raise_for_status()
                
                result = response.json()
                intent = result['choices'][0]['message']['content'].strip()
                
                if intent in self.intent_labels:
                    return intent
                return '其他'
                
            except requests.exceptions.RequestException as e:
                print(f"请求错误 (尝试 {attempt + 1}/{self.retry_count}): {str(e)}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # 使用更长的重试延迟
                continue
            except Exception as e:
                print(f"其他错误: {str(e)}")
                return '其他'
                
        print(f"分析文本失败，已达到最大重试次数: {text[:50]}...")
        return '其他'
    
    def analyze_dataset(self, df):
        """分析整个数据集的意图"""
        tqdm.pandas(desc="分析意图")
        
        # 添加错误处理和进度显示
        try:
            df['intent'] = df['cleaned_text'].progress_apply(self.analyze_text)
            
            # 保存带意图标注的数据
            output_file = os.path.join(self.processed_dir, 'labeled_data.csv')
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 统计各意图的数量
            intent_stats = df['intent'].value_counts().to_dict()
            
            # 保存意图统计信息
            stats_file = os.path.join(self.processed_dir, 'intent_stats.json')
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(intent_stats, f, ensure_ascii=False, indent=4)
            
            return df
            
        except Exception as e:
            print(f"处理数据集时发生错误: {str(e)}")
            raise 