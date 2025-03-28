import pandas as pd
import re
import os
import emoji

class TextCleaner:
    def __init__(self):
        self.processed_dir = 'data/processed'
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def clean_text(self, text):
        # 转换为字符串并去除首尾空白
        text = str(text).strip()
        
        # 去除表情符号
        text = emoji.replace_emoji(text, '')
        
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 去除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 替换多个空格为单个空格
        text = re.sub(r'\s+', ' ', text)
        
        # 去除纯符号和无意义的重复字符
        text = re.sub(r'([！？。，；：、~～!?.,;:\-_]+)(?:\1+)', r'\1', text)
        
       
        # 去除纯符号行
        if not re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', text):
            return ''
        
        return text.strip()
    
    def process_data(self, input_file):
        # 读取Excel数据
        df = pd.read_excel(input_file)
        
        # 打印列名，方便调试
        print("Excel文件的列名:", df.columns.tolist())
        
        # 获取第一个列作为文本列
        text_column = df.columns[0]
        
        # 在清洗前记录原始行数
        original_row_count = len(df)
        
        # 清洗文本
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # 删除清洗后为空的行
        df = df[df['cleaned_text'].str.len() > 0]
        
        # 保存预处理后的数据
        output_file = os.path.join(self.processed_dir, 'cleaned_data.csv')
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 打印清洗统计信息
        print(f"原始数据行数: {original_row_count}")
        print(f"清洗后数据行数: {len(df)}")
        print(f"清洗掉的行数: {original_row_count - len(df)}")
        
        return df 