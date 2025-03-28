from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.intent_analyzer import IntentAnalyzer
import os

def main():
    # 创建必要的目录
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 1. 数据预处理
    cleaner = TextCleaner()
    df = cleaner.process_data('data/raw/语料.xlsx')
    print("数据预处理完成")
    
    # 2. 意图分析
    analyzer = IntentAnalyzer()
    labeled_df = analyzer.analyze_dataset(df)
    print("意图分析完成")
    
    # 保存处理后的数据集
    labeled_df.to_csv('data/processed/labeled_data.csv', index=False)
    print("已保存标注后的数据集到 data/processed/labeled_data.csv")

if __name__ == "__main__":
    main() 