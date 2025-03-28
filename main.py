from src.models.train import train_intent_classifier, evaluate_model
import pandas as pd
import torch

def main():
    # 加载预处理后的数据，指定编码格式
    try:
        labeled_df = pd.read_csv('data/processed/labeled_data.csv', encoding='utf-8-sig')
    except UnicodeDecodeError:
        # 如果 UTF-8 失败，尝试使用 GBK 编码
        labeled_df = pd.read_csv('data/processed/labeled_data.csv', encoding='gbk')
    print("加载标注数据完成")
    
    # 1. 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, X_test, y_test, id_to_label = train_intent_classifier(labeled_df)
    print("模型训练完成")
    
    # 2. 评估模型
    evaluation_report = evaluate_model(model, X_test, y_test, tokenizer, id_to_label, device)
    print("\n模型评估报告：")
    print(evaluation_report)

if __name__ == "__main__":
    main()