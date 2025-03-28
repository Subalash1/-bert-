import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
import numpy as np
import os
import json
import pandas as pd
from torch.utils.data import WeightedRandomSampler
import random
import jieba

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def augment_text(text, n_samples=2):
    """对文本进行数据增强"""
    augmented_texts = []
    words = list(jieba.cut(text))
    
    for _ in range(n_samples):
        aug_text = words.copy()
        
        # 1. 随机删除词（保持核心语义）
        if len(aug_text) > 5:
            n_to_delete = max(1, len(aug_text) // 10)  # 删除10%的词
            positions = random.sample(range(len(aug_text)), n_to_delete)
            aug_text = [w for i, w in enumerate(aug_text) if i not in positions]
        
        # 2. 随机重复词（强调）
        if len(aug_text) > 3:
            pos = random.randint(0, len(aug_text)-1)
            aug_text.insert(pos, aug_text[pos])
        
        # 转回文本
        augmented_text = ''.join(aug_text)
        if augmented_text != text and len(augmented_text) > 2:
            augmented_texts.append(augmented_text)
    
    # 如果没有生成足够的增强文本，用原文补充
    while len(augmented_texts) < n_samples:
        augmented_texts.append(text)
    
    return augmented_texts

def balance_dataset(df, min_samples=80):
    """平衡数据集，对样本量少的类别进行数据增强"""
    class_counts = df['intent'].value_counts()
    augmented_data = []
    
    print("原始类别分布：")
    print(class_counts)
    
    for intent, count in class_counts.items():
        if count < min_samples:
            print(f"\n增强类别 '{intent}' (原始样本数: {count})")
            # 需要增强的样本数
            n_aug = min_samples - count
            # 获取该类别的所有样本
            intent_samples = df[df['intent'] == intent]
            
            # 对每个样本进行增强
            for _, row in intent_samples.iterrows():
                n_samples = max(1, n_aug // count)  # 每个样本需要增强的次数
                aug_texts = augment_text(row['cleaned_text'], n_samples)
                
                for aug_text in aug_texts:
                    augmented_data.append({
                        'cleaned_text': aug_text,
                        'intent': intent
                    })
    
    # 将增强的数据添加到原始数据集
    if augmented_data:
        augmented_df = pd.DataFrame(augmented_data)
        df = pd.concat([df, augmented_df], ignore_index=True)
        
        print("\n增强后的类别分布：")
        print(df['intent'].value_counts())
    
    return df

def train_intent_classifier(df, model_save_dir='models'):
    """训练模型，使用简单的训练/验证集划分"""
    # 数据增强和平衡
    df = balance_dataset(df)
    
    # 确保模型保存目录存在
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 准备标签映射
    unique_labels = sorted(df['intent'].unique())
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    
    # 保存标签映射
    with open(os.path.join(model_save_dir, 'label_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump({'label_to_id': label_to_id, 'id_to_label': id_to_label}, f, ensure_ascii=False, indent=2)
    
    # 准备数据
    texts = df['cleaned_text'].values
    labels = df['intent'].map(label_to_id).values
    
    # 打印数据集信息
    print(f"数据集大小: {len(texts)}")
    print(f"标签数量: {len(labels)}")
    print(f"唯一标签: {unique_labels}")
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, 
        test_size=0.2, 
        stratify=labels,
        random_state=42
    )
    
    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
    
    # 加载tokenizer和模型
    model_path = r"C:\Users\Li\Desktop\work\models--bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(unique_labels)
    )
    
    # 创建数据集
    train_dataset = IntentDataset(X_train, y_train, tokenizer)
    val_dataset = IntentDataset(X_val, y_val, tokenizer)
    
    # 计算类别权重
    class_counts = np.bincount(y_train, minlength=len(unique_labels))
    total_samples = len(y_train)
    class_weights = torch.FloatTensor(total_samples / (len(unique_labels) * class_counts))
    
    # 创建带权重的采样器
    samples_weights = [class_weights[label] for label in y_train]
    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(y_train),
        replacement=True
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 训练设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    class_weights = class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-5,
        steps_per_epoch=len(train_loader),
        epochs=15
    )
    
    # 早停设置
    best_val_loss = float('inf')
    patience = 4
    patience_counter = 0
    best_model_state = None
    
    # 训练循环
    for epoch in range(15):
        # 训练阶段
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}:')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
    
    # 加载最佳模型状态
    model.load_state_dict(best_model_state)
    
    # 保存最佳模型
    model.save_pretrained(os.path.join(model_save_dir, 'best_model'))
    tokenizer.save_pretrained(os.path.join(model_save_dir, 'best_model'))
    
    return model, tokenizer, X_val, y_val, id_to_label

def evaluate_model(model, X_test, y_test, tokenizer, id_to_label, device):
    model.to(device)
    
    # 准备测试数据
    test_dataset = IntentDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 评估模型
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 转换预测结果为原始标签
    pred_labels = [id_to_label[pred] for pred in all_preds]
    true_labels = [id_to_label[label] for label in all_labels]
    
    # 生成评估报告
    report = classification_report(true_labels, pred_labels)
    
    return report

def predict_intent(text, model, tokenizer, id_to_label, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 准备输入数据
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1)
        
    return id_to_label[pred.item()]