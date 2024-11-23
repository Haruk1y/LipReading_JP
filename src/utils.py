import yaml
import torch
import random
import numpy as np
import json
from datetime import datetime
import os

def set_seed(seed):
    """再現性のためのシード設定"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_config(config_path):
    """設定ファイルの読み込み"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # ログファイルの準備
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
        self.metrics_file = os.path.join(log_dir, f"metrics_{timestamp}.json")
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_per': [],
            'best_val_per': float('inf')
        }
    
    def log_message(self, message):
        """メッセージをログファイルに書き込む"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
        print(message)
    
    def log_epoch(self, epoch, train_loss, val_loss, val_per):
        """エポックの結果を記録"""
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_per'].append(val_per)
        
        if val_per < self.metrics['best_val_per']:
            self.metrics['best_val_per'] = val_per
        
        message = (f"\nEpoch {epoch}:\n"
                  f"Train Loss: {train_loss:.4f}\n"
                  f"Val Loss: {val_loss:.4f}\n"
                  f"Val PER: {val_per:.2f}%\n"
                  f"Best Val PER: {self.metrics['best_val_per']:.2f}%")
        
        self.log_message(message)
        
        # メトリクスをJSONファイルに保存
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)