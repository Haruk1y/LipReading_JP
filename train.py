import torch
from pathlib import Path
from src.dataset import ROHANDataset
from src.model import LipReadingModel
from src.trainer import train_model
from src.utils import set_seed, load_config

def main():
    # 設定
    config = {
        'data_root': 'data',
        'batch_size': 1,
        'num_epochs': 30,
        'learning_rate': 0.00005,
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'accumulation_steps': 4  # Gradient accumulation steps
    }

    # シード設定
    set_seed(config['seed'])

    print(f"Initializing datasets with data_root: {config['data_root']}")
    print(f"Checking directory structure:")
    print(f"Data root exists: {Path(config['data_root']).exists()}")
    print(f"Processed directory exists: {(Path(config['data_root']) / 'processed').exists()}")
    print(f"Raw directory exists: {(Path(config['data_root']) / 'raw').exists()}")

    try:
        train_dataset = ROHANDataset(
            data_root=config['data_root'],
            split='train'
        )
        print(f"Successfully loaded training dataset with {len(train_dataset)} samples")
        if hasattr(train_dataset, 'invalid_data'):
            print(f"Removed {len(train_dataset.invalid_data)} invalid samples")
        
        val_dataset = ROHANDataset(
            data_root=config['data_root'],
            split='val'
        )
        print(f"Successfully loaded validation dataset with {len(val_dataset)} samples")
    
    except Exception as e:
        print(f"Error initializing datasets: {e}")
        raise

    # モデルの準備
    model = LipReadingModel(
        vocab_size=len(train_dataset.label_processor.all_phonemes)
    )

    # 学習の実行
    train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        device=config['device']
    )

if __name__ == '__main__':
    main()