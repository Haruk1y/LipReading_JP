import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import json
from datetime import datetime
import argparse
import Levenshtein
from tqdm import tqdm

from src.dataset import ROHANDataset, collate_fn
from src.model import LipReadingModel
from src.trainer import DiversityLoss

def load_model(model, checkpoint_path, device):
    """モデルの重みを読み込む"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with CER: {checkpoint['cer']:.2f}%")
    return model

def calculate_cer(pred_phonemes, target_phonemes):
    """CERを計算"""
    # 音素を単一の文字として結合
    unique_chars = {phoneme: chr(0xE000 + i) for i, phoneme in enumerate(set(pred_phonemes + target_phonemes))}
    pred_str = ''.join(unique_chars[p] for p in pred_phonemes)
    target_str = ''.join(unique_chars[p] for p in target_phonemes)
    
    # Levenshtein距離の計算
    distance = Levenshtein.distance(pred_str, target_str)
    
    # CERの計算（編集距離を文字数で割る）
    cer = (distance / len(target_phonemes)) * 100 if target_phonemes else 0
    return cer

def test_model(checkpoint_path, data_root, output_dir, device='cuda'):
    """モデルのテストを実行"""
    # 出力ディレクトリの作成
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # タイムスタンプ付きの結果ファイル名を生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"test_results_{timestamp}.txt"
    detailed_results_file = output_dir / f"detailed_results_{timestamp}.json"
    
    try:
        # テストデータセットの準備
        print("Loading test dataset...")
        test_dataset = ROHANDataset(
            data_root=data_root,
            split='test'
        )
        print(f"Test dataset loaded with {len(test_dataset)} samples")
        
        # DataLoaderの準備
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # モデルの準備
        print("Initializing model...")
        model = LipReadingModel(
            vocab_size=len(test_dataset.label_processor.all_phonemes)
        )
        model = load_model(model, checkpoint_path, device)
        model = model.to(device)
        model.eval()
        
        # Loss functionの準備
        criterion = DiversityLoss(
            num_classes=len(test_dataset.label_processor.all_phonemes),
            smoothing=0.1,
            diversity_weight=0.1,
            temperature=1.0
        ).to(device)
        
        # テスト結果の記録用
        all_results = []
        total_cer = 0
        total_samples = 0
        
        # ファイルヘッダーの書き込み
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"Test Results - {timestamp}\n")
            f.write(f"Model checkpoint: {checkpoint_path}\n")
            f.write("="*50 + "\n\n")
        
        # テストの実行
        print("Starting evaluation...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                try:
                    frames = batch['frames'].to(device)
                    phoneme_sequence = batch['phoneme_sequence'].to(device)
                    video_name = test_dataset.video_files[i]
                    
                    # Generate target mask for decoder
                    tgt_mask = model.generate_square_subsequent_mask(phoneme_sequence.size(1)).to(device)
                    
                    # Forward pass
                    output = model(frames, phoneme_sequence[:, :-1], tgt_mask)
                    
                    # Get predictions
                    predictions = output.argmax(dim=-1)
                    
                    # Convert predictions and targets to phonemes
                    pred_phonemes = [test_dataset.label_processor.idx2phoneme[idx.item()] 
                                   for idx in predictions[0] if idx.item() != -1]
                    target_phonemes = [test_dataset.label_processor.idx2phoneme[idx.item()]
                                     for idx in phoneme_sequence[0] if idx.item() != -1]
                    
                    # Calculate CER
                    cer = calculate_cer(pred_phonemes, target_phonemes)
                    
                    # 結果の記録
                    result = {
                        'video_name': video_name,
                        'target': target_phonemes,
                        'prediction': pred_phonemes,
                        'cer': cer
                    }
                    all_results.append(result)
                    
                    total_cer += cer
                    total_samples += 1
                    
                    # 結果をファイルに書き込み
                    with open(results_file, 'a', encoding='utf-8') as f:
                        f.write(f"Video: {video_name}\n")
                        f.write(f"Target    : {' '.join(target_phonemes)}\n")
                        f.write(f"Prediction: {' '.join(pred_phonemes)}\n")
                        f.write(f"CER: {cer:.2f}%\n")
                        f.write("-"*50 + "\n")
                    
                except Exception as e:
                    print(f"Error processing sample {i} ({video_name if 'video_name' in locals() else 'unknown'}): {str(e)}")
                    continue
        
        # 平均CERの計算と記録
        avg_cer = total_cer / total_samples if total_samples > 0 else float('inf')

        # CERが低い順（昇順）にソート
        best_results = sorted(all_results, key=lambda x: x['cer'])[:3]
        
        # 最終結果の書き込みの部分を修正
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*50 + "\n")
            f.write(f"Final Results:\n")
            f.write(f"Total samples processed: {total_samples}\n")
            f.write(f"Average CER: {avg_cer:.2f}%\n\n")
            
            # 最も低いCER Top3の結果を追加
            f.write("Top 3 Best CER Results:\n")
            f.write("-"*30 + "\n")
            for i, result in enumerate(best_results, 1):
                f.write(f"#{i} CER: {result['cer']:.2f}%\n")
                f.write(f"Video: {result['video_name']}\n")
                f.write(f"Target    : {' '.join(result['target'])}\n")
                f.write(f"Prediction: {' '.join(result['prediction'])}\n")
                f.write("-"*30 + "\n")
        
        # 詳細な結果をJSONファイルに保存
        with open(detailed_results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'checkpoint_path': str(checkpoint_path),
                'total_samples': total_samples,
                'average_cer': avg_cer,
                'results': all_results
            }, f, indent=4, ensure_ascii=False)
        
        # 最後の出力メッセージにも追加
        print(f"\nTesting completed!")
        print(f"Average CER: {avg_cer:.2f}%")
        print(f"\nTop 3 Best CER Results:")
        for i, result in enumerate(best_results, 1):
            print(f"#{i} Video: {result['video_name']}, CER: {result['cer']:.2f}%")
        print(f"\nResults saved to: {results_file}")
        print(f"Detailed results saved to: {detailed_results_file}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Test Lip Reading Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='data',
                      help='Path to data root directory')
    parser.add_argument('--output_dir', type=str, default='test_results',
                      help='Directory to save test results')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    test_model(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=args.device
    )

if __name__ == '__main__':
    main()