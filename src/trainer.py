import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import Levenshtein
import numpy as np
from tqdm import tqdm
from src.dataset import collate_fn  # collate_fnをインポート
from src.utils import TrainingLogger
import os

class LipReadingTrainer:
    def __init__(self, model, device, num_phonemes, label_processor, learning_rate=0.00005):
        """
        Args:
            model: 学習するモデル
            device: 使用するデバイス（'cuda' or 'cpu'）
            num_phonemes: 音素の総数
            label_processor: データセットのラベルプロセッサー
            learning_rate: 学習率
        """
        self.model = model.to(device)
        self.device = device

        self.criterion = LabelSmoothingLoss(num_phonemes, smoothing=0.1)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.98)
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=30,
            steps_per_epoch=1000,
            pct_start=0.1
        )
        self.num_phonemes = num_phonemes
        self.label_processor = label_processor

        # ラベル変換用の辞書を保存
        self.idx2phoneme = label_processor.idx2phoneme
        self.phoneme2idx = label_processor.phoneme2idx
        
    def train_epoch(self, train_loader):
        """1エポックの学習を行う"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            try:
                frames = batch['frames'].to(self.device)
                frame_labels = batch['frame_labels'].to(self.device)
                phoneme_sequence = batch['phoneme_sequence'].to(self.device)
                frames_lengths = batch['frames_lengths'].to(self.device)
                phonemes_lengths = batch['phonemes_lengths'].to(self.device)

                # Create target mask for decoder
                tgt_mask = self.model.generate_square_subsequent_mask(
                    phoneme_sequence.size(1)
                ).to(self.device)
                
                self.optimizer.zero_grad()

                # Forward pass
                logits = self.model(
                    frames,
                    phoneme_sequence[:, :-1],  # 入力系列は最後の要素を除く
                    tgt_mask
                )

                # Compute loss only on valid positions
                loss = self.criterion(
                    logits.view(-1, self.num_phonemes),
                    phoneme_sequence[:, 1:].contiguous().view(-1)  # ターゲットは最初の要素を除く
                )

                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # バッチごとの損失を出力
                if batch_idx % 10 == 0:
                    print(f"\nBatch {batch_idx}, Loss: {loss.item():.4f}")
                    print(f"Frames shape: {frames.shape}")
                    print(f"Phoneme sequence shape: {phoneme_sequence.shape}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                print(f"Frames shape: {frames.shape if 'frames' in locals() else 'N/A'}")
                print(f"Phoneme sequence shape: {phoneme_sequence.shape if 'phoneme_sequence' in locals() else 'N/A'}")
                continue
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    @torch.no_grad()
    def evaluate(self, val_loader, logger):
        """評価を行う"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        # 予測の分布を記録
        prediction_distribution = {}
        
        for batch in tqdm(val_loader):
            frames = batch['frames'].to(self.device)
            phoneme_sequence = batch['phoneme_sequence'].to(self.device)
            
            # 予測
            tgt_mask = self.model.generate_square_subsequent_mask(phoneme_sequence.size(1)).to(self.device)
            logits = self.model(frames, phoneme_sequence[:, :-1], tgt_mask)

            # Loss calculation
            loss = self.criterion(
                logits.view(-1, self.num_phonemes),
                phoneme_sequence[:, 1:].contiguous().view(-1)
            )
            
            # Get predictions
            predictions = logits.argmax(dim=-1)

            # 予測の分布を記録
            for pred in predictions.cpu().numpy().flatten():
                if pred != -1:  # パディングを除外
                    phoneme = self.idx2phoneme.get(pred, 'UNK')
                    prediction_distribution[phoneme] = prediction_distribution.get(phoneme, 0) + 1

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(phoneme_sequence[:, 1:].cpu().numpy())
            
            total_loss += loss.item()


        # 予測分布の出力
        logger.log_message("\nPrediction Distribution:")
        total_predictions = sum(prediction_distribution.values())
        for phoneme, count in sorted(prediction_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_predictions) * 100
            logger.log_message(f"{phoneme}: {percentage:.2f}% ({count} times)")

        # エラー率の計算
        cer = self.calculate_error_rates(all_predictions, all_targets, logger)
        avg_loss = total_loss / len(val_loader)
        
        logger.log_message(f"\nValidation Results:")
        logger.log_message(f"Average Loss: {avg_loss:.4f}")
        logger.log_message(f"CER: {cer:.2f}%")
        
        return avg_loss, cer

    def inference(self, frames, max_len=200):
        """シーケンスの推論を行う"""
        batch_size = frames.size(0)
        
        # Start tokens (silB)
        decoder_input = torch.ones(batch_size, 1).long().to(self.device)
        
        for _ in range(max_len - 1):
            # Create mask
            tgt_mask = self.model.generate_square_subsequent_mask(decoder_input.size(1)).to(self.device)
            
            # Predict next token
            output = self.model(frames, decoder_input, tgt_mask)
            next_token = output[:, -1].argmax(dim=-1, keepdim=True)
            
            # Append prediction
            decoder_input = torch.cat([decoder_input, next_token], dim=-1)
            
            # Check if all sequences have generated end token
            if (next_token == self.model.decoder.embedding.num_embeddings-1).all():
                break
                
        return decoder_input

    def calculate_error_rates(self, predictions, targets, logger):
        """
        CERを計算
        
        Args:
            predictions: 予測された音素のインデックスリスト
            targets: 正解の音素のインデックスリスト
            logger: ログ出力用のLoggerインスタンス
        
        Returns:
            tuple: (CER)
        """

        total_cer = 0
        total_chars = 0
        
        # サンプルの予測結果を表示用に保存
        sample_results = []
        
        for pred_seq, target_seq in zip(predictions, targets):
            # パディングの除去
            pred = [idx for idx in pred_seq if idx != -1]
            target = [idx for idx in target_seq if idx != -1]
            
            # 音素列に変換
            pred_phonemes = [self.idx2phoneme[idx] for idx in pred]
            target_phonemes = [self.idx2phoneme[idx] for idx in target]

            # 音素を単一の文字として結合
            # 各音素を一意の単一文字に置き換えて、レーベンシュタイン距離を計算
            unique_chars = {phoneme: chr(0xE000 + i) for i, phoneme in enumerate(set(pred_phonemes + target_phonemes))}
            
            pred_str = ''.join(unique_chars[p] for p in pred_phonemes)
            target_str = ''.join(unique_chars[p] for p in target_phonemes)
            
            # Levenshtein距離の計算
            distance = Levenshtein.distance(pred_str, target_str)
            
            # CERの計算（編集距離を文字数で割る）
            total_chars += len(target_phonemes)
            total_cer += distance
            
           # デバッグ情報の保存
            if len(sample_results) < 5:  # 最初の5つのサンプルのみ保存
                # 予測の分布を確認
                pred_distribution = {}
                for p in pred_phonemes:
                    pred_distribution[p] = pred_distribution.get(p, 0) + 1
                
                sample_results.append({
                    'target': target_phonemes,
                    'prediction': pred_phonemes,
                    'cer': (distance / len(target_phonemes)) * 100 if target_phonemes else 0,
                    'prediction_distribution': pred_distribution
                })
        
        # サンプル結果の出力
        logger.log_message("\nSample Predictions:")
        for i, result in enumerate(sample_results):
            logger.log_message(f"\nSample {i+1}:")
            logger.log_message(f"Target    : {' '.join(result['target'])}")
            logger.log_message(f"Prediction: {' '.join(result['prediction'])}")
            logger.log_message(f"CER: {result['cer']:.2f}%")
            logger.log_message(f"Prediction distribution: {result['prediction_distribution']}")
        
        # 全体のCERを計算
        cer = (total_cer / total_chars * 100) if total_chars > 0 else float('inf')
        
        # 予測の多様性をチェック
        if all(len(result['prediction_distribution']) == 1 for result in sample_results):
            logger.log_message("\nWARNING: Model is predicting only one type of phoneme!")
        
        return cer

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def train_model(model, train_dataset, val_dataset, num_epochs=30, batch_size=4, device='cuda'):
    """モデルの学習を実行"""
    # Logger の初期化
    logger = TrainingLogger()
    logger.log_message("Starting training...")

    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=1,  # worker数を2に減らす
        collate_fn=collate_fn,  # collate_fnを追加
        pin_memory=True,  # GPUを使用する場合のパフォーマンス向上
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=1,  # worker数を2に減らす
        collate_fn=collate_fn,  # collate_fnを追加
        pin_memory=True,  # GPUを使用する場合のパフォーマンス向上
    )
    
    # Initialize trainer
    trainer = LipReadingTrainer(
        model=model,
        device=device,
        num_phonemes=len(train_dataset.label_processor.all_phonemes),
        label_processor=train_dataset.label_processor,
        learning_rate=0.00005  # 学習率を小さめに設定
    )
    
    best_cer = float('inf')
    patience = 5
    no_improve = 0
    
    try:
        for epoch in range(num_epochs):
            logger.log_message(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 訓練
            logger.log_message("Training phase started...")
            train_loss = trainer.train_epoch(train_loader)
            
            # 評価
            logger.log_message("Validation phase started...")
            val_loss, cer = trainer.evaluate(val_loader, logger)
            
            # モデルの保存と早期終了の判定
            if cer < best_cer:
                improvement = best_cer - cer
                best_cer = cer
                no_improve = 0
                
                # モデルの保存
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'cer': cer,
                }, f'logs/train2/model_epoch_{epoch+1}_cer_{cer:.2f}.pth')
                
                logger.log_message(f"New best model saved! CER improved by {improvement:.2f}%")
            else:
                no_improve += 1
                # if no_improve >= patience:
                #     logger.log_message("Early stopping triggered")
                #     break
            
            # エポックの要約
            logger.log_message(f"\nEpoch Summary:")
            logger.log_message(f"Train Loss: {train_loss:.4f}")
            logger.log_message(f"Val Loss: {val_loss:.4f}")
            logger.log_message(f"CER: {cer:.2f}%")
            logger.log_message(f"Best CER: {best_cer:.2f}%")
    
    except KeyboardInterrupt:
        logger.log_message("\nTraining interrupted by user")
    except Exception as e:
        logger.log_message(f"\nError during training: {str(e)}")
        raise
    finally:
        logger.log_message("\nTraining completed")