# 使用方法
## 各ディレクトリ・ファイルについて
- data：ROHAN4600データセットの動画・口ランドマーク・音素ラベルを格納する（ROHANについてはhttps://zunko.jp/multimodal_dev/twdashbord.php 参照）
  - preprocess：データセットを学習用・バリデーション用・テスト用に分割する
  - processed：分割後のデータセット。txtファイルに各操作に用いられるデータ名が格納されている
- src  各種トレーニング・テスト用のスクリプト
- train.py, test.py メインの学習・テスト用スクリプト

## 学習・テスト用データの作り方
1. data以下のprocessedにtrain/val/testフォルダを配置する。
2. 各フォルダ内にlab/landmarks/videosフォルダを配置する
3. *_list.txtに整合するようにデータセットからデータを配置していく
(現在は例として一部のデータのみ格納しているが，*_list.txtに記述されている全てのデータを入れるようにする)

## 実行環境
実行環境はistクラスタ内で正常に動作するように，poetryを用いて構築した。(pyproject.tomlを参照)
また，pyenv等を用いてpythonバージョン3.10.15を用いて実験を行った。

## 学習方法
1. data/processed/trainフォルダ内にtrain_list.txt通りのデータを格納する
2. data/processed/valフォルダ内にval_list.txt通りのデータを格納する
3. train.pyを必要なプロパティを指定して実行する。無効なデータがある場合，自動的に学習データから省くようにしている。

## テスト方法
1. data/processed/testフォルダ内にtest_list.txt通りのデータを格納する
2. 自由課題提出フォルダのGoogleドライブから学習済み重みをダウンロード https://drive.google.com/file/d/1h4dQcsue5cU8iE3GE5ZfxGcmfKC-6fzW/view?usp=share_link
3. ダウンロードした重みを指定して，test.pyを実行する。テスト結果がログとしてファイルに出力される。

## 実装
学習に使用したモデルは，[[T.Arakane+, 2022]](https://link.springer.com/chapter/10.1007/978-3-031-25825-1_34#Tab1)内で提案されているConformerをもとにして実装を行った。
CERを小さくするために，予測子音数を多くするようなペナルティ項や，学習率の調整などをおこなっている。