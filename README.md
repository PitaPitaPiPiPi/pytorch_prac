# pytorch_prac

PyTorchでMNIST分類を行う学習用リポジトリです。  
5層MLPと自作SGDオプティマイザを用いて学習・評価を行います。

## Features

- MNISTデータセットの読み込みと前処理
- 5層MLPによる分類モデル
- 自作SGDオプティマイザ
- 学習ログの保存（`logs/`）

## Requirements

- Python 3.x
- torch
- torchvision
- tqdm

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --lr 0.05 --epochs 10 --batch_size 128
```

引数の例:
- `--lr`: 学習率（default: 0.05）
- `--epochs`: エポック数（default: 10）
- `--batch_size`: バッチサイズ（default: 128）

## Project Structure

```
.
├── main.py
├── models/
│   └── mlp.py
├── optimizer/
│   └── custom_sgd.py
├── utils/
│   ├── data_loader.py
│   └── logger.py
└── requirements.txt
```

## License

Not specified.
