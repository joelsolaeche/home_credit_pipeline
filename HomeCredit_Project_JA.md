# 🏦 Home Credit Default Risk - ML パイプラインプロジェクト

> **ローン返済不能リスクを予測する機械学習パイプライン（AUC-ROC 75.5%）**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.0+-green.svg)](https://lightgbm.readthedocs.io)

## 📋 プロジェクト概要

このプロジェクトは、**ローン返済不能（デフォルト）リスクを予測するためのエンドツーエンドの機械学習パイプライン**を実装しています。

- 🎯 **AUC-ROC 75.5%**（LightGBM 最適化モデル）
- 🚀 **本番運用を意識した sklearn パイプライン**
- 📊 **ビジネス視点を含む詳細な EDA**
- 🔧 **モジュール設計・テスト・ドキュメント完備**
- ⚡ **複数モデル比較**（LR → RF → LightGBM）

## 🚀 クイックスタート

### インストール
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### プロジェクト実行
```bash
jupyter notebook "AnyoneAI - Sprint Project 02.ipynb"
python run_dashboard.py
pytest tests/
```

## 📁 プロジェクト構成

```
📁 fintech_pipeline_ml/
├── 📓 AnyoneAI - Sprint Project 02.ipynb
├── 🌐 streamlit_app.py
├── 🚀 run_dashboard.py
├── 📄 README_Portfolio.md
├── 📁 src/
│   ├── data_utils.py
│   ├── preprocessing.py
│   └── config.py
├── 📁 dataset/
├── 📁 tests/
├── requirements.txt
└── streamlit_requirements.txt
```

## 🎯 モデルの主要結果

| モデル | 検証 AUC | 学習時間 | 特徴 |
|--------|---------|----------|-------|
| ロジスティック回帰 | 0.6769 | 約1.3秒 | ベースライン |
| ランダムフォレスト | 0.7379 | 約14分 | 最適化モデル |
| **🏆 LightGBM（最適化）** | **0.7552** | 約8分 | **最高性能** |

## 💼 ビジネスインパクト

- AUC-ROC 75.5%（ベースライン比 +25%）
- デフォルト顧客の 80% を検知
- 約 $11.9M の純利益改善効果
- 24万6千件以上のサンプル分析

## 🛠️ 技術的実装

### コア機能
- sklearn 前処理パイプライン
- RandomizedSearchCV によるハイパーパラメータ最適化
- クロスバリデーション
- 特徴量エンジニアリング
- 本番運用向けモジュール構造

### 使用技術
Python / Pandas / NumPy / Scikit-learn / LightGBM / Matplotlib / Seaborn / Streamlit / Plotly / Jupyter

## 🎯 インタラクティブダッシュボード

起動：
```
python run_dashboard.py
```

機能：
- モデル比較・主要メトリクス
- 相関ヒートマップ・データ分析
- ROC 曲線・特徴量重要度
- リスク分析と財務インパクト
- 顧客プロフィールに基づくリアルタイム予測

## 🧪 テスト & QA

```bash
pytest tests/
isort --profile=black . && black --line-length 88 .
flake8 src/
```

## 📊 ワークフロー

1. EDA
2. 特徴量エンジニアリング
3. モデル開発
4. ハイパーパラメータ最適化
5. パイプライン構築
6. 検証 & テスト

## 🎯 学習成果

- ML パイプライン構築スキル
- Streamlit ダッシュボード開発
- リスクモデリング知識
- 本番レベルのコード品質
- ビジネスインパクト分析
- モデル最適化・文書化スキル
