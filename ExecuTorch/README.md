# ExecuTorch

本ディレクトリは、フェリックス機械学習倶楽部の以下のブログ記事のソースコードを管理しています。
- [ExecuTorchを用いたQuantization Aware Training実装](https://ai.ferix.jp/blog?id=ko5bd_adugq)

## 概要
本記事では、ExecuTorchを用いたQuantization Aware Training（QAT）の実装方法について解説しています。
ExecuTorchは、モバイルやエッジデバイス向けの推論機能を提供するPyTorchエコシステムの一部であり、MetaやAppleなどの企業がサポートしています。
手順は、PyTorchモデルをATenグラフ形式でエクスポートし、XNNPACKQuantizerを使って量子化処理を組み込み、学習を行います。
最終的に、学習済みモデルをExecuTorch Program形式にコンパイルし、効率的な推論を可能にします。

## ディレクトリ構成
```
|- qat.ipynb               # Quantization Aware Training実装
|- utils.py                # utils
|- models
 |- resnet18_qat_ep10.pte  # 学習済みモデル
```
