# ExecuTorch

## 概要
本ディレクトリは、フェリックス機械学習倶楽部の以下のブログ記事のソースコードを管理しています。
- [ExecuTorchを用いたQuantization Aware Training実装](https://ai.ferix.jp/blog?id=ko5bd_adugq)
- [ExecuTorchランタイムを使用した推論実行ファイルの作成](https://ai.ferix.jp/blog?id=b1onjucwhjw)


ExecuTorchはモバイルおよびエッジデバイス上での推論機能を実現するためのエンドツーエンドソリューションであり、  
PyTorchモデルのエッジデバイスへの効率的なデプロイを可能にします。  
ExecuTorchの開発元であるMetaでは，既に自社製品への統合を進めており，例えばMeta Quest 3のハンドトラッキング機能で利用しているそうです．  
本ディレクトリで紹介する2つの記事では，ExecuTorchを用いたQuantization Aware Trainingと推論実行ファイルの作成を実装例を用いて解説します．  
記事は前後半に分かれており，  
前半の[ExecuTorchを用いたQuantization Aware Training実装](https://ai.ferix.jp/blog?id=ko5bd_adugq)では、  
ResNet18にMNISTの画像分類タスクを学習させます。  
PyTorchモデルから始め、ATenグラフ形式へのエクスポート、QAT、.pteファイルへの変換の一連の処理を実装します。  
後半の[ExecuTorchランタイムを使用した推論実行ファイルの作成](https://ai.ferix.jp/blog?id=b1onjucwhjw)では、  
この.pteファイルをロードして推論を行う実行ファイルを作成します。  

## ディレクトリ構成
```
|- qat.ipynb               # Quantization Aware Training実装
|- utils.py                # utils
|- models
 |- resnet18_qat_ep10.pte  # 学習済みモデル
|- build_src
 |- main.cpp
 |- CMakeLists.txt
```
