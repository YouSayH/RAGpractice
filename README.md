# ローカルGPUを活用したRAGの練習(メモ)

このリポジトリは、独自のドキュメントを知識源として質問応答を行うRAG（Retrieval-Augmented Generation）システムを、ローカル環境でゼロから構築する過程をまとめたものです。Anacondaを使った環境構築から、システムのチューニング、発生した問題とその解決策までを記録しています。

## 🚀 主な特徴

  * **多様なファイル形式に対応**: `txt`, `md`, `pdf`, `docx`, `xlsx`, `pptx`など、複数のファイル形式を知識源として自動で読み込みます。
  * **GPU活用**: ローカルのNVIDIA GPUを活用し、`sentence-transformers`による高速なベクトル化を実現します。
  * **柔軟なLLM選択**: 回答生成部分を、**①Gemini API**, **②Hugging Faceモデルの直接実行**, **③LM Studio経由**の3つの方法から簡単に切り替え可能です。
  * **設定の外部化**: `config.yaml`の`provider`を一行書き換えるだけで、モデルの実行方法をコードを触らずに変更できます。
  * **高度なRAGチューニング**: `クエリ変換`や`構造化チャンキング`など、検索精度を向上させるための実践的なテクニックを導入しています。

-----

## 🛠️ システム構成図

```
【データベース構築フェーズ】
[複数ファイル] -> [build_database.py] -> [ベクトル化] -> [ChromaDB (ローカルDB)]

【質疑応答フェーズ】
[ユーザーの質問] -> [query_rag.py] -> [質問をベクトル化] -> [ChromaDBから関連情報検索]
                                         ↓
                         [情報 + 質問] -> [LLM (API/ローカル/LM Studio)] -> [最終的な回答]
```

-----
## 🧠 RAG（Retrieval-Augmented Generation）の原理

Retrieval-Augmented Generation（RAG）は、大規模言語モデル（LLM）が持つ広範な一般知識と、外部の特定の知識ソースを動的に統合するためのフレームワークです。これにより、LLMが内部知識のみに依存することで生じるハルシネーション（事実に基づかない情報の生成）を抑制し、より正確で信頼性の高い回答を生成することが可能になります。

本システムにおけるRAGの処理フローは、以下の2つの主要なフェーズで構成されます。

1.  **検索フェーズ（Retrieval）**
    ユーザーからの質問（クエリ）を受け取ると、まずそのクエリをベクトル化します。次に、そのクエリベクトルを用いて、あらかじめドキュメント群から作成されたベクトルデータベース（ChromaDB）内を検索し、意味的に関連性が高いドキュメントチャンクを複数取得します。

2.  **生成フェーズ（Generation）**
    検索フェーズで取得したドキュメントチャンクを、ユーザーの元の質問と共にコンテキストとしてプロンプトに組み込みます。この拡張されたプロンプトをLLMに入力し、与えられたコンテキスト情報に基づいて最終的な回答を生成させます。このアプローチにより、回答は特定の情報源に根差したものとなり、その出典を明確にすることが可能となります。


-----

## 📖 使い方 (インストールから実行まで)

### 1\. 事前準備

  * **Anaconda**: Pythonの環境管理ツール。
  * **NVIDIA GPU**: CUDAが利用可能なGPU。
  * **Git**: ソースコードの管理ツール。
  * **(任意) LM Studio**: ローカルLLMを手軽に実行するためのデスクトップアプリ。

### 2\. 環境構築

**① Condaの初期設定 (初回のみ)**
AnacondaをPCにインストールした直後、ターミナルで`conda`コマンドを使えるようにするために、以下の初期設定コマンドを実行します。

```bash
conda init
```

実行後、**必ずターミナルを再起動**してください。

**② リポジトリのクローンと移動**

```bash
git clone https://github.com/YouSayH/RAGpractice.git
cd RAGpractice
```

**③ Conda仮想環境の作成と有効化**

```bash
conda create -n rag_env python=3.11 -y
conda activate rag_env
```

**④ 必要なライブラリのインストール**
`openai`ライブラリが追加されています。

```bash
# (PyTorchのGPU版を先にインストール)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
# (残りのライブラリ)
pip install sentence-transformers chromadb PyYAML langchain langchain-community pypdf python-docx openpyxl python-pptx tqdm python-dotenv google-generativeai transformers bitsandbytes openai
```

### 3\. 設定ファイルの準備

1.  **`config.yaml`の編集**:
      * `generation:`ブロックにある`provider:`で使用したい方法を選びます。（`gemini`, `huggingface`, `lmstudio`のいずれか）
      * それぞれの設定（モデル名など）を環境に合わせて編集します。
2.  **`.env`ファイルの作成**:
      * Gemini APIを利用する場合、このファイルを作成しAPIキーを設定します。
    <!-- end list -->
    ```
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```
3.  **`source_documents`フォルダ**:
      * 知識源としたいファイル（PDF, Wordなど）をこのフォルダに格納します。

### 4\. 実行

1.  **データベースの構築**: 最初に、`source_documents`の内容をベクトル化します。
    ```bash
    python build_database.py
    ```
2.  **質疑応答**: データベースを使って質問に答えます。
    ```bash
    # (LM Studioを使う場合は、事前にサーバーを起動しておきます)
    python query_rag.py
    ```

-----

## ⚙️ モデル選定について

### 埋め込みモデル (Embedding)

テキストをベクトル化する部分です。多言語で高性能な`intfloat/multilingual-e5-large`を基本としています。

### 生成モデル (Generation)

`config.yaml`の`provider`で以下の3つから選択します。

  * **`gemini` (Gemini API)**: ハードウェアの制約なく、手軽に高性能なモデルを試せます。
  * **`huggingface` (ローカルLLM)**: オフラインで動作し、データプライバシーの観点から安全です。`4-bit量子化`により、コンシューマ向けGPUでも動作可能です。
  * **`lmstudio` (LM Studio経由)**: GUIで手軽にモデルを管理しつつ、APIサーバーとして利用できます。様々なモデルを手軽に試したい場合に便利です。

-----

## 📚 主要ライブラリ

| ライブラリ | 役割 |
| :--- | :--- |
| **PyTorch** | 深層学習の基盤。GPU利用に必須。 |
| **sentence-transformers** | テキストをベクトル化（Embedding）する。 |
| **ChromaDB** | ベクトルを保存・検索するデータベース。 |
| **LangChain** | ドキュメントの読み込み・分割など、RAGの部品を便利に扱う。 |
| **transformers** | Hugging FaceのローカルLLMをロード・実行する。 |
| **google-generativeai** | Gemini APIを利用する。 |
| **openai** | LM StudioのAPIサーバーと通信する。 |
| **PyYAML** | `config.yaml`を読み込む設定管理。 |

-----

## 🤔 直面した問題点と改善の過程

  * **Hugging Faceの認証**: `Mistral`や`Gemma`などの高性能モデルは、利用前にサイトでの規約同意と`huggingface-cli login`による認証が必要でした。
  * **検索精度の課題**: 当初、「コードを教えて」という質問に対し、説明文が検索されてしまう問題が発生しました。これは`構造化チャンキング`や`クエリ変換`といった手法を導入することで大幅に改善しました。
  * **コードの進化**: 当初はLM Studio連携のみでしたが、`transformers`ライブラリで直接モデルを実行する方法を追加しました。現在は`config.yaml`で両方の方法を選択できるようにリファクタリングし、保守性と拡張性を高めています。

-----

## 💡 今後の改善点

  * **クラス化**: `DatabaseBuilder`や`RAGSystem`といったクラスに責務を分割し、よりオブジェクト指向な設計にする。
  * **Web UI**: `Streamlit`などを使い、対話型のWebアプリケーション化する。
  * **高度なRAG**:
      * **ハイブリッド検索**: 現在のベクトル検索に加え、キーワード検索を組み合わせることで検索精度を向上させる。
      * **再ランキング (Re-ranking)**: 検索で取得した情報の関連度を再度評価し、より精度の高い情報をLLMに渡す。
  * **データ品質の向上**:
      * **データクレンジング**: 知識源となるドキュメントから不要な情報（ノイズ）を除去し、表記揺れを統一する。
      * **メタデータ付与**: ドキュメントにタグやカテゴリを付与し、検索時にフィルタリングできるようにする。
  * **評価と改善サイクルの確立**:
      * **フィードバック学習**: 生成された回答の品質を評価し、その結果をモデルやプロンプトの改善に活かす仕組みを導入する。
      * **パイプラインの継続監視**: 回答精度を客観的な指標で測定し、継続的に評価する。
  * **カスタム埋め込み学習**: 専門分野のデータセットで埋め込みモデルをファインチューニングし、ドメイン特化の検索精度向上を目指す。
  * **説明可能なAI**: 生成された答えの根拠を説明させるようにする。