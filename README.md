# ローカルGPUを活用したRAGの練習(メモ)

このリポジトリは、独自のドキュメントを知識源として質問応答を行うRAG（Retrieval-Augmented Generation）システムを、ローカル環境でゼロから構築する過程をまとめたものです。Anacondaを使った環境構築から、システムのチューニング、発生した問題とその解決策までを記録しています。

## 🚀 主な特徴

  - **多様なファイル形式に対応**: `txt`, `md`, `pdf`, `docx`, `xlsx`, `pptx`など、複数のファイル形式を知識源として自動で読み込み。
  - **GPU活用**: ローカルのNVIDIA GPUを活用し、`sentence-transformers`による高速なベクトル化を実現。
  - **柔軟なLLM選択**: 回答生成部分を、ローカルLLM（`Gemma`など）とAPI（`Gemini API`）で簡単に切り替え可能。
  - **設定の外部化**: `config.yaml`により、モデル名やチューニングパラメータをコードを触らずに変更可能。
  - **高度なRAGチューニング**: `クエリ変換`や`構造化チャンキング`など、検索精度を向上させるための実践的なテクニックを導入。

-----

## 🛠️ システム構成図

```
【データベース構築フェーズ】
[複数ファイル] -> [build_database.py] -> [ベクトル化] -> [ChromaDB (ローカルDB)]

【質疑応答フェーズ】
[ユーザーの質問] -> [query_rag.py] -> [質問をベクトル化] -> [ChromaDBから関連情報検索] -> [情報 + 質問] -> [LLM (ローカル or API)] -> [最終的な回答]
```

-----

## 🧠 RAGの仕組み

このシステムは、LLMが持つ汎用的な知識に、私たちが用意した専門知識を融合させる技術です。これは「**持ち込み可のテスト**」に例えられます。

1.  **検索 (Retrieval)**: ユーザーの質問に対し、まずLLMに聞く前に、手元の「参考書」（ベクトルDB）から関連性の高いページだけを探し出します。
2.  **生成 (Generation)**: LLMに「この参考ページの情報だけを元に、質問に答えてください」と具体的な指示を出します。これにより、LLMが不確かな情報で回答する（ハルシネーション）のを防ぎ、根拠に基づいた正確な回答を生成させます。

-----

## 📖 使い方 (インストールから実行まで)

### 1\. 事前準備

  - **Anaconda**: Pythonの環境管理ツール。
  - **NVIDIA GPU**: CUDAが利用可能なGPU。
  - **Git**: ソースコードの管理ツール。

### 2\. 環境構築

**① Condaの初期設定 (初回のみ)**
AnacondaをPCにインストールした直後、ターミナルで`conda`コマンドを使えるようにするために、以下の初期設定コマンドを実行します。

```bash
conda init
```

実行後、**必ずターミナルを再起動**してください。

**② リポジトリのクローンと移動**

```bash
git clone <リポジトリのURL>
cd <リポジトリ名>
```

**③ Conda仮想環境の作成と有効化**

```bash
conda create -n rag_env python=3.11 -y
conda activate rag_env
```

**④ 必要なライブラリのインストール**

```bash
# (PyTorchのGPU版を先にインストール)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
# (残りのライブラリ)
pip install sentence-transformers chromadb PyYAML langchain langchain-community pypdf python-docx openpyxl python-pptx tqdm python-dotenv google-generativeai transformers bitsandbytes
```

### 3\. 設定ファイルの準備

1.  **`config.yaml`**: モデル名やチャンクサイズなどをこのファイルで設定します。詳細はファイル内のコメントを確認してください。
2.  **`.env`**: APIを利用する場合、このファイルを作成しAPIキーを設定します。
    ```
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```
3.  **`source_documents`**: 知識源としたいファイル（PDF, Wordなど）をこのフォルダに格納します。

### 4\. 実行

1.  **データベースの構築**: 最初に、`source_documents`の内容をベクトル化します。
    ```bash
    python build_database.py
    ```
2.  **質疑応答**: データベースを使って質問に答えます。
    ```bash
    python query_rag.py
    ```

-----

## ⚙️ モデル選定について

### 埋め込みモデル (Embedding)

テキストをベクトル化する「脳」の部分です。多言語で汎用的に高性能な`intfloat/multilingual-e5-large`を基本としつつ、日本語特化モデルを試すことで、特定のタスクでの精度向上が期待できます。

### 生成モデル (Generation)

検索した情報を元に回答を作成する部分です。

  - **Gemini API**: ハードウェアの制約なく、手軽に高性能なモデルを試せます。
  - **ローカルLLM (Gemmaなど)**: オフラインで動作し、データプライバシーの観点から安全です。`4-bit量子化`により、コンシューマ向けGPUでも動作可能です。

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
| **PyYAML** | `config.yaml`を読み込む設定管理。 |

-----

## 🤔 直面した問題点と改善の過程

  - **Hugging Faceの認証**: `Mistral`や`Gemma`などの高性能モデルは、利用前にサイトでの規約同意と`huggingface-cli login`による認証が必要でした。これは、モデルの責任ある利用を促すための仕組みです。
  - **検索精度の課題**: 当初、`「コードを教えて」`という質問に対し、コードブロックではなく説明文が検索されてしまう問題が発生しました。
      - **試行1: チャンクサイズの調整**: `chunk_size`を大きくしたが、根本的な解決には至りませんでした。
      - **試行2: 構造化チャンキング**: `MarkdownHeaderTextSplitter`を導入し、見出し単位で分割することで精度が向上しました。
      - **試行3: クエリ変換**: ユーザーの質問をLLMで検索に適した形に変換する手法を導入し、目的の情報を引き当てる精度が大幅に向上しました。
      - **試行4: 複数コンテキストの利用**: 検索結果を複数件取得し、LLMに渡すことで、より網羅的な回答を生成できるようになりました。
  - **コードの進化**: 単一スクリプトから始まり、役割分離、設定の外部化（`config.yaml`）へと、保守性と拡張性を高めるリファクタリングを重ねました。

-----

## 💡 今後の改善点

  - **クラス化**: `DatabaseBuilder`や`RAGSystem`といったクラスに責務を分割し、よりオブジェクト指向な設計にする。
  - **Web UI**: `Streamlit`などを使い、対話型のWebアプリケーション化する。
  - **高度なRAG**: 検索精度をさらに高める「リランキング」や、キーワード検索を組み合わせる「ハイブリッド検索」を導入する。

-----