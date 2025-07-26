import os
import glob
import torch
import yaml
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import MarkdownHeaderTextSplitter

def load_config(config_path="config.yaml"):
    """YAML設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_documents(directory_path):
    """指定されたディレクトリから様々な形式のドキュメントを読み込む"""
    loader_map = {
        ".txt": TextLoader,
        ".md": TextLoader,
        ".py": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".pptx": UnstructuredPowerPointLoader,
    }
    
    documents = []
    print(f"'{directory_path}' からドキュメントを読み込んでいます...")
    
    all_files = glob.glob(os.path.join(directory_path, "**/*"), recursive=True)
    supported_files = [f for f in all_files if os.path.splitext(f)[1].lower() in loader_map]
    
    for file_path in supported_files:
        ext = os.path.splitext(file_path)[1].lower()
        loader_class = loader_map[ext]
        
        print(f"📄 '{file_path}' を読み込み中...")
        try:
            if ext in [".txt", ".md", ".py"]:
                loader = loader_class(file_path, encoding="utf-8")
            else:
                loader = loader_class(file_path)
            
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return documents

def main():
    """データベース構築のメイン処理"""
    config = load_config()
    db_config = config['database']
    embedding_config = config['embedding']
    
    print("--- データベース構築開始 ---")

    if not os.path.exists(db_config['source_directory']):
        print(f"エラー: ソースディレクトリ '{db_config['source_directory']}' が見つかりません。")
        return

    documents = load_documents(db_config['source_directory'])
    if not documents:
        print("読み込むドキュメントが見つかりませんでした。")
        return
    print(f"\n合計 {len(documents)} 個のドキュメント（ページ）を読み込みました。")

    # Markdownの見出しレベル2, 3, 4を基準にドキュメントを分割する
    headers_to_split_on = [
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    chunks = []
    for doc in documents:
        # 各ドキュメントファイルの内容を分割
        chunks.extend(markdown_splitter.split_text(doc.page_content))
    
    print(f"ドキュメントを {len(chunks)} 個のチャンクに分割しました。")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nデバイス '{device}' を使用してEmbeddingモデルをロードします。")
    embedding_model = SentenceTransformer(embedding_config['model_name'], device=device)
    
    client = chromadb.PersistentClient(path=db_config['persist_directory'])
    
    try:
        client.delete_collection(name=db_config['collection_name'])
        print(f"既存のコレクション '{db_config['collection_name']}' を削除しました。")
    except Exception:
        pass
    
    collection = client.create_collection(name=db_config['collection_name'])

    print("チャンクをベクトル化し、データベースに保存しています...")
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding Chunks")):
        text_content = chunk.page_content
        embedding = embedding_model.encode(text_content, normalize_embeddings=True).tolist()
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[text_content]
        )

    print("\n--- データベース構築完了 ---")
    print(f"合計 {collection.count()} 個のチャンクがデータベースに保存されました。")

if __name__ == "__main__":
    main()