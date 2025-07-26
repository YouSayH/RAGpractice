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
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
    chunking_config = config['chunking']
    
    print("--- データベース構築開始 ---")

    if not os.path.exists(db_config['source_directory']):
        print(f"エラー: ソースディレクトリ '{db_config['source_directory']}' が見つかりません。")
        return

    documents = load_documents(db_config['source_directory'])
    if not documents:
        print("読み込むドキュメントが見つかりませんでした。")
        return
    print(f"\n合計 {len(documents)} 個のドキュメント（ページ）を読み込みました。")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunking_config['chunk_size'],
        chunk_overlap=chunking_config['chunk_overlap']
    )
    chunks = text_splitter.split_documents(documents)
    print(f"ドキュメントを {len(chunks)} 個のチャンクに分割しました。")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nデバイス '{device}' を使用してEmbeddingモデルをロードします。")
    embedding_model = SentenceTransformer(embedding_config['model_name'], device=device)
    
    client = chromadb.PersistentClient(path=db_config['persist_directory'])

    # データベースのクリーンな再構築
    # 以下の処理により、スクリプト実行時に常にデータベースをまっさらな状態から作り直します。
    # これにより、元ファイルが削除された場合に、古い情報がDBに残り続けるのを防ぎます。
    try:
        client.delete_collection(name=db_config['collection_name'])
        print(f"既存のコレクション '{db_config['collection_name']}' を削除しました。")
    except Exception:# コレクションが存在しない場合は何もしない
        pass
    
    collection = client.create_collection(name=db_config['collection_name'])

    print("チャンクをベクトル化し、データベースに保存しています...")
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding Chunks")):
        text = chunk.page_content
        embedding = embedding_model.encode(text, normalize_embeddings=True).tolist()
        collection.add(ids=[f"chunk_{i}"], embeddings=[embedding], documents=[text])

    print("\n--- データベース構築完了 ---")
    print(f"合計 {collection.count()} 個のチャンクがデータベースに保存されました。")

if __name__ == "__main__":
    main()