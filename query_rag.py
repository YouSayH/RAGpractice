import os
import torch
import yaml
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai
from dotenv import load_dotenv

def load_config(config_path="config.yaml"):
    """YAML設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def transform_query(query_text: str, llm) -> str:
    """
    LLMを使って、元の質問を検索に適した形に変換する。
    """
    print("🔄 元の質問を検索用に変換しています...")
    
    prompt = f"""ユーザーからの以下の質問に答えるために最も適しているであろう、架空のドキュメント（文章）を生成してください。
この生成されたドキュメントは、ベクトル検索のクエリとして使用されます。回答そのものではなく、検索対象として最も理想的な文章を作成してください。

# 質問:
{query_text}

# 生成するドキュメント:
"""
    
    try:
        response = llm.generate_content(prompt)
        transformed_query = response.text.strip()
        print(f"✅ 変換後のクエリ: 「{transformed_query[:80]}...」")
        return transformed_query
    except Exception as e:
        print(f"⚠️ クエリ変換中にエラーが発生しました: {e}")
        return query_text

def query_rag(query_text: str, config: dict) -> str:
    """RAGの質疑応答を実行するメイン関数"""
    db_config = config['database']
    embedding_config = config['embedding']
    gen_config = config['generation']

    # --- 1. 準備 (共通) ---
    if not os.path.exists(db_config['persist_directory']):
        return f"エラー: データベース '{db_config['persist_directory']}' が見つかりません。"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(embedding_config['model_name'], device=device)
    client = chromadb.PersistentClient(path=db_config['persist_directory'])
    collection = client.get_collection(name=db_config['collection_name'])

    # --- 2. クエリ変換と検索 (Retrieval) ---
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        return "エラー: クエリ変換のためにGOOGLE_API_KEYが必要です。"
    genai.configure(api_key=GOOGLE_API_KEY)
    transform_llm = genai.GenerativeModel(gen_config['api']['model_name'])
    
    transformed_query = transform_query(query_text, transform_llm)
    
    print("\n-------------------------------------------")
    print(f"元の質問: {query_text}")
    print(f"変換後の検索クエリ: {transformed_query}")
    print("-------------------------------------------")

    search_query = ""
    while True:
        choice = input("どちらのクエリで検索しますか？ [1: 変換後, 2: 自分で入力]: ")
        if choice == '1':
            search_query = transformed_query
            break
        elif choice == '2':
            search_query = input("新しい検索クエリを入力してください: ")
            break
        else:
            print("1か2を入力してください。")

    print(f"\n🔍 最終クエリ「{search_query[:80]}...」で情報を検索中...")
    query_embedding = embedding_model.encode(search_query, normalize_embeddings=True).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3 
    )
    
    if not results['documents'][0]:
        return "関連する情報がデータベースに見つかりませんでした。"
    
    retrieved_documents = results['documents'][0]
    context_for_prompt = "\n\n---\n\n".join(retrieved_documents)
    
    print(f"✅ 関連情報を{len(retrieved_documents)}件発見。LLMに渡します。")

    # --- 3.生成(configの値に応じて切り替え) ---
    if gen_config['use_local_llm']:
        local_config = gen_config['local']
        print(f"\n🤖 ローカルLLM ({local_config['model_name']}) で回答を生成中...")
        
        tokenizer = AutoTokenizer.from_pretrained(local_config['model_name'])
        model = AutoModelForCausalLM.from_pretrained(
            local_config['model_name'], torch_dtype=torch.float16, load_in_4bit=True, device_map="auto"
        )
        prompt = f"""<start_of_turn>user
以下の「参考情報」の中から、ユーザーの「質問」に答えるために必要な情報だけを抜き出して、回答を生成してください。

# 参考情報
{context_for_prompt}

# 質問
{query_text}<end_of_turn>
<start_of_turn>model
"""
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generation_output = model.generate(tokenized_prompt, max_new_tokens=1024)
        prompt_length = len(tokenized_prompt[0])
        answer = tokenizer.decode(generation_output[0][prompt_length:], skip_special_tokens=True)
        return answer.strip()
    else:
        api_config = gen_config['api']
        print(f"\n🤖 Gemini API ({api_config['model_name']}) で最終回答を生成中...")
        
        llm = genai.GenerativeModel(api_config['model_name'])
        
        prompt = f"""以下の複数の「参考情報」の中から、ユーザーからの「質問」に答えるために必要な情報だけを抽出し、それに基づいて回答を生成してください。
回答以外の余計な言葉（「回答：」などの接頭辞や、質問の繰り返しなど）は一切含めないでください。

# 参考情報
{context_for_prompt}

# 質問
{query_text}
"""
        try:
            response = llm.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"エラー: Gemini APIでの回答生成に失敗しました。詳細: {e}"

if __name__ == "__main__":
    config = load_config()

    while True:
        question = input("\n質問を入力してください (終了するには 'exit' と入力): ")
        if question.lower() == 'exit':
            print("終了します。")
            break
        
        final_answer = query_rag(question, config)
        print("\n💡 最終的な回答:")
        print("===============================")
        print(final_answer)