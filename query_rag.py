import os
import torch
import yaml
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- LLMプロバイダーに応じて必要なライブラリをインポート ---
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except ImportError:
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig = None, None, None

try:
    import openai
except ImportError:
    openai = None


def load_config(config_path="config.yaml"):
    """YAML設定ファイルを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def transform_query(query_text: str, config: dict) -> str:
    """LLMを使って、元の質問を検索に適した形に変換する (Gemini APIを使用)"""
    print("🔄 元の質問を検索用に変換しています...")
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    if not GOOGLE_API_KEY or not genai:
        print("⚠️ GOOGLE_API_KEYが見つからないか、'google-generativeai'が未インストールのため、クエリ変換をスキップします。")
        return query_text

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        transform_llm = genai.GenerativeModel(config['generation']['gemini']['model_name'])
        
        prompt = f"""ユーザーからの以下の質問に答えるために最も適しているであろう、架空のドキュメント（文章）を生成してください。
この生成されたドキュメントは、ベクトル検索のクエリとして使用されます。回答そのものではなく、検索対象として最も理想的な文章を作成してください。

# 質問:
{query_text}

# 生成するドキュメント:
"""
        response = transform_llm.generate_content(prompt)
        transformed_query = response.text.strip()
        print(f"✅ 変換後のクエリ: 「{transformed_query[:80]}...」")
        return transformed_query
    except Exception as e:
        print(f"⚠️ クエリ変換中にエラーが発生しました: {e}")
        return query_text

def generate_with_gemini(prompt: str, config: dict) -> str:
    """Gemini APIで回答を生成"""
    if not genai:
        return "エラー: 'google-generativeai'がインストールされていません。"
    api_config = config['generation']['gemini']
    print(f"\n🤖 Gemini API ({api_config['model_name']}) で回答を生成中...")
    try:
        llm = genai.GenerativeModel(api_config['model_name'])
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"エラー: Gemini APIでの回答生成に失敗しました。詳細: {e}"

def generate_with_huggingface(prompt: str, device: str, config: dict) -> str:
    """Hugging Faceのモデルを直接実行して回答を生成"""
    if not AutoTokenizer:
        return "エラー: 'transformers'または'bitsandbytes'がインストールされていません。"
    local_config = config['generation']['huggingface']
    print(f"\n🤖 Hugging Faceモデル ({local_config['model_name']}) で回答を生成中...")
    
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(local_config['model_name'])
        model = AutoModelForCausalLM.from_pretrained(
            local_config['model_name'],
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generation_output = model.generate(tokenized_prompt, max_new_tokens=1024, temperature=0.7)
        prompt_length = len(tokenized_prompt[0])
        answer = tokenizer.decode(generation_output[0][prompt_length:], skip_special_tokens=True)
        return answer.strip()
    except Exception as e:
        return f"エラー: Hugging Faceモデルの実行に失敗しました。詳細: {e}"

def generate_with_lmstudio(prompt: str, config: dict) -> str:
    """LM Studio経由で回答を生成"""
    if not openai:
        return "エラー: 'openai'ライブラリがインストールされていません。"
    local_config = config['generation']['lmstudio']
    print(f"\n🤖 LM Studio ({local_config['base_url']}) 経由で回答を生成中...")
    
    try:
        client = openai.OpenAI(base_url=local_config['base_url'], api_key=local_config['api_key'])
        completion = client.chat.completions.create(
            model=local_config['model'],
            messages=[
                {"role": "system", "content": "あなたは優秀なAIアシスタントです。提供された参考情報に基づいて、日本語で質問に答えてください。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ LM Studioとの通信中にエラーが発生しました: {e}\nLM Studioでサーバーが起動しているか、設定が正しいか確認してください。"


def query_rag(query_text: str, config: dict) -> str:
    """RAGの質疑応答を実行するメイン関数"""
    db_config = config['database']
    embedding_config = config['embedding']
    gen_config = config['generation']
    provider = gen_config.get('provider', 'gemini') # デフォルトはgemini

    # --- 1. 準備 (共通) ---
    if not os.path.exists(db_config['persist_directory']):
        return f"エラー: データベース '{db_config['persist_directory']}' が見つかりません。"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(embedding_config['model_name'], device=device)
    client = chromadb.PersistentClient(path=db_config['persist_directory'])
    collection = client.get_collection(name=db_config['collection_name'])

    # --- 2. クエリ変換と検索 (Retrieval) ---
    transformed_query = transform_query(query_text, config)
    
    print("\n-------------------------------------------")
    print(f"元の質問: {query_text}")
    print(f"変換後の検索クエリ: {transformed_query}")
    print("-------------------------------------------")

    search_query = transformed_query
    if transformed_query != query_text:
        while True:
            choice = input("どちらのクエリで検索しますか？ [1: 変換後, 2: 自分で入力, Enter: 変換後]: ")
            if choice == '1' or choice == '':
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
    
    context_for_prompt = "\n\n---\n\n".join(results['documents'][0])
    print(f"✅ 関連情報を{len(results['documents'][0])}件発見。LLMに渡します。")

    # --- 3. プロンプト作成と生成 ---
    # プロンプトのテンプレート
    base_prompt = f"""以下の「参考情報」だけを厳密に参照し、ユーザーの「質問」に答えてください。
参考情報に答えや関連情報がない場合は、憶測で答えず、正直に「分かりません」と回答してください。

# 参考情報
{context_for_prompt}

# 質問
{query_text}
"""
    
    # プロバイダーに応じて処理を分岐
    if provider == 'gemini':
        return generate_with_gemini(base_prompt, config)
        
    elif provider == 'huggingface':
        # Hugging Face (Gemma) 用のプロンプト形式に変換
        hf_prompt = f"<start_of_turn>user\n{base_prompt}<end_of_turn>\n<start_of_turn>model\n"
        return generate_with_huggingface(hf_prompt, device, config)
        
    elif provider == 'lmstudio':
        # LM Studio (Chat形式) では、プロンプトをそのまま渡す
        return generate_with_lmstudio(base_prompt, config)
        
    else:
        return f"エラー: 設定ファイルにあるプロバイダー '{provider}' は無効です。"


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