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

def query_rag(query_text: str, config: dict) -> str:
    """RAGの質疑応答を実行するメイン関数"""
    db_config = config['database']
    embedding_config = config['embedding']
    gen_config = config['generation']

    # 1.準備(共通)
    if not os.path.exists(db_config['persist_directory']):
        return f"エラー: データベース '{db_config['persist_directory']}' が見つかりません。"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(embedding_config['model_name'], device=device)
    client = chromadb.PersistentClient(path=db_config['persist_directory'])
    collection = client.get_collection(name=db_config['collection_name'])

    # 2.検索 (共通)
    print(f"🔍 質問「{query_text}」に基づいて情報を検索中...")
    query_embedding = embedding_model.encode(query_text, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    
    if not results['documents'][0]:
        return "関連する情報がデータベースに見つかりませんでした。"
    
    retrieved_document = results['documents'][0][0]
    print(f"✅ 関連情報を発見: 「{retrieved_document[:80]}...」")

    # 3.生成(configの値に応じて切り替え)
    if gen_config['use_local_llm']:
        local_config = gen_config['local']
        print(f"\nローカルLLM ({local_config['model_name']}) で回答を生成中...")
        
        tokenizer = AutoTokenizer.from_pretrained(local_config['model_name'])
        model = AutoModelForCausalLM.from_pretrained(
            local_config['model_name'], torch_dtype=torch.float16, load_in_4bit=True, device_map="auto"
        )
        prompt = f"""<start_of_turn>user
以下の参考情報のみを使用して、質問に答えてください。

# 参考情報
{retrieved_document}

# 質問
{query_text}<end_of_turn>
<start_of_turn>model
"""
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generation_output = model.generate(tokenized_prompt, max_new_tokens=150)
        # プロンプトとして入力した部分の長さを取得
        prompt_length = len(tokenized_prompt[0])
        # 生成された部分だけをデコード
        answer = tokenizer.decode(generation_output[0][prompt_length:], skip_special_tokens=True)
        return answer.strip()
    else:# USE_LOCAL_LLM false
        api_config = gen_config['api']
        print(f"\nGemini API ({api_config['model_name']}) で最終回答を生成中...")
        
        load_dotenv()
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            return "エラー: GOOGLE_API_KEYが.envファイルに設定されていません。"
            
        genai.configure(api_key=GOOGLE_API_KEY)
        llm = genai.GenerativeModel(api_config['model_name'])
        
        prompt = f"""以下の「参考情報」を読み、ユーザーからの「質問」に答える文章を生成してください。
回答以外の余計な言葉（「回答：」などの接頭辞や、質問の繰り返しなど）は一切含めないでください。

# 参考情報
{retrieved_document}

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

    # 連続して質問できるチャットループ
    while True:
        question = input("\n質問を入力してください (終了するには 'exit' と入力): ")
        if question.lower() == 'exit':
            print("終了します。")
            break
        
        final_answer = query_rag(question, config)
        print("\n💡 最終的な回答:")
        print("===============================")
        print(final_answer)