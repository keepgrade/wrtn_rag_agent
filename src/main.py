import os
import json
import openai
import pandas as pd
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import faiss

# ✅ 환경변수에서 API 키 로딩
openai.api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"

# ✅ 설정값
DATA_PATH = "../data/criminal_law_test.csv"
RAW_DOC_PATH = "../data/raw_docs.txt"
PROCESSED_DOC_PATH = "../data/processed_docs.json"
FAISS_INDEX_PATH = "../faiss_index/"
BATCH_INPUT_PATH = "batch_input.jsonl"
BATCH_OUTPUT_PATH = "batch_output.jsonl"
EMBEDDING_MODEL = "text-embedding-small"
GPT_MODEL = "gpt-4o-mini"
TOP_K = 5


def load_kmmlu():
    print("[1] KMMLU 평가셋 로딩")

    # "Criminal-Law" 카테고리 명시
    ds = load_dataset("HAERAE-HUB/KMMLU", "Criminal-Law", split="test")
    # 바로 DataFrame으로 변환 가능
    df = ds.to_pandas()
    df.to_csv(DATA_PATH, index=False)
    return df


def preprocess_docs():
    print("[2] RAG 문서 전처리")
    with open(RAW_DOC_PATH, encoding="utf-8") as f:
        text = f.read()
    # 문단 단위로 나누기 (빈 줄 기준)
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    with open(PROCESSED_DOC_PATH, "w", encoding="utf-8") as f:
        json.dump(paragraphs, f, ensure_ascii=False, indent=2)
    return paragraphs


def embed_paragraphs(paragraphs):
    print("[3] 문단 임베딩 + FAISS 생성")
    
    if len(paragraphs) == 0:
        print("❗ paragraphs가 비어 있습니다. 문서를 먼저 확인하세요.")
        return None, []

    embeddings = []
    for i in tqdm(range(0, len(paragraphs), 10)):
        chunk = paragraphs[i:i+10]
        response = openai.Embedding.create(
            model=EMBEDDING_MODEL,
            input=chunk
        )
        for j, r in enumerate(response["data"]):
            embeddings.append(r["embedding"])

    if len(embeddings) == 0:
        print("❗ 문단 임베딩이 실패했습니다. API 응답이 비어있습니다.")
        return None, []
    
    vectors = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(vectors)
    faiss.write_index(index, os.path.join(FAISS_INDEX_PATH, "index.bin"))
    
    try:
        with open(PROCESSED_DOC_PATH, encoding="utf-8") as f:
            paragraphs = json.load(f)
    except json.JSONDecodeError:
        print("❗ processed_docs.json 파일이 비어있거나 잘못되었습니다. 다시 전처리합니다.")
        paragraphs = preprocess_docs()
    return index, paragraphs


def retrieve_context(question, index, docs, k=TOP_K):
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=[question]
    )
    query_vec = np.array(response["data"][0]["embedding"]).astype("float32").reshape(1, -1)
    D, I = index.search(query_vec, k)
    return "\n".join([docs[i] for i in I[0]])


def generate_batch_input(df, index, docs):
    print("[4] batch_input.jsonl 생성")
    with open(BATCH_INPUT_PATH, "w", encoding="utf-8") as f:
        for i, row in tqdm(df.iterrows(), total=len(df)):
            context = retrieve_context(row["question"], index, docs)
            prompt = f"{context}\n\nQ: {row['question']}\nA:\n(보기 중 하나를 고르세요: A, B, C, D)"
            item = {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": GPT_MODEL,
                    "messages": [
                        {"role": "system", "content": "다음 질문에 보기 중 하나(A, B, C, D)로만 답하십시오."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0
                }
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def evaluate_batch_output(df):
    print("[5] batch_output.jsonl 평가")
    if not os.path.exists(BATCH_OUTPUT_PATH):
        print("❗ batch_output.jsonl 파일이 존재하지 않습니다.")
        return

    with open(BATCH_OUTPUT_PATH, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    correct = 0
    total = len(lines)
    for item in lines:
        idx = int(item["custom_id"])
        predicted = item.get("response", "").strip().upper()[0]
        answer = df.iloc[idx]["answer"].strip().upper()
        if predicted == answer:
            correct += 1

    acc = correct / total
    print(f"\n✅ 정확도: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    # Step 1: 평가셋 로딩
    if not os.path.exists(DATA_PATH):
        df = load_kmmlu()
    else:
        df = pd.read_csv(DATA_PATH)

    # Step 2: 문서 전처리
    if not os.path.exists(PROCESSED_DOC_PATH):
        paragraphs = preprocess_docs()
    else:
        try:
            with open(PROCESSED_DOC_PATH, encoding="utf-8") as f:
                paragraphs = json.load(f)
        except json.JSONDecodeError:
            print("❗ processed_docs.json이 비어 있습니다. 다시 전처리합니다.")
            paragraphs = preprocess_docs()

    # Step 3: 임베딩 + 벡터 DB 생성
    if not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.bin")):
        index, docs = embed_paragraphs(paragraphs)
    else:
        index = faiss.read_index(os.path.join(FAISS_INDEX_PATH, "index.bin"))
        with open(os.path.join(FAISS_INDEX_PATH, "docs.json"), encoding="utf-8") as f:
            docs = json.load(f)

    # Step 4: batch_input.jsonl 생성
    generate_batch_input(df, index, docs)

    # Step 5: batch_output.jsonl 평가 (응답 후 수동으로 넣어야 함)
    evaluate_batch_output(df)

