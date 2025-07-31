## 🧠 형사법 RAG 평가 파이프라인 (rag-agent)

📎 [Notion 문서 보기](https://www.notion.so/WRTN-241c434bf60280c68fcbc437871c0897?source=copy_link)


KMMLU 형사법 테스트셋을 기반으로 RAG (Retrieval-Augmented Generation) 구조를 활용하여 GPT 모델의 정답률을 향상시키는 파이프라인입니다.
형사법 문단들을 FAISS 벡터 DB에 임베딩한 뒤, 질문에 가장 적합한 문맥을 검색하여 GPT 모델에 전달하는 구조입니다.

---

### 📁 프로젝트 구조

```
rag-agent/
├── data/
│   ├── criminal_law_test.csv        # KMMLU 형사법 평가셋
│   ├── load_data.py                 # 평가셋 로딩 스크립트
│   ├── processed_docs.json          # 전처리된 문단 리스트
│   └── raw_docs.txt                 # 원문 문단 (빈 줄로 구분)
│
├── faiss_index/
│   └── index.bin                    # FAISS 인덱스
│
├── src/
│   ├── batch_input.jsonl            # GPT 배치 요청 데이터
│   └── main.py                      # 전체 RAG 파이프라인
│
├── .env                             # OpenAI API 키 등 환경 변수
├── docker-compose.yml              # 도커 실행 구성 (선택)
├── Dockerfile                       # 도커 컨테이너 정의
├── requirements.txt                # Python 패키지 의존성 목록
├── run.sh                          # 실행 스크립트
├── EDA.py                          # 탐색적 데이터 분석 (선택)
└── README.md                       # 프로젝트 설명 파일
```

---

### ⚙️ 설치 및 실행

#### 1. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

또는 수동 설치:

```bash
pip install openai datasets faiss-cpu scikit-learn pandas tqdm
```

#### 2. `.env` 파일 설정

`.env` 파일에 OpenAI API 키를 추가:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

### 🚀 실행 방법

```bash
python src/main.py
```

순서대로 다음 작업이 실행됩니다:

1. `criminal_law_test.csv` 평가셋 로딩 (없을 경우 자동 생성)
2. `raw_docs.txt` 전처리 → `processed_docs.json` 생성
3. 문단 임베딩 → `faiss_index/index.bin` 생성
4. GPT 입력용 `batch_input.jsonl` 생성
5. (선택) 응답 결과 `batch_output.jsonl` 평가

---

### ✍️ raw\_docs.txt 예시

```text
형법 제10조는 심신상실자의 행위는 벌하지 않는다고 규정한다.

형법 제250조는 사람을 살해한 자는 사형 또는 무기 또는 5년 이상의 징역에 처한다고 명시하고 있다.

... (50자 이상 문단 10개 이상 필요)
```

---

### 📊 정확도 평가

`batch_output.jsonl`에 OpenAI 응답을 붙여넣은 뒤 정확도 평가가 가능합니다:

```bash
✅ 정확도: 0.6850 (137/200)
```

---

### 🐳 Docker 실행 (선택)

```bash
docker build -t rag-agent .
docker run --env-file .env rag-agent
```

---

### 📬 문의 및 기여

* 기여 환영합니다! PR 또는 이슈를 통해 참여해주세요.
* 문의: [your\_email@example.com](mailto:your_email@example.com)

---

필요하시면 다음도 도와드릴 수 있습니다:

* `run.sh` 자동 실행 스크립트 정리
* `Dockerfile` 개선
* `Makefile` 추가
* Google Colab 연동 노트북

