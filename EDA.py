# 1. Hugging Face 데이터셋 로드 함수 import
from datasets import load_dataset

# 2. 'KMMLU' 데이터셋 중 'Accounting' 서브카테고리 로드
ds = load_dataset("HAERAE-HUB/KMMLU", "Accounting")

# 3. 전체 데이터셋 구조 출력 (train/dev/test split 존재 여부 확인)
print(ds)

# 4. 각 split별 데이터 크기 확인
for split in ds:
    print(f"{split} size:", len(ds[split]))

# 5. 각 split의 feature(컬럼) 확인
print("Features:", ds['train'].features)

# 6. train split의 첫 번째 샘플 확인
print("\nSample train example:")
print(ds['train'][0])

# 7. pandas로 변환하여 EDA 하기 쉽게 준비
import pandas as pd

df_train = ds['train'].to_pandas()
df_test = ds['test'].to_pandas()
df_dev = ds['dev'].to_pandas()

# 8. 결측치 확인
print("\nNull values in train set:")
print(df_train.isnull().sum())

# 9. 정답 분포 확인
print("\nAnswer distribution:")
print(df_train['answer'].value_counts())

# 10. Human Accuracy 평균 확인
print("\nAverage Human Accuracy (train):")
print(df_train['Human Accuracy'].mean())

# 11. 예시 문항 여러 개 출력 (질문 + 선택지 + 정답)
print("\nSample questions:")
for i in range(5):
    q = df_train.iloc[i]
    print(f"\nQ{i+1}: {q['question']}")
    print(f"A. {q['A']}")
    print(f"B. {q['B']}")
    print(f"C. {q['C']}")
    print(f"D. {q['D']}")
    print(f"Answer: {q['answer']} | Human Acc: {q['Human Accuracy']}")

