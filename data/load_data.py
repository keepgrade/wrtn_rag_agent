from datasets import load_dataset
import pandas as pd

# "Criminal-Law" 카테고리 명시
ds = load_dataset("HAERAE-HUB/KMMLU", "Criminal-Law", split="test")

# 바로 DataFrame으로 변환 가능
df = ds.to_pandas()

# 저장
df.to_csv("criminal_law_test.csv", index=False)
print("Criminal-Law test셋 저장 완료!")

