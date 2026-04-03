import pandas as pd

FILE_PATH = "data/raw/chicago_crimes_raw.csv"

print("🔍 Reading only header...")
df_head = pd.read_csv(FILE_PATH, nrows=5)

print("\nColumns:")
for col in df_head.columns:
    print("-", col)

print("\nTotal columns:", len(df_head.columns))

print(df_head.shape)
print(df_head.isna().sum())
print(df_head.head())
print(df_head.describe())