import pandas as pd

df = pd.read_csv("data/diabetes.csv")

print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
