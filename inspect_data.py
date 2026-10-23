
import pandas as pd

df = pd.read_csv("dataset/city_day.csv")

print("Dataset loaded successfully!")
print("Dataset size:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())
