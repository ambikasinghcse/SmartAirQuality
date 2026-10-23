
import pandas as pd

# Load the real air quality dataset
df = pd.read_csv("dataset/city_day.csv")

print("Original dataset shape:", df.shape)

# Select useful columns for machine learning
columns = ["PM2.5", "PM10", "NO2", "CO", "AQI"]
available_columns = [col for col in columns if col in df.columns]

df = df[available_columns].copy()

# Remove rows where AQI is missing
df = df.dropna(subset=["AQI"])

# Fill missing pollution readings with their median
for col in available_columns:
    if col != "AQI":
        df[col] = df[col].fillna(df[col].median())

# Remove rows with missing values that could not be filled
df = df.dropna()

# Save the cleaned dataset
df.to_csv("dataset/cleaned_air_quality.csv", index=False)

print("Dataset cleaned successfully!")
print("Cleaned dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Saved to dataset/cleaned_air_quality.csv")
