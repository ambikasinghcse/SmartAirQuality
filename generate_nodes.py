
import pandas as pd
import numpy as np

df = pd.read_csv("dataset/city_day.csv")

columns = ["City", "Date", "PM2.5", "PM10", "NO2", "CO", "AQI"]
available = [col for col in columns if col in df.columns]
df = df[available].copy()

pollutants = [
    col for col in ["PM2.5", "PM10", "NO2", "CO"]
    if col in df.columns
]

df = df.dropna(subset=pollutants, how="all")
df = df.sample(n=min(30, len(df)), random_state=42)
df = df.reset_index(drop=True)

rng = np.random.default_rng(42)
all_nodes = []

for i, node in enumerate(["Node 1", "Node 2", "Node 3"]):
    node_df = df.copy()
    factor = [0.90, 1.00, 1.10][i]

    for col in pollutants:
        noise = rng.uniform(0.97, 1.03, size=len(node_df))
        node_df[col] = (node_df[col] * factor * noise).round(2)

    if "AQI" in node_df.columns:
        node_df["AQI"] = (
            node_df["AQI"] * factor
        ).round(2)

    node_df["Node"] = node
    node_df["Data_Type"] = "SIMULATED_NODE_READINGS"
    all_nodes.append(node_df)

simulated = pd.concat(all_nodes, ignore_index=True)

simulated.to_csv("dataset/node_readings.csv", index=False)

print("Simulated IoT nodes updated successfully!")
print("Number of nodes:", simulated["Node"].nunique())
print("Total readings:", len(simulated))
print(simulated.groupby("Node").size())
print("Saved to dataset/node_readings.csv")


