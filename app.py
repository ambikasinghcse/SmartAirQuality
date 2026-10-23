
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

with open("air_quality_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load simulated IoT node readings
nodes_df = pd.read_csv("dataset/node_readings.csv")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    alert = None

    if request.method == "POST":
        pm25 = float(request.form["pm25"])
        pm10 = float(request.form["pm10"])
        no2 = float(request.form["no2"])
        co = float(request.form["co"])

        data = pd.DataFrame(
            [[pm25, pm10, no2, co]],
            columns=["PM2.5", "PM10", "NO2", "CO"]
        )

        prediction = round(float(model.predict(data)[0]), 2)

        if prediction <= 50:
            alert = "Good"
        elif prediction <= 100:
            alert = "Moderate"
        elif prediction <= 150:
            alert = "Unhealthy for Sensitive Groups"
        else:
            alert = "Poor - Further Assessment Needed"

    # Show the latest sample for each simulated node
    node_columns = [
        col for col in ["Node", "City", "PM2.5", "PM10", "AQI"]
        if col in nodes_df.columns
    ]

    node_records = (
        nodes_df[node_columns]
        .groupby("Node", sort=True)
        .tail(1)
        .to_dict(orient="records")
    )

    # Prepare PM2.5 readings for the graph
    graph_labels = []
    graph_values = []

    if "PM2.5" in nodes_df.columns:
        for node in sorted(nodes_df["Node"].dropna().unique()):
            node_data = nodes_df[
                nodes_df["Node"] == node
            ].dropna(subset=["PM2.5"]).tail(10)

            graph_labels.extend(
                [f"{node} - {i + 1}" for i in range(len(node_data))]
            )

            graph_values.extend(node_data["PM2.5"].tolist())

    return render_template(
        "index.html",
        prediction=prediction,
        alert=alert,
        nodes=node_records,
        graph_labels=graph_labels,
        graph_values=graph_values
    )


if __name__ == "__main__":
    app.run(debug=True)

