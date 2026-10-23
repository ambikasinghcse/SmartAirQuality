
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_air_quality.csv")

# Input features and target
features = ["PM2.5", "PM10", "NO2", "CO"]
target = "AQI"

# Keep only available columns and remove missing values
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# Train Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100, random_state=42
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluate models
print("Decision Tree MAE:", mean_absolute_error(y_test, dt_predictions))
print("Decision Tree R2:", r2_score(y_test, dt_predictions))

print("Random Forest MAE:", mean_absolute_error(y_test, rf_predictions))
print("Random Forest R2:", r2_score(y_test, rf_predictions))

# Save the trained model
with open("air_quality_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

print("Random Forest model saved successfully!")
