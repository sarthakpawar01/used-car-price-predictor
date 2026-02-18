import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("car_dataset_clean.csv")
df = df.drop("Unnamed: 0", axis=1)

print("Dataset Loaded!")

# -----------------------
# Separate Features
# -----------------------
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# -----------------------
# Separate Categorical & Numeric
# -----------------------
categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

# -----------------------
# One Hot Encoding
# -----------------------
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X_cat = encoder.fit_transform(X[categorical_cols])
X_cat = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out())

X_num = X[numeric_cols].reset_index(drop=True)

X_final = pd.concat([X_num, X_cat], axis=1)

# -----------------------
# Train Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

# -----------------------
# Train Model
# -----------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------
# Evaluate
# -----------------------
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

# -----------------------
# Save Model + Encoder
# -----------------------
joblib.dump(model, "car_price_model.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(X_final.columns.tolist(), "model_columns.pkl")

print("Model saved successfully!")
