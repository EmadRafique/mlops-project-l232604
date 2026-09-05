import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
 
DATA_PATH = os.path.join("data", "dataset.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")
 
# --- Hyperparameters ---

N_ESTIMATORS = 100
RANDOM_STATE = 42
LEARNING_RATE = 0.1

 
def load_data(path):
    print(f"[INFO] Loading dataset from {path} ...")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df
 
 
def preprocess(df):
    df["area_sqft"] = (df["area_sqft"] - df["area_sqft"].mean()) / df["area_sqft"].std()
    X = df.drop(columns=["price"])
    y = df["price"]
    return X, y
 
 
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    print(f"[INFO] Validation RMSE: {rmse:.2f}")
    return model
 
 
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}")
 
 
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    X, y = preprocess(df)
    model = train_model(X, y)
    save_model(model, MODEL_PATH)