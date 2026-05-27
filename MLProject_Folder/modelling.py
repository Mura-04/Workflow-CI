import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# 1. Konfigurasi Path
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "heart_preprocessing.csv")

# 2. Load Data
df = pd.read_csv(data_path)
X = df.drop("target", axis=1) 
y = df["target"]

# 3. Training Model sederhana
model = RandomForestClassifier()
model.fit(X, y)

# 4. Log model ke MLflow agar 'mlflow models build-docker' bisa bekerja
with mlflow.start_run():
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="MLmodel",
        registered_model_name="HeartDiseaseModel"
    )
    print("Model berhasil di-log ke MLflow!")