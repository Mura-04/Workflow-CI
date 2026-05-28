import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # 1. Konfigurasi Data (Path relatif agar jalan di Docker/Local)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "heart_preprocessing.csv")

    # 2. Cek apakah file ada
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {data_path}")

    print(f"[*] Memuat data dari: {data_path}")
    df = pd.read_csv(data_path)

    # Asumsi kolom terakhir adalah target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Training Model
    print("[*] Melatih model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluasi
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"[+] Model Accuracy: {acc:.4f}")

    # 5. Log ke MLflow
    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", registered_model_name="HeartDiseaseModel")
        mlflow.log_metric("accuracy", acc)
        print("[√] Model berhasil di-log ke MLflow!")

if __name__ == "__main__":
    main()