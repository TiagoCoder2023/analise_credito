import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "clientes.csv"
MODEL_PATH = "model.joblib"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    target_col = "score_credito"
    id_col = "id_cliente"
    categorical_cols = ["profissao", "mix_credito", "comportamento_pagamento"]

    x = df.drop(columns=[target_col, id_col])
    y = df[target_col]

    numeric_cols = [c for c in x.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = RandomForestClassifier(random_state=42, n_estimators=200)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    pipeline.fit(x_train, y_train)
    preds = pipeline.predict(x_test)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc:.4f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
