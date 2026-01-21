import joblib
import pandas as pd
import streamlit as st
DATA_PATH = "clientes.csv"
MODEL_PATH = "model.joblib"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def load_table(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("Arquivo nao enviado.")
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Formato nao suportado. Use CSV ou Excel.")


def validate_columns(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    return [col for col in required_cols if col not in df.columns]


def coerce_categoricals(
    df: pd.DataFrame, categorical_cols: list[str]
) -> pd.DataFrame:
    coerced = df.copy()
    for col in categorical_cols:
        if col in coerced.columns:
            coerced[col] = coerced[col].astype(str)
    return coerced


def show_predictions(
    base_df: pd.DataFrame,
    preds,
    small_limit: int = 5,
) -> None:
    result = base_df.copy()
    result["score_previsto"] = preds
    score_map = {
        "Poor": "Ruim",
        "Standard": "Normal",
        "Good": "Bom",
        "Excellent": "Excelente",
    }
    result["score_previsto_pt"] = result["score_previsto"].map(score_map)
    result["score_previsto_pt"] = result["score_previsto_pt"].fillna(
        result["score_previsto"]
    )
    st.success("Previsao concluida!")

    if len(result) <= small_limit:
        st.subheader("Resultados (texto)")
        name_cols = ["nome", "nome_cliente", "cliente", "name"]
        name_col = next((c for c in name_cols if c in result.columns), None)
        if name_col is None and "id_cliente" in result.columns:
            name_col = "id_cliente"
        for _, row in result.iterrows():
            if name_col:
                st.write(
                    f"{row[name_col]}: {row['score_previsto_pt']}"
                )
            else:
                st.write(f"Score previsto: {row['score_previsto_pt']}")
    else:
        st.subheader("Resultados (planilha)")
        st.dataframe(result.head())
        csv_out = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar previsoes", csv_out, "previsoes.csv", "text/csv"
        )


def main() -> None:
    st.title("Analise de Credito - Demo")
    st.write("Envie uma tabela ou preencha um unico cliente.")

    model = load_model()

    target_col = "score_credito"
    id_col = "id_cliente"
    categorical_cols = ["profissao", "mix_credito", "comportamento_pagamento"]
    loan_cols = [
        "emprestimo_carro",
        "emprestimo_casa",
        "emprestimo_pessoal",
        "emprestimo_credito",
        "emprestimo_estudantil",
    ]

    modo = st.radio("Modo de previsao", ["Arquivo CSV/Excel", "Formulario"])

    if modo == "Arquivo CSV/Excel":
        st.subheader("Upload da tabela")
        uploaded = st.file_uploader("Escolha um arquivo", type=["csv", "xlsx", "xls"])
        if uploaded is not None:
            try:
                user_df = load_table(uploaded)
            except ValueError as exc:
                st.error(str(exc))
                return

            st.write("Preview da tabela enviada:")
            st.dataframe(user_df.head())

            required_cols = [
                c for c in load_data().columns if c not in [target_col, id_col]
            ]
            missing = validate_columns(user_df, required_cols)
            if missing:
                st.error(
                    "Colunas obrigatorias ausentes: " + ", ".join(missing)
                )
                return

            user_features = user_df[required_cols].copy()
            user_features = coerce_categoricals(
                user_features, categorical_cols
            )
            preds = model.predict(user_features)

            show_predictions(user_df, preds)
    else:
        df = load_data()
        feature_cols = [c for c in df.columns if c not in [target_col, id_col]]
        numeric_cols = [
            c for c in feature_cols if c not in categorical_cols + loan_cols
        ]

        defaults = df[feature_cols].copy()
        numeric_defaults = defaults[numeric_cols].median(
            numeric_only=True
        ).to_dict()
        categorical_defaults = {
            c: str(df[c].dropna().iloc[0]) for c in categorical_cols
        }
        loan_defaults = {
            c: bool(int(df[c].dropna().iloc[0])) for c in loan_cols
        }

        with st.form("input_form"):
            st.subheader("Dados do cliente")

            inputs: dict[str, object] = {}

            for col in numeric_cols:
                value = float(numeric_defaults.get(col, 0.0))
                inputs[col] = st.number_input(col, value=value)

            for col in categorical_cols:
                options = sorted(df[col].dropna().astype(str).unique().tolist())
                default = categorical_defaults.get(col, options[0] if options else "")
                inputs[col] = st.selectbox(
                    col, options, index=options.index(default)
                )

            st.subheader("Emprestimos")
            for col in loan_cols:
                inputs[col] = (
                    1 if st.checkbox(col, value=loan_defaults[col]) else 0
                )

            submitted = st.form_submit_button("Prever score")

        if submitted:
            input_df = pd.DataFrame([inputs], columns=feature_cols)
            input_df = coerce_categoricals(input_df, categorical_cols)
            prediction = model.predict(input_df)
            show_predictions(input_df, prediction)


if __name__ == "__main__":
    main()
