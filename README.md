# Analise de Credito

Projeto simples com modelo de machine learning e uma interface web em Streamlit
para prever o score de credito de um cliente.

## Como funciona

A aplicacao oferece duas formas de previsao:

- Formulario: entrada de um unico cliente para obter o score previsto.
- Arquivo: upload de planilhas (CSV ou Excel) com previsoes em lote.

O modelo carregado em `model.joblib` recebe apenas as colunas de entrada e gera
o score para cada linha. No modo de arquivo, a aplicacao adiciona a coluna
`score_previsto` e disponibiliza o resultado para download.

## Por que foi feito assim

- Interface unica com dois modos: reduz duplicacao de codigo e facilita o uso.
- Upload de arquivo: atende o caso real de previsao em lote.
- Formulario: atende o caso de teste rapido sem precisar montar planilha.
- Validacao de colunas: evita erros quando a planilha nao tem o formato esperado.
- Codificacao de texto com `LabelEncoder`: garante que colunas com texto sejam
  transformadas em numeros antes da previsao, mantendo compatibilidade com o
  modelo treinado.
- Suporte a CSV e Excel: aumenta a chance de o usuario conseguir enviar seus
  dados sem conversoes extras.

## Desafios principais

- Manter compatibilidade entre o que o modelo espera e o que o usuario envia.
- Tratar categorias de texto diferentes das vistas no treino sem quebrar.
- Garantir que arquivos em formatos diferentes sejam lidos corretamente.

## Estrutura

- `train_model.py`: treina o modelo e salva em `model.joblib`
- `app.py`: interface visual para previsao e upload de tabelas
- `clientes.csv`: base de treino de referencia
- `model.joblib`: modelo treinado usado nas previsoes
