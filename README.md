# 🚍 Análise de Clientes e Insights de Marketing (ClickBus Challenge MVP)

## 📄 Descrição do Projeto
Este projeto é um MVP (Minimum Viable Product) desenvolvido em Streamlit que realiza uma análise completa de uma base de dados de clientes de uma empresa de transporte rodoviário. A aplicação segmenta clientes usando o modelo RFM e K-Means, prevê a probabilidade de próxima compra, compara viações em rotas específicas e fornece recomendações estratégicas de marketing.

**Link para o aplicativo online:** [Ainda não disponível - será preenchido no Passo 3]

---

## ✨ Funcionalidades

O dashboard é dividido em 4 seções principais:

* **Dashboard Geral:** Visão macro dos KPIs mais importantes, como faturamento total, passagens vendidas, clientes únicos e um dashboard interativo do Power BI integrado.
* **Segmentação de Clientes:** Clusterização de clientes em perfis (Campeões, Em Risco, Inativos, etc.) usando K-Means sobre métricas RFM (Recência, Frequência, Valor Monetário).
* **Previsão da Próxima Compra:** Utiliza um modelo de Machine Learning (Random Forest) para prever a probabilidade de um cliente selecionado realizar uma compra nos próximos 7 e 30 dias.
* **Comparador de Viações:** Uma ferramenta para analisar uma rota específica e comparar as top 3 viações com base no preço médio, popularidade e diferenciais.
* **Painel de Marketing:** Oferece estratégias de marketing acionáveis e detalhadas para cada segmento de cliente, com a opção de exportar a lista de clientes para campanhas.

---

## 🛠️ Como Executar o Projeto Localmente

1.  Clone este repositório.
2.  Crie e ative um ambiente virtual:
    ```bash
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```
3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
4.  Execute o aplicativo Streamlit:
    ```bash
    streamlit run app.py
    ```