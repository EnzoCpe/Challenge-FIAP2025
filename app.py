# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
import plotly.express as px
import streamlit.components.v1 as components
import random # Importado para gerar variações de texto

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Opcional: Instale se não tiver
try:
    from geopy.geocoders import Nominatim
    import folium
    from streamlit_folium import st_folium
except ImportError:
    st.error("Algumas bibliotecas de mapa não foram encontradas. Execute: pip install geopy folium streamlit-folium")
    st.stop()


# ==============================
# Configuração da Página
# ==============================
st.set_page_config(
    page_title="ClickBus Challenge MVP",
    page_icon="🚍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS para expandir a largura do app e remover padding ---
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 100% !important;
        padding: 1rem 2rem 1rem 2rem !important;
    }
    iframe {
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================
# Função de leitura otimizada (cache_resource)
# ==============================
@st.cache_resource
def load_and_cache_data(filepath="base_utilizar.xlsx"):
    cache_filename = "cache_base_utilizar.parquet"

    if os.path.exists(cache_filename):
        try:
            df = pd.read_parquet(cache_filename)
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            st.warning(f"Não foi possível ler o cache. Lendo o arquivo original. Erro: {e}")
            try:
                os.remove(cache_filename)
            except Exception:
                pass
    
    if not os.path.exists(filepath):
        st.error(f"Arquivo de dados original '{filepath}' não encontrado! Coloque-o na mesma pasta que o app.py.")
        return pd.DataFrame()

    try:
        st.info(f"Arquivo de cache não encontrado. Lendo '{filepath}' pela primeira vez. Isso pode levar um momento...")
        df_raw = pd.read_excel(filepath, dtype=str)
        df_raw.columns = df_raw.columns.str.strip()
        df_raw.to_parquet(cache_filename, index=False)
        st.success("Arquivo lido e cache criado para carregamentos futuros!")
        return df_raw
    except Exception as e:
        st.error(f"Erro ao ler o arquivo Excel '{filepath}': {e}")
        return pd.DataFrame()


# ==============================
# Preprocessamento (cache_data)
# ==============================
@st.cache_data
def preprocess_data(_df, column_map, sample_fraction=1.0):
    if _df is None or _df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = _df.rename(columns=column_map).copy()

    if sample_fraction < 1.0 and 'fk_contact' in df.columns:
        customer_ids = df['fk_contact'].dropna().unique()
        if len(customer_ids) > 0:
            sampled_ids = np.random.choice(customer_ids, size=int(len(customer_ids) * sample_fraction), replace=False)
            df = df[df['fk_contact'].isin(sampled_ids)].copy()

    if 'date_purchase' in df.columns:
        df['date_purchase'] = pd.to_datetime(df['date_purchase'], errors='coerce')
        df.dropna(subset=['date_purchase'], inplace=True)
    else:
        df['date_purchase'] = pd.NaT

    if 'gmv_success' in df.columns:
        df['gmv_success'] = pd.to_numeric(df['gmv_success'], errors='coerce').fillna(0)
    else:
        df['gmv_success'] = 0

    if 'total_tickets_quantity_success' in df.columns:
        df['total_tickets_quantity_success'] = pd.to_numeric(df['total_tickets_quantity_success'], errors='coerce').fillna(0)
    else:
        df['total_tickets_quantity_success'] = 0

    if ('place_origin_departure' in df.columns) and ('place_destination_departure' in df.columns):
        df['rota'] = df['place_origin_departure'].astype(str) + " → " + df['place_destination_departure'].astype(str)
    else:
        df['rota'] = "N/A"

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    snapshot_date = df['date_purchase'].max() + dt.timedelta(days=1)

    if 'id_compra' in df.columns:
        df_customer = df.groupby('fk_contact').agg(
            recency=('date_purchase', lambda x: (snapshot_date - x.max()).days),
            frequency=('id_compra', 'nunique'),
            monetary_value=('gmv_success', 'sum')
        )
    else:
        df_customer = df.groupby('fk_contact').agg(
            recency=('date_purchase', lambda x: (snapshot_date - x.max()).days),
            frequency=('rota', 'count'),
            monetary_value=('gmv_success', 'sum')
        )

    most_frequent_route = df.groupby('fk_contact')['rota'].agg(lambda x: x.mode()[0] if not x.empty else "N/A").rename('most_frequent_route')
    df_customer = df_customer.join(most_frequent_route)
    
    # Pré-filtragem de outliers antes de retornar
    # Remove clientes cujo valor gasto está acima do quantil 99.5, que geralmente captura erros de dados
    high_value_threshold = df_customer['monetary_value'].quantile(0.995)
    df_customer = df_customer[df_customer['monetary_value'] <= high_value_threshold]
    
    df_customer.dropna(subset=['recency', 'frequency', 'monetary_value'], inplace=True)
    
    return df, df_customer


# ==============================
# Funções de Formatação e Modelagem (código mantido)
# ==============================
def formatar_valores(coluna, valor):
    if pd.isna(valor): return "-"
    if coluna == 'recency':
        return f"{int(round(valor))} dias"
    elif coluna == 'frequency':
        return f"{int(round(valor))} compras"
    elif coluna == 'monetary_value':
        if valor >= 1_000_000: return f"R$ {valor/1_000_000:,.2f} mi"
        elif valor >= 1_000: return f"R$ {valor/1_000:,.2f} mil"
        else: return f"R$ {valor:,.2f}"
    return valor

def treinar_random_forest(df_customer, days_threshold, random_state=42):
    res = {"ok": False}
    df_pred = df_customer.copy().dropna(subset=['recency', 'frequency', 'monetary_value'])
    df_pred['target'] = (df_pred['recency'] <= days_threshold).astype(int)

    if df_pred['target'].nunique() < 2:
        res['error'] = f"Amostra insuficiente para o limiar de {days_threshold} dias."
        return res

    X = df_pred[['frequency', 'monetary_value', 'recency']].values
    y = df_pred['target'].values
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=random_state, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1).fit(X_train, y_train)
    
    res.update({"ok": True, "model": model, "scaler": scaler})
    return res

def generate_explanation(customer_series, pop_df, probability, period_type):
    rec = customer_series['recency']
    freq = customer_series['frequency']
    mv = customer_series['monetary_value']
    pop_rec_median = pop_df['recency'].median()
    pop_freq_median = pop_df['frequency'].median()
    pop_mv_median = pop_df['monetary_value'].median()

    positive_factors = []
    if rec <= pop_rec_median / 2:
        positive_factors.append("sua compra recente")
    if freq > pop_freq_median * 1.5:
        positive_factors.append("sua alta frequência de compras")
    if mv > pop_mv_median * 2:
        positive_factors.append("seu alto valor gasto (GMV)")

    negative_factors = []
    if rec > pop_rec_median * 1.5:
        negative_factors.append("o tempo considerável desde a última compra")
    if freq == 1 and pop_freq_median > 1:
        negative_factors.append("ter comprado apenas uma vez")
    if mv < pop_mv_median / 2:
        negative_factors.append("seu baixo valor gasto (GMV)")

    if probability > 0.40:
        if positive_factors:
            return f"A alta probabilidade se deve principalmente a fatores como {', '.join(positive_factors)}, alinhando o cliente ao perfil de compra de {period_type}."
        else:
            return f"O modelo identificou uma forte tendência de compra de {period_type} para este cliente com base em seu padrão de consumo."
    elif probability > 0.01:
        if positive_factors and not negative_factors:
             return f"A probabilidade é impulsionada por {', '.join(positive_factors)}, mas outros fatores do seu perfil reduzem a chance de uma compra de {period_type}."
        elif not positive_factors and negative_factors:
            return f"A baixa probabilidade é justificada por fatores como {', '.join(negative_factors)}, distanciando o cliente do perfil de compra de {period_type}."
        elif positive_factors and negative_factors:
            return f"Há um misto de sinais: enquanto {', '.join(positive_factors)} o aproximam, fatores como {', '.join(negative_factors)} reduzem a probabilidade final."
        else:
            return f"O perfil deste cliente possui características mistas que resultam em uma probabilidade de compra moderada para o {period_type}."
    else:
        if negative_factors:
            return f"A probabilidade quase nula é resultado de fatores como {', '.join(negative_factors)}, que desalinham o cliente do perfil de compra esperado para o {period_type}."
        else:
            return f"O perfil de consumo do cliente não apresenta as características que o modelo aprendeu a associar com uma compra de {period_type}."

@st.cache_data
def recomendar_proxima_rota(customer_id, _df_transacoes, _df_customer_clustered):
    scores = {}
    dados_justificativa = {} 

    hist_cliente = _df_transacoes[_df_transacoes['fk_contact'] == customer_id].sort_values('date_purchase', ascending=False)
    if not hist_cliente.empty:
        rotas_freq = hist_cliente['rota'].value_counts()
        if not rotas_freq.empty:
            rota_mais_freq = rotas_freq.index[0]
            contagem_freq = rotas_freq.iloc[0]
            dados_justificativa[rota_mais_freq] = dados_justificativa.get(rota_mais_freq, {})
            dados_justificativa[rota_mais_freq]['freq'] = {'motivo': "É a sua rota preferida", 'contagem': contagem_freq, 'score': 10}
        
        ultima_rota = hist_cliente.iloc[0]['rota']
        if ultima_rota:
            dados_justificativa[ultima_rota] = dados_justificativa.get(ultima_rota, {})
            dados_justificativa[ultima_rota]['ultima'] = {'motivo': "Foi sua última viagem", 'score': 7}

    if 'cluster_nome' in _df_customer_clustered.columns:
        cluster_cliente = _df_customer_clustered.loc[customer_id, 'cluster_nome']
        if pd.notna(cluster_cliente):
            ids_cluster = _df_customer_clustered[_df_customer_clustered['cluster_nome'] == cluster_cliente].index
            rotas_cluster = _df_transacoes[_df_transacoes['fk_contact'].isin(ids_cluster)]['rota'].value_counts().head(3)
            
            peso_cluster = 3
            for rota in rotas_cluster.index:
                if rota not in dados_justificativa: 
                    dados_justificativa[rota] = dados_justificativa.get(rota, {})
                    dados_justificativa[rota]['cluster'] = {'motivo': f"Popular no segmento '{cluster_cliente}'", 'score': peso_cluster}
                peso_cluster -= 1

    for rota, dados in dados_justificativa.items():
        scores[rota] = sum(v['score'] for k, v in dados.items())

    if not scores:
        rota_popular = _df_transacoes['rota'].mode()[0]
        return rota_popular, {"fallback": {'motivo': "Não há histórico, sugerindo a rota mais popular geral."}}

    rota_recomendada = max(scores, key=scores.get)
    return rota_recomendada, dados_justificativa.get(rota_recomendada, {})

def gerar_texto_justificativa_personalizado(detalhes):
    if "fallback" in detalhes:
        return detalhes["fallback"]["motivo"]

    frases_iniciais = [
        "Nossa análise sugere esta rota por alguns motivos claros:",
        "A recomendação para esta rota se baseia principalmente em seu histórico:",
        "Chegamos a esta sugestão com base nos seguintes pontos:"
    ]
    texto = random.choice(frases_iniciais) + "\n"
    total_score = 0
    
    motivos_texto = []
    
    if 'freq' in detalhes:
        motivo = detalhes['freq']
        motivos_texto.append(f"* **{motivo['motivo']}, comprada {motivo['contagem']} vezes.** (+{motivo['score']} pontos)")
        total_score += motivo['score']
        
    if 'ultima' in detalhes:
        motivo = detalhes['ultima']
        motivos_texto.append(f"* **{motivo['motivo'].capitalize()}.** (+{motivo['score']} pontos)")
        total_score += motivo['score']

    if 'cluster' in detalhes:
        motivo = detalhes['cluster']
        motivos_texto.append(f"* **{motivo['motivo'].capitalize()}.** (+{motivo['score']} pontos)")
        total_score += motivo['score']
        
    texto += "\n".join(motivos_texto)
    texto += f"\n\n**Pontuação Total de Confiança: {total_score}**"
    
    return texto


# ==============================
# Início do App Streamlit
# ==============================
raw_df = load_and_cache_data()
if raw_df is None or raw_df.empty:
    st.stop()

# Sidebar (código mantido)
st.sidebar.title("🚍 ClickBus Challenge")
st.sidebar.markdown("### Mapeamento de colunas")
all_columns = sorted(raw_df.columns.tolist())
def find_column(options, possibilities):
    for p in possibilities:
        if p in options: return options.index(p)
    return 0
col_map = {}
col_map['fk_contact'] = st.sidebar.selectbox("Coluna ID do Cliente", all_columns, index=find_column(all_columns, ['fk_contact', 'contact_id']))
if 'id_compra' not in all_columns:
    st.sidebar.error("Coluna obrigatória 'id_compra' não encontrada. Renomeie a coluna de pedido para 'id_compra'.")
    st.stop()
col_map['id_compra'] = 'id_compra'
col_map['date_purchase'] = st.sidebar.selectbox("Coluna Data da Compra", all_columns, index=find_column(all_columns, ['date_purchase', 'date_approved', 'date_departure']))
col_map['gmv_success'] = st.sidebar.selectbox("Coluna Valor (GMV)", all_columns, index=find_column(all_columns, ['gmv_success', 'valor', 'gmv']))
col_map['total_tickets_quantity_success'] = st.sidebar.selectbox("Coluna Qtd. Passagens", all_columns, index=find_column(all_columns, ['total_tickets_quantity_success', 'total_tickets', 'tickets']))
col_map['place_origin_departure'] = st.sidebar.selectbox("Coluna Origem Partida", all_columns, index=find_column(all_columns, ['place_origin_departure', 'origin', 'origem']))
col_map['place_destination_departure'] = st.sidebar.selectbox("Coluna Destino Partida", all_columns, index=find_column(all_columns, ['place_destination_departure', 'destination', 'destino']))
col_map['nome_viacao'] = st.sidebar.selectbox("Coluna Nome da Viação", all_columns, index=find_column(all_columns, ['nome_viacao', 'empresa', 'viacao']))
column_mapping = {v: k for k, v in col_map.items()}
st.sidebar.markdown("---")
st.sidebar.subheader("Configurações de Análise")
sample_size = st.sidebar.slider("Amostragem dos Dados (%)", 10, 100, 100, 10, help="Use uma amostragem menor para uma análise mais rápida em bases de dados muito grandes.")
menu = st.sidebar.radio("Navegação:", ["Dashboard Geral", "1. Segmentação de Clientes", "2. Previsão da Próxima Compra", "3. Análise e Recomendação de Rota", "4. Indicações de Marketing"])

df, df_customer = preprocess_data(raw_df, column_mapping, sample_fraction=sample_size / 100.0)
if df.empty or df_customer.empty:
    st.warning("Dados insuficientes após preprocessamento. Verifique o mapeamento de colunas e os dados.")
    st.stop()
st.session_state.df = df
st.session_state.df_customer = df_customer

# ==============================
# Início das Páginas do App
# ==============================
if menu == "Dashboard Geral":
    # (Código mantido)
    st.title("📊 Dashboard Geral")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Faturamento Total", formatar_valores('monetary_value', df['gmv_success'].sum()))
    col2.metric("🎟️ Passagens Vendidas", f"{int(df['total_tickets_quantity_success'].sum()):,} passagens")
    col3.metric("👥 Clientes Únicos", f"{int(df['fk_contact'].nunique()):,} clientes")
    col4.metric("🛒 Pedidos Realizados", f"{int(df['id_compra'].nunique()):,} pedidos")
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("🏆 Top 10 Rotas Mais Vendidas")
        top_rotas = df['rota'].value_counts().head(10).reset_index()
        top_rotas.columns = ['Rota', 'Número de Viagens']
        st.dataframe(top_rotas)
    with c2:
        st.subheader("📈 Faturamento Mensal")
        if 'date_purchase' in df.columns and not df['date_purchase'].isna().all():
            monthly_revenue = df.set_index('date_purchase').groupby(pd.Grouper(freq='M'))['gmv_success'].sum()
            monthly_revenue_df = monthly_revenue.to_frame(name='Faturamento (R$)')
            st.line_chart(monthly_revenue_df)
        else:
            st.info("Coluna de data ausente ou inválida.")
    st.markdown("---")
    st.subheader("📊 Análise Detalhada (Power BI)")
    power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiNTUyMmIwZWEtYTZiMy00ZjkzLTllZmMtYzhhOWEwYzdmMGViIiwidCI6IjExZGJiZmUyLTg5YjgtNDU0OS1iZTEwLWNlYzM2NGU1OTU1MSIsImMiOjR9"
    iframe_code = f'<iframe title="ChallengDatacore" src="{power_bi_url}" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;" allowFullScreen="true"></iframe>'
    st.markdown(f'<div style="position: relative; width: 100%; height: 650px;">{iframe_code}</div>', unsafe_allow_html=True)

elif menu == "1. Segmentação de Clientes":
    st.title("👥 1. Segmentação de Clientes (RFM + KMeans)")
    
    k = 4
    features = ['recency', 'frequency', 'monetary_value']
    if len(df_customer) < k:
        st.error(f"Não há clientes suficientes ({len(df_customer)}) para formar {k} clusters.")
        st.stop()
    X = df_customer[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_scaled)
    df_customer['cluster'] = kmeans.labels_
    def nomear_clusters_robusto(df_all_customers, df_clustered):
        quantiles = df_all_customers[features].quantile([0.25, 0.5, 0.75]).to_dict()
        cluster_summary = df_clustered.groupby('cluster')[features].mean()
        personas = {
            "🏆 Clientes Campeões": {'recency': lambda r: r <= quantiles['recency'][0.25],'frequency': lambda f: f >= quantiles['frequency'][0.75],'monetary_value': lambda m: m >= quantiles['monetary_value'][0.75]},
            "⚠️ Clientes em Risco": {'recency': lambda r: r >= quantiles['recency'][0.75],'frequency': lambda f: f >= quantiles['frequency'][0.50],'monetary_value': lambda m: m >= quantiles['monetary_value'][0.50]},
            "😴 Clientes Inativos": {'recency': lambda r: r >= quantiles['recency'][0.75],'frequency': lambda f: f <= quantiles['frequency'][0.25]},
            "🆕 Novos Clientes": {'recency': lambda r: r <= quantiles['recency'][0.25],'frequency': lambda f: f <= quantiles['frequency'][0.25]}
        }
        cluster_scores = {cid: {p: 0 for p in personas} for cid in cluster_summary.index}
        for cid, row in cluster_summary.iterrows():
            for persona_name, rules in personas.items():
                score = sum(1 for feature, rule in rules.items() if rule(row[feature]))
                cluster_scores[cid][persona_name] = score
        final_names = {}
        assigned_personas = set()
        sorted_clusters = sorted(cluster_scores.items())
        for cid, scores in sorted_clusters:
            best_personas = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            for persona, score in best_personas:
                if persona not in assigned_personas:
                    final_names[cid] = persona
                    assigned_personas.add(persona)
                    break
            if cid not in final_names:
                final_names[cid] = "💰 Clientes Potenciais"
        return final_names
    cluster_nomes = nomear_clusters_robusto(df_customer, df_customer)
    df_customer['cluster_nome'] = df_customer['cluster'].map(cluster_nomes)
    st.session_state.df_customer_clustered = df_customer
    st.subheader("Resumo por Cluster")
    
    # --- ALTERAÇÃO AQUI: USANDO MEDIANA PARA UMA VISÃO MAIS REALISTA ---
    resumo = df_customer.groupby('cluster_nome')[features].median().round(2)
    
    resumo_formatado = resumo.copy()
    colunas_legendas = {'recency': "⏱️ Recência Mediana",'frequency': "🔁 Frequência Mediana",'monetary_value': "💰 Gasto Mediano (GMV)"}
    for col in features:
        resumo_formatado[col] = resumo[col].apply(lambda x: formatar_valores(col, x))
    resumo_formatado.rename(columns=colunas_legendas, inplace=True)
    st.dataframe(resumo_formatado)
    st.info("💡 A tabela acima usa a **mediana**, que representa o valor do 'cliente típico' de cada grupo e é mais resistente a outliers (valores extremos) do que a média.")
    
    st.subheader("Principais Rotas por Cluster")
    clusters = sorted(df_customer['cluster_nome'].dropna().unique())
    for c in clusters:
        st.markdown(f"**{c}**")
        clientes_do_cluster = df_customer[df_customer['cluster_nome'] == c].index
        rotas_do_cluster = df[df['fk_contact'].isin(clientes_do_cluster)]['rota'].value_counts().head(5)
        if not rotas_do_cluster.empty:
            tabela_rotas = rotas_do_cluster.reset_index()
            tabela_rotas.columns = ['Rota', 'Número de Viagens']
            st.dataframe(tabela_rotas)
        else:
            st.write("Nenhuma rota registrada para este cluster.")
    st.subheader("Distribuição de Clientes por Cluster")
    dist_df = df_customer['cluster_nome'].value_counts().reset_index()
    dist_df.columns = ['Cluster', 'Quantidade de Clientes']
    fig = px.pie(dist_df, values='Quantidade de Clientes', names='Cluster', title='Distribuição Percentual de Clientes por Cluster', hole=.4, color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

elif menu == "2. Previsão da Próxima Compra":
    # (Código restaurado na versão anterior)
    st.title("⏰ 2. Previsão da Próxima Compra")
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if st.button("Treinar Modelo de Previsão", disabled=st.session_state.models_trained):
        with st.spinner("Treinando modelos para 7 e 30 dias..."):
            train_res_30d = treinar_random_forest(df_customer, days_threshold=30)
            train_res_7d = treinar_random_forest(df_customer, days_threshold=7)
            if not train_res_30d.get('ok', False):
                st.error(f"Erro ao treinar modelo de 30 dias: {train_res_30d.get('error', 'Erro desconhecido.')}")
            elif not train_res_7d.get('ok', False):
                st.error(f"Erro ao treinar modelo de 7 dias: {train_res_7d.get('error', 'Erro desconhecido.')}")
            else:
                st.session_state.prediction_model_30d = train_res_30d['model']
                st.session_state.prediction_scaler_30d = train_res_30d['scaler']
                st.session_state.prediction_model_7d = train_res_7d['model']
                st.session_state.prediction_scaler_7d = train_res_7d['scaler']
                st.session_state.models_trained = True
                st.success("✅ Modelos treinados e prontos!")
                st.rerun()
    if st.session_state.models_trained:
        st.info("Modelos já treinados. Selecione um cliente para ver a previsão.")
        st.subheader("Realizar Previsão para um Cliente")
        customer_id_list = [''] + df_customer.index.tolist()
        customer_id = st.selectbox("Selecione um cliente:", options=customer_id_list)
        if customer_id:
            customer_data = df_customer.loc[[customer_id]][['frequency', 'monetary_value', 'recency']].copy()
            scaler_30d = st.session_state.prediction_scaler_30d
            model_30d = st.session_state.prediction_model_30d
            customer_scaled_30d = scaler_30d.transform(customer_data.values)
            prob_30d = model_30d.predict_proba(customer_scaled_30d)[0][1]
            scaler_7d = st.session_state.prediction_scaler_7d
            model_7d = st.session_state.prediction_model_7d
            customer_scaled_7d = scaler_7d.transform(customer_data.values)
            prob_7d = model_7d.predict_proba(customer_scaled_7d)[0][1]
            customer_series = df_customer.loc[customer_id]
            population_df = df_customer
            explanation_7d = generate_explanation(customer_series, population_df, prob_7d, "curto prazo")
            explanation_30d = generate_explanation(customer_series, population_df, prob_30d, "médio prazo")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prob. Compra (Próximos 7 dias)", f"{prob_7d:.2%}")
                st.info(f"💡 **Análise:** {explanation_7d}")
            with col2:
                st.metric("Prob. Compra (Próximos 30 dias)", f"{prob_30d:.2%}")
                st.info(f"💡 **Análise:** {explanation_30d}")
            st.markdown("---")
            col_map, col_resumo = st.columns([2, 1])
            with col_map:
                st.subheader("Visão Detalhada do Cliente")
                rota = df_customer.loc[customer_id, 'most_frequent_route']
                if isinstance(rota, str) and "→" in rota:
                    origem, destino = rota.split(" → ")
                    geolocator = Nominatim(user_agent="clickbus_app_v5")
                    try:
                        origem_coords = geolocator.geocode(origem)
                        destino_coords = geolocator.geocode(destino)
                        if origem_coords and destino_coords:
                            mapa = folium.Map(location=[(origem_coords.latitude + destino_coords.latitude)/2, (origem_coords.longitude + destino_coords.longitude)/2], zoom_start=6)
                            folium.Marker([origem_coords.latitude, origem_coords.longitude], tooltip=f"Origem: {origem}", icon=folium.Icon(color='green')).add_to(mapa)
                            folium.Marker([destino_coords.latitude, destino_coords.longitude], tooltip=f"Destino: {destino}", icon=folium.Icon(color='red')).add_to(mapa)
                            folium.PolyLine(locations=[(origem_coords.latitude, origem_coords.longitude), (destino_coords.latitude, destino_coords.longitude)], color="blue", weight=3, opacity=0.8).add_to(mapa)
                            st.markdown("##### 🗺️ Rota Mais Frequente do Cliente")
                            st_folium(mapa, width=700, height=400)
                        else:
                            st.warning("Não foi possível localizar as coordenadas para a rota no mapa.")
                    except Exception as e:
                        st.error(f"Erro ao gerar mapa: {e}")
                else:
                    st.info("Este cliente não possui uma rota frequente registrada.")
                st.markdown("**Histórico (últimas 5 compras)**")
                hist_cliente = df[df['fk_contact'] == customer_id].sort_values('date_purchase', ascending=False)
                if not hist_cliente.empty:
                    last5 = hist_cliente[['date_purchase', 'rota', 'gmv_success']].head(5)
                    last5_display = last5.copy()
                    last5_display['date_purchase'] = last5_display['date_purchase'].dt.strftime('%d/%m/%Y')
                    novos_nomes = {'date_purchase': 'Data', 'rota': 'Rota', 'gmv_success': 'Valor Pago (R$)'}
                    last5_display = last5_display.rename(columns=novos_nomes)
                    st.dataframe(last5_display.reset_index(drop=True))
                else:
                    st.info("Não há histórico de rotas para este cliente.")
            with col_resumo:
                st.subheader("💡 Análise da Rota")
                st.write("A rota exibida no mapa é o trajeto que o cliente mais realizou em seu histórico de compras.")
                st.write("Esta informação é um forte indicativo de preferência e pode ser usada para entender a probabilidade de uma nova compra. Clientes com rotas bem definidas tendem a repetir o padrão, tornando-os alvos ideais para campanhas focadas nesse trajeto específico.")

elif menu == "3. Análise e Recomendação de Rota":
    # (Código mantido)
    st.title("🛣️ 3. Análise e Recomendação de Rota")
    tab1, tab2 = st.tabs(["🔮 Previsão de Próxima Rota por Cliente", "📊 Comparador de Viações por Rota"])
    with tab1:
        st.subheader("Previsão da Próxima Rota de um Cliente")
        if 'df_customer_clustered' not in st.session_state:
            st.info("Execute a '1. Segmentação de Clientes' primeiro para habilitar a previsão.")
            st.stop()
        customer_id_list = [''] + df_customer.index.tolist()
        customer_id = st.selectbox("Selecione um cliente para prever a próxima rota:", options=customer_id_list, key="rota_pred_selectbox")
        if customer_id:
            df_cust_clustered = st.session_state.df_customer_clustered
            rota_prevista, detalhes_justificativa = recomendar_proxima_rota(customer_id, df, df_cust_clustered)
            col_pred, col_just = st.columns([2, 1])
            with col_pred:
                st.metric("🎯 Rota Mais Provável", rota_prevista)
            with col_just:
                texto_final = gerar_texto_justificativa_personalizado(detalhes_justificativa)
                st.info(texto_final)
            with st.expander("Ver detalhes do cliente e histórico"):
                st.write(f"**ID do Cliente:** {customer_id}")
                cliente_info = df_cust_clustered.loc[customer_id]
                st.write(f"**Segmento (Cluster):** {cliente_info['cluster_nome']}")
                st.markdown("**Histórico de Compras:**")
                hist_cliente = df[df['fk_contact'] == customer_id].sort_values('date_purchase', ascending=False)
                if not hist_cliente.empty:
                    hist_display = hist_cliente[['date_purchase', 'rota', 'gmv_success']].copy()
                    hist_display['date_purchase'] = hist_display['date_purchase'].dt.strftime('%d/%m/%Y')
                    hist_display.columns = ['Data', 'Rota Comprada', 'Valor (R$)']
                    st.dataframe(hist_display.reset_index(drop=True))
                else:
                    st.write("Nenhum histórico encontrado.")
    with tab2:
        st.subheader("Comparativo de Viações para uma Rota Específica")
        if 'nome_viacao' not in df.columns:
            st.error("Para usar o comparador, mapeie a coluna 'Nome da Viação' na sidebar.")
        elif pd.api.types.is_datetime64_any_dtype(df['nome_viacao']):
            st.error("Erro de Mapeamento: A coluna 'Nome da Viação' parece ser uma data. Corrija na sidebar.")
        else:
            lista_rotas = [''] + sorted(df['rota'].unique().tolist())
            rota_sugerida_idx = lista_rotas.index(df['rota'].mode()[0]) if df['rota'].mode()[0] in lista_rotas else 0
            rota_selecionada = st.selectbox("Selecione a Rota", options=lista_rotas, index=rota_sugerida_idx, key="rota_comp_selectbox")
            if rota_selecionada:
                df_rota = df[df['rota'] == rota_selecionada].copy()
                if df_rota.empty:
                    st.warning("Não há dados de viagem suficientes para esta rota.")
                else:
                    comparativo = df_rota.groupby('nome_viacao').agg(preco_medio=('gmv_success', 'mean'),num_viagens=('id_compra', 'count')).sort_values('num_viagens', ascending=False).reset_index()
                    diferenciais_db = {"Viação Cometa": ["📶 Wi-Fi Grátis", "🔌 Tomadas USB"], "Viação 1001": ["🛌 Assento Leito", "💧 Água a Bordo"],"Expresso Brasileiro": ["✅ Rota Expressa"], "Viação Kaissara": ["🐾 Pet Friendly"], "Auto Viação Catarinense": ["📺 Telas Individuais"]}
                    top_viacoes = comparativo.head(3)
                    if top_viacoes.empty:
                        st.warning("Não foi possível gerar um comparativo.")
                    else:
                        st.markdown(f"##### Comparativo para: {rota_selecionada}")
                        cols = st.columns(len(top_viacoes))
                        for i, row in top_viacoes.iterrows():
                            with cols[i]:
                                st.markdown(f"<h6>🚌 {row['nome_viacao']}</h6>", unsafe_allow_html=True)
                                st.metric("Preço Médio", f"R$ {row['preco_medio']:.2f}")
                                np.random.seed(len(row['nome_viacao']))
                                avaliacao_simulada = np.random.uniform(4.0, 4.9)
                                st.caption(f"Avaliação: {avaliacao_simulada:.1f}/5.0 ⭐")
                                st.progress(int(avaliacao_simulada * 20))
                                st.metric("Popularidade", f"{row['num_viagens']} viagens")
                                diferenciais = diferenciais_db.get(row['nome_viacao'], ["Não informado"])
                                st.markdown("**Diferenciais:**")
                                for d in diferenciais:
                                    st.markdown(f"- {d}")

elif menu == "4. Indicações de Marketing":
    # (Código mantido)
    st.title("🎯 4. Painel de Controle de Marketing")
    if 'df_customer_clustered' not in st.session_state or 'cluster_nome' not in st.session_state.df_customer_clustered.columns:
        st.info("Rode primeiro a '1. Segmentação de Clientes' para gerar os clusters e habilitar as indicações.")
        st.stop()
    df_clustered = st.session_state.df_customer_clustered
    df_transacoes = st.session_state.df
    cluster_names = sorted(df_clustered['cluster_nome'].dropna().unique().tolist())
    tabs = st.tabs([f" {name} " for name in cluster_names])
    for tab, cluster_name in zip(tabs, cluster_names):
        with tab:
            segment_df = df_clustered[df_clustered['cluster_nome'] == cluster_name]
            st.subheader(f"Visão Geral do Segmento: {cluster_name}")
            total_gmv = segment_df['monetary_value'].sum()
            num_clientes = len(segment_df)
            avg_gmv_cliente = total_gmv / num_clientes if num_clientes > 0 else 0
            kpi_cols = st.columns(3)
            kpi_cols[0].metric("👥 Nº de Clientes", f"{num_clientes:,}")
            kpi_cols[1].metric("💰 GMV Histórico Total", formatar_valores('monetary_value', total_gmv))
            kpi_cols[2].metric("👤 GMV Médio por Cliente", formatar_valores('monetary_value', avg_gmv_cliente))
            st.markdown("---")
            with st.expander("💡 Estratégias e Recomendações de Ação"):
                if "Campeões" in cluster_name:
                    st.success("**Diagnóstico:** Clientes de alto valor, compram com frequência e recentemente. São seus melhores e mais leais clientes.")
                    st.markdown("""- **Ação Principal:** Programa de Fidelidade VIP.\n- **Objetivo:** Manter a lealdade e aumentar o engajamento.\n- **Táticas:** Ofereça acesso antecipado a promoções, crie um sistema de pontos e envie descontos exclusivos.""")
                elif "Risco" in cluster_name:
                    st.warning("**Diagnóstico:** Clientes que compravam bem, mas não o fazem há algum tempo. Alto risco de churn.")
                    st.markdown("""- **Ação Principal:** Campanha de Reengajamento Personalizada.\n- **Objetivo:** Incentivar uma nova compra e evitar a perda do cliente.\n- **Táticas:** Envie um e-mail "Sentimos sua falta!" com um cupom de desconto agressivo (ex: 25% OFF) para a rota preferida deles.""")
                elif "Inativos" in cluster_name:
                    st.error("**Diagnóstico:** Clientes que não compram há muito tempo e tinham baixa frequência. Provavelmente perdidos.")
                    st.markdown("""- **Ação Principal:** Campanha de Reativação Agressiva.\n- **Objetivo:** Uma última tentativa de trazê-los de volta.\n- **Táticas:** Ofereça o maior desconto possível (ex: 50% na primeira passagem de volta) e comunique novidades. Se não houver resposta, considere removê-los da lista ativa.""")
                else: 
                    st.info("**Diagnóstico:** Clientes recentes ou com gastos moderados. Grande potencial de crescimento.")
                    st.markdown("""- **Ação Principal:** Campanha de Incentivo à 2ª Compra.\n- **Objetivo:** Garantir uma boa experiência e estimular a recompra.\n- **Táticas:** Envie e-mails de boas-vindas, ofereça um pequeno desconto para a segunda compra e peça feedback sobre a primeira viagem.""")
            with st.expander(f"👥 Ver e Exportar os {num_clientes} Clientes deste Segmento"):
                display_cols = ['recency', 'frequency', 'monetary_value', 'most_frequent_route']
                df_para_exibir = segment_df[display_cols].sort_values('recency').reset_index()
                novos_nomes_tabela = {'fk_contact': 'ID Cliente', 'recency': 'Última Compra (dias)', 'frequency': 'Nº de Compras','monetary_value': 'GMV Total (R$)', 'most_frequent_route': 'Rota Preferida'}
                df_para_exibir.rename(columns=novos_nomes_tabela, inplace=True)
                st.dataframe(df_para_exibir, use_container_width=True)
                @st.cache_data
                def convert_df_to_csv(_df):
                    return _df.to_csv(index=False).encode('utf-8')
                csv = convert_df_to_csv(df_para_exibir)
                st.download_button(label="📥 Baixar lista em CSV", data=csv, file_name=f'clientes_{cluster_name.lower().replace(" ", "_")}.csv', mime='text/csv')
