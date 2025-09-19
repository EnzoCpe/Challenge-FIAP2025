# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime as dt
import plotly.express as px
import streamlit.components.v1 as components

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium

# ==============================
# Configuração da Página
# ==============================
st.set_page_config(
    page_title="ClickBus Challenge MVP",
    page_icon="🚍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS para expandir a largura do app ---
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 95% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================
# Função de leitura otimizada (cache_resource)
# ==============================
@st.cache_resource
def load_and_cache_data(filepath="amostra_dados.parquet"):
    if not os.path.exists(filepath):
        st.error(f"Arquivo de amostra '{filepath}' não encontrado! Execute o script 'criar_amostra.py' primeiro.")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(filepath)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Erro ao ler o arquivo Parquet: {e}")
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
        df_customer = df.groupby('fk_contact').agg({
            'date_purchase': lambda x: (snapshot_date - x.max()).days,
            'id_compra': 'nunique',
            'gmv_success': 'sum'
        }).rename(columns={'date_purchase': 'recency', 'id_compra': 'frequency', 'gmv_success': 'monetary_value'})
    else:
        df_customer = df.groupby('fk_contact').agg({
            'date_purchase': lambda x: (snapshot_date - x.max()).days,
            'order_row_count': ('rota', 'count') if 'rota' in df.columns else ('fk_contact', 'count'),
            'gmv_success': 'sum'
        })
        if isinstance(df_customer.columns, pd.MultiIndex):
            cnt_col = [c for c in df_customer.columns if c != 'date_purchase' and c != 'gmv_success'][0]
            df_customer = df_customer.rename(columns={cnt_col: 'frequency', 'date_purchase': 'recency', 'gmv_success': 'monetary_value'})
        else:
            df_customer = df_customer.rename(columns={'date_purchase': 'recency', df_customer.columns[1]: 'frequency', 'gmv_success': 'monetary_value'})

    df_customer = df_customer.rename_axis('fk_contact').reset_index().set_index('fk_contact')
    most_frequent_route = df.groupby('fk_contact')['rota'].agg(lambda x: x.mode()[0] if not x.empty else "N/A").rename('most_frequent_route')
    df_customer = df_customer.join(most_frequent_route)
    
    df_customer.dropna(subset=['recency', 'frequency', 'monetary_value'], inplace=True)
    
    return df, df_customer


# ==============================
# Funções de Formatação e Modelagem
# ==============================
def formatar_valores(coluna, valor):
    if coluna == 'recency':
        if pd.isna(valor): return "-"
        return f"{int(round(valor))} dias"
    elif coluna == 'frequency':
        if pd.isna(valor): return "-"
        return f"{int(round(valor))} compras"
    elif coluna == 'monetary_value':
        if pd.isna(valor): return "-"
        if valor >= 1_000_000: return f"R$ {valor/1_000_000:,.2f} milhões"
        elif valor >= 1_000: return f"R$ {valor/1_000:,.2f} mil"
        else: return f"R$ {valor:,.2f}"
    return valor

def treinar_random_forest(df_customer, days_threshold, random_state=42):
    res = {"ok": False}
    df_pred = df_customer.copy().dropna(subset=['recency', 'frequency', 'monetary_value'])
    df_pred['target'] = (df_pred['recency'] <= days_threshold).astype(int)

    if df_pred['target'].nunique() < 2:
        res['error'] = f"Amostra insuficiente para o limiar de {days_threshold} dias (apenas uma classe no target)."
        return res

    X = df_pred[['frequency', 'monetary_value', 'recency']].values
    y = df_pred['target'].values
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=random_state, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1).fit(X_train, y_train)
    
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


# ==============================
# Início do App Streamlit
# ==============================
raw_df = load_and_cache_data()
if raw_df is None or raw_df.empty:
    st.stop()

# Sidebar
st.sidebar.title("🚍 ClickBus Challenge")
st.sidebar.markdown("### Mapeamento de colunas (ajuste se necessário)")
all_columns = sorted(raw_df.columns.tolist())
def find_column(options, possibilities):
    for p in possibilities:
        if p in options: return options.index(p)
    return 0

col_map = {}
col_map['fk_contact'] = st.sidebar.selectbox("Coluna ID do Cliente (fk_contact)", all_columns, index=find_column(all_columns, ['fk_contact', 'contact_id']))
if 'id_compra' not in all_columns:
    st.sidebar.error("Coluna obrigatória 'id_compra' não encontrada no arquivo. Renomeie a coluna de pedido para 'id_compra'.")
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
sample_size = st.sidebar.slider("Amostragem dos Dados (%)", 10, 100, 100, 10)
menu = st.sidebar.radio("Navegação:", ["Dashboard Geral", "1. Segmentação de Clientes", "2. Previsão da Próxima Compra", "3. Recomendação de Rota", "4. Indicações de Marketing"])

# Preprocessamento
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
            monthly_revenue = df.set_index('date_purchase').groupby(pd.Grouper(freq='ME'))['gmv_success'].sum() # Correção de 'M' para 'ME'
            monthly_revenue_df = monthly_revenue.to_frame(name='Faturamento (R$)')
            st.line_chart(monthly_revenue_df)
        else:
            st.info("Coluna de data ausente ou inválida.")
    
    st.markdown("---")
    st.subheader("📊 Análise Detalhada (Power BI)")
    
    power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiNTUyMmIwZWEtYTZiMy00ZjkzLTllZmMtYzhhOWEwYzdmMGViIiwidCI6IjExZGJiZmUyLTg5YjgtNDU0OS1iZTEwLWNlYzM2NGU1OTU1MSIsImMiOjR9"
    
    iframe_code = f'<iframe title="ChallengDatacore" src="{power_bi_url}" style="width:100%; height:600px; border:none;" allowFullScreen="true"></iframe>'
    
    components.html(iframe_code, height=620)


elif menu == "1. Segmentação de Clientes":
    st.title("👥 1. Segmentação de Clientes (RFM + KMeans)")
    
    k = 4
    features = ['recency', 'frequency', 'monetary_value']
    
    df_customer_clean = df_customer.copy()

    if len(df_customer_clean) < k:
        st.error(f"Não há clientes suficientes ({len(df_customer_clean)}) para formar {k} clusters.")
        st.stop()

    X_log = np.log1p(df_customer_clean[features])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    df_customer_clean['cluster'] = kmeans.labels_
    
    df_customer = df_customer.join(df_customer_clean['cluster'])

    def nomear_clusters(df_clusters):
        cluster_summary = df_clusters.groupby('cluster')[['recency', 'frequency', 'monetary_value']].mean()
        
        if len(cluster_summary) < 4:
            return {int(c): f"Segmento {int(c)}" for c in cluster_summary.index}

        cluster_summary['rec_rank'] = cluster_summary['recency'].rank(ascending=False)
        cluster_summary['freq_rank'] = cluster_summary['frequency'].rank(ascending=True)
        cluster_summary['mon_rank'] = cluster_summary['monetary_value'].rank(ascending=True)
        cluster_summary['rfm_score'] = cluster_summary['rec_rank'] + cluster_summary['freq_rank'] + cluster_summary['mon_rank']

        sorted_clusters = cluster_summary.sort_values('rfm_score', ascending=False)
        
        nomes = {}
        nomes[sorted_clusters.index[0]] = "🏆 Clientes Campeões"
        
        remaining = sorted_clusters.iloc[1:]
        
        worst_recency = remaining.sort_values('rec_rank').iloc[0]
        if worst_recency['freq_rank'] >= 3.0: 
             nomes[worst_recency.name] = "⚠️ Clientes em Risco"
        else:
             nomes[worst_recency.name] = "😴 Clientes Inativos"
        
        unnamed_ids = [idx for idx in remaining.index if idx not in nomes]
        unnamed_clusters = remaining.loc[unnamed_ids].sort_values('rec_rank', ascending=False)
        
        if len(unnamed_clusters) > 0:
            nomes[unnamed_clusters.index[0]] = "🆕 Novos Clientes"
        
        if len(unnamed_clusters) > 1:
            nomes[unnamed_clusters.index[1]] = "💰 Clientes Econômicos"
        
        return nomes

    cluster_nomes = nomear_clusters(df_customer)
    df_customer['cluster_nome'] = df_customer[df_customer['cluster'].notna()]['cluster'].map(cluster_nomes)
    st.session_state.df_customer_clustered = df_customer

    colunas_legendas = {
        'recency': "⏱️ Tempo desde a última compra",
        'frequency': "🔁 Número de compras",
        'monetary_value': "💰 Valor gasto"
    }
    st.subheader("Resumo por Cluster")
    resumo = df_customer.groupby('cluster_nome')[features].mean().round(2)
    resumo_formatado = resumo.copy()
    for col in features:
        resumo_formatado[col] = resumo[col].apply(lambda x: formatar_valores(col, x))
    resumo_formatado.rename(columns=colunas_legendas, inplace=True)
    st.dataframe(resumo_formatado)
    
    st.subheader("Principais rotas por cluster")
    clusters = sorted(df_customer['cluster_nome'].dropna().unique())
    for c in clusters:
        st.markdown(f"**{c}**")
        clientes_do_cluster = df_customer[df_customer['cluster_nome'] == c].index
        rotas_do_cluster = df[df['fk_contact'].isin(clientes_do_cluster)]['rota'].value_counts().head(10)
        if not rotas_do_cluster.empty:
            tabela_rotas = rotas_do_cluster.reset_index()
            tabela_rotas.columns = ['Rota', 'Número de Viagens']
            st.dataframe(tabela_rotas)
        else:
            st.write("Nenhuma rota registrada para este cluster.")

    st.subheader("Distribuição de Clientes por Cluster")
    dist_df = df_customer['cluster_nome'].value_counts().reset_index()
    dist_df.columns = ['Cluster', 'Quantidade de Clientes']
    
    dist_df = dist_df.sort_values('Quantidade de Clientes', ascending=True)
    fig = px.bar(
        dist_df,
        x='Quantidade de Clientes',
        y='Cluster',
        orientation='h',
        title='Distribuição de Clientes por Segmento',
        text='Quantidade de Clientes',
        labels={'Quantidade de Clientes': 'Nº de Clientes', 'Cluster': 'Segmento'}
    )
    fig.update_traces(textposition='inside', textfont_color='white')
    st.plotly_chart(fig, use_container_width=True)

elif menu == "2. Previsão da Próxima Compra":
    st.title("⏰ 2. Previsão da Próxima Compra")
    if 'models_trained' not in st.session_state:
        st.info("Por favor, treine os modelos de previsão primeiro.")
        if st.button("Treinar Modelos Agora"):
            with st.spinner("Treinando modelos para 7 e 30 dias..."):
                train_res_30d = treinar_random_forest(df_customer, days_threshold=30)
                train_res_7d = treinar_random_forest(df_customer, days_threshold=7)
                if not train_res_30d.get('ok', False) or not train_res_7d.get('ok', False):
                    st.error("Erro ao treinar um ou ambos os modelos.")
                else:
                    st.session_state.prediction_model_30d = train_res_30d['model']
                    st.session_state.prediction_scaler_30d = train_res_30d['scaler']
                    st.session_state.prediction_model_7d = train_res_7d['model']
                    st.session_state.prediction_scaler_7d = train_res_7d['scaler']
                    st.session_state.models_trained = True
                    st.success("✅ Modelos treinados e prontos!")
                    st.rerun()
        else:
            st.stop()

    st.subheader("Realizar Previsão para um Cliente")
    customer_id = st.selectbox("Selecione um cliente:", options=df_customer.index)
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
            st.metric("Probabilidade de Compra (Próximos 7 dias)", f"{prob_7d:.2%}")
            st.info(f"💡 **Análise do Modelo:** {explanation_7d}")
        with col2:
            st.metric("Probabilidade de Compra (Próximos 30 dias)", f"{prob_30d:.2%}")
            st.info(f"💡 **Análise do Modelo:** {explanation_30d}")
        st.markdown("---")
        col_map, col_resumo = st.columns([2, 1])
        with col_map:
            rota = df_customer.loc[customer_id, 'most_frequent_route']
            if isinstance(rota, str) and "→" in rota:
                origem, destino = rota.split(" → ")
                geolocator = Nominatim(user_agent="clickbus_app_v4")
                try:
                    origem_coords = geolocator.geocode(origem)
                    destino_coords = geolocator.geocode(destino)
                    if origem_coords and destino_coords:
                        mapa = folium.Map(location=[(origem_coords.latitude + destino_coords.latitude)/2, (origem_coords.longitude + destino_coords.longitude)/2], zoom_start=7)
                        folium.Marker([origem_coords.latitude, origem_coords.longitude], tooltip=f"Origem: {origem}").add_to(mapa)
                        folium.Marker([destino_coords.latitude, destino_coords.longitude], tooltip=f"Destino: {destino}").add_to(mapa)
                        folium.PolyLine(locations=[(origem_coords.latitude, origem_coords.longitude), (destino_coords.latitude, destino_coords.longitude)], color="blue", weight=4, opacity=0.8).add_to(mapa)
                        st.subheader("🗺️ Rota Mais Frequente do Cliente")
                        st_folium(mapa, width=700, height=450)
                    else:
                        st.warning("Não foi possível localizar as coordenadas para a rota no mapa.")
                except Exception:
                    st.error("Erro ao gerar mapa. O serviço de geolocalização pode estar indisponível.")
            else:
                st.info("Este cliente não possui uma rota frequente registrada.")
            st.markdown("**Histórico (últimas 5 rotas)**")
            hist_cliente = df[df['fk_contact'] == customer_id].sort_values('date_purchase', ascending=False)
            if not hist_cliente.empty:
                last5 = hist_cliente[['date_purchase', 'place_origin_departure', 'place_destination_departure', 'rota', 'gmv_success']].head(5)
                last5_display = last5.copy()
                last5_display['date_purchase'] = last5_display['date_purchase'].dt.strftime('%d/%m/%Y')
                last5_display['gmv_success'] = last5_display['gmv_success'].map("R$ {:,.2f}".format)
                novos_nomes = {'date_purchase': 'Data da Compra', 'place_origin_departure': 'Origem', 'place_destination_departure': 'Destino', 'rota': 'Rota', 'gmv_success': 'Valor Pago'}
                last5_display = last5_display.rename(columns=novos_nomes)
                st.dataframe(last5_display.reset_index(drop=True))
            else:
                st.info("Não há histórico de rotas para este cliente.")
            rotas_cliente = df[df['fk_contact'] == customer_id]['rota'].value_counts().head(10)
            if not rotas_cliente.empty:
                fig, ax = plt.subplots()
                rotas_cliente.sort_values().plot(kind='barh', ax=ax)
                ax.set_xlabel("Nº de Compras")
                ax.set_ylabel("Rotas")
                ax.set_title("Rotas Mais Frequentes do Cliente")
                st.pyplot(fig)
        with col_resumo:
            st.subheader("💡 Por que esta rota foi prevista?")
            st.write("A rota exibida foi definida como a próxima mais provável porque corresponde ao trajeto que o cliente mais realizou no histórico. O modelo usa recência, frequência e valor gasto para estimar probabilidade de compra e, portanto, considera rotas com alta recorrência como fortes candidatas a serem repetidas.")

elif menu == "3. Recomendação de Rota":
    st.title("🛣️ 3. Comparador de Viações")
    if 'nome_viacao' not in df.columns:
        st.error("Para usar o comparador, por favor, adicione uma coluna com o nome da viação em seu arquivo e mapeie-a na sidebar.")
        st.stop()
    if pd.api.types.is_datetime64_any_dtype(df['nome_viacao']):
        st.error("Erro de Mapeamento: A coluna selecionada para 'Nome da Viação' parece ser uma data. Por favor, corrija a seleção na sidebar.")
        st.stop()
    st.markdown("Selecione uma rota para analisar um comparativo detalhado entre as empresas de ônibus.")
    lista_rotas = sorted(df['rota'].unique().tolist())
    rota_sugerida = df['rota'].mode()[0]
    rota_selecionada = st.selectbox("Selecione a Rota", options=lista_rotas, index=lista_rotas.index(rota_sugerida))
    if st.button(f"Analisar Comparativo para: {rota_selecionada}"):
        df_rota = df[df['rota'] == rota_selecionada].copy()
        if df_rota.empty:
            st.warning("Não há dados de viagem suficientes para esta rota.")
        else:
            comparativo = df_rota.groupby('nome_viacao').agg(preco_medio=('gmv_success', 'mean'), num_viagens=('id_compra', 'count')).sort_values('num_viagens', ascending=False).reset_index()
            diferenciais_db = {"Viação Cometa": ["📶 Wi-Fi Grátis", "🔌 Tomadas USB"], "Viação 1001": ["🛌 Assento Leito", "💧 Água a Bordo"], "Expresso Brasileiro": ["✅ Rota Expressa"], "Viação Kaissara": ["🐾 Pet Friendly"], "Auto Viação Catarinense": ["📺 Telas Individuais"]}
            top_viacoes = comparativo.head(3)
            if top_viacoes.empty:
                st.warning("Não foi possível gerar um comparativo. Verifique os dados para esta rota.")
            else:
                st.markdown(f"### 📊 Comparativo para: {rota_selecionada}")
                cols = st.columns(len(top_viacoes))
                for i, row in top_viacoes.iterrows():
                    with cols[i]:
                        st.markdown(f"<h5>🚌 {row['nome_viacao']}</h5>", unsafe_allow_html=True)
                        st.metric("Preço Médio", f"R$ {row['preco_medio']:.2f}")
                        np.random.seed(len(row['nome_viacao']))
                        avaliacao_simulada = np.random.uniform(4.0, 4.9)
                        st.caption(f"Avaliação: {avaliacao_simulada:.1f}/5.0")
                        st.progress(int(avaliacao_simulada * 20))
                        st.metric("Popularidade", f"{row['num_viagens']} viagens registradas")
                        diferenciais = diferenciais_db.get(row['nome_viacao'], ["Não informado"])
                        st.markdown("**Diferenciais:**")
                        for d in diferenciais:
                            st.markdown(f"- {d}")
                st.markdown("---")
                st.subheader("Visualizações Adicionais")
                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    fig_price = px.bar(top_viacoes.sort_values('preco_medio', ascending=True), x='preco_medio', y='nome_viacao', orientation='h', title='Comparativo de Preço Médio', labels={'preco_medio': 'Preço Médio (R$)', 'nome_viacao': 'Viação'}, text='preco_medio')
                    fig_price.update_traces(texttemplate='R$ %{text:.2f}', textposition='inside')
                    st.plotly_chart(fig_price, use_container_width=True)
                with g_col2:
                    fig_share = px.pie(top_viacoes, values='num_viagens', names='nome_viacao', title='Market Share na Rota (por nº de viagens)', hole=.4)
                    st.plotly_chart(fig_share, use_container_width=True)

elif menu == "4. Indicações de Marketing":
    st.title("🎯 4. Painel de Controle de Marketing")
    if 'df_customer_clustered' not in st.session_state:
        st.info("Rode primeiro a 'Segmentação de Clientes' para gerar os clusters e habilitar as indicações.")
        st.stop()
    df_clustered = st.session_state.df_customer_clustered
    df_transacoes = st.session_state.df
    cluster_names = sorted(df_clustered['cluster_nome'].dropna().unique())
    tabs = st.tabs([f"🔹 {name}" for name in cluster_names])
    for tab, cluster_name in zip(tabs, cluster_names):
        with tab:
            segment_df = df_clustered[df_clustered['cluster_nome'] == cluster_name]
            st.subheader(f"Visão Geral do Segmento: {cluster_name}")
            total_gmv = segment_df['monetary_value'].sum()
            num_clientes = len(segment_df)
            avg_gmv_cliente = total_gmv / num_clientes if num_clientes > 0 else 0
            kpi_cols = st.columns(3)
            kpi_cols[0].metric("👥 Nº de Clientes no Segmento", f"{num_clientes:,}")
            kpi_cols[1].metric("💰 Valor Histórico Total (GMV)", formatar_valores('monetary_value', total_gmv))
            kpi_cols[2].metric("👤 GMV Médio por Cliente", formatar_valores('monetary_value', avg_gmv_cliente))
            st.markdown("---")
            with st.expander("💡 Estratégias e Recomendações de Ação"):
                if "Campeões" in cluster_name:
                    st.success("**Diagnóstico:** Clientes de alto valor, compram com frequência e recentemente. São seus melhores e mais leais clientes.")
                    st.markdown("""**Ação Principal:** Programa de Fidelidade VIP.  \n**Objetivo:** Manter a lealdade e aumentar o engajamento.  \n**Táticas:**\n1. Ofereça acesso antecipado a promoções.\n2. Crie um sistema de pontos que podem ser trocados por passagens.\n3. Envie brindes ou descontos exclusivos no aniversário de cadastro.""")
                elif "Risco" in cluster_name:
                    st.warning("**Diagnóstico:** Clientes que costumavam comprar com frequência, mas não compram há algum tempo. Alto risco de churn.")
                    clientes_risco_ids = segment_df.index
                    if not df_transacoes[df_transacoes['fk_contact'].isin(clientes_risco_ids)].empty:
                        rota_comum_risco = df_transacoes[df_transacoes['fk_contact'].isin(clientes_risco_ids)]['rota'].mode().iloc[0]
                        st.markdown(f"""**Ação Principal:** Campanha de Reengajamento Personalizada.  \n**Objetivo:** Incentivar uma nova compra e evitar a perda do cliente.  \n**Táticas:**\n1. Envie um e-mail com o assunto "Sentimos sua falta!".\n2. Ofereça um cupom de desconto agressivo (ex: 25% OFF).\n3. **Insight de Dados:** A rota mais popular para este grupo é **{rota_comum_risco}**. Foque a campanha de marketing nesta rota para máxima eficácia.""")
                elif "Inativos" in cluster_name:
                    st.error("**Diagnóstico:** Clientes que não compram há muito tempo e tinham baixa frequência. Provavelmente perdidos.")
                    st.markdown("""**Ação Principal:** Campanha de Reativação Agressiva.  \n**Objetivo:** Tentar uma última vez trazê-los de volta.  \n**Táticas:**\n1. Ofereça o maior desconto disponível (ex: 50% na primeira passagem de volta).\n2. Comunique as novidades da plataforma ou novas rotas disponíveis.\n3. Se não houver resposta, considere removê-los da lista de marketing ativo.""")
                elif "Novos" in cluster_name or "Econômicos" in cluster_name:
                    st.info("**Diagnóstico:** Clientes recentes ou que gastam pouco, geralmente com poucas compras. Potencial de se tornarem campeões.")
                    st.markdown("""**Ação Principal:** Campanha de Boas-Vindas e Incentivo à 2ª Compra.  \n**Objetivo:** Garantir uma boa primeira experiência e estimular a recompra.  \n**Táticas:**\n1. Envie uma série de e-mails de boas-vindas mostrando os benefícios da plataforma.\n2. Após a primeira viagem, envie um pequeno desconto para a segunda compra.\n3. Peça feedback sobre a primeira experiência de compra e viagem.""")
            with st.expander(f"👥 Ver e Exportar os {num_clientes} Clientes deste Segmento"):
                display_cols = ['recency', 'frequency', 'monetary_value', 'most_frequent_route']
                df_para_exibir = segment_df[display_cols].sort_values('recency').reset_index()
                novos_nomes_tabela = {'fk_contact': 'ID do Cliente', 'recency': 'Última Compra (dias)', 'frequency': 'Nº de Compras', 'monetary_value': 'GMV Total Gasto (R$)', 'most_frequent_route': 'Rota Preferida'}
                df_para_exibir.rename(columns=novos_nomes_tabela, inplace=True)
                st.dataframe(df_para_exibir, use_container_width=True)
                @st.cache_data
                def convert_df_to_csv(_df):
                    return _df.to_csv(index=False).encode('utf-8')
                csv = convert_df_to_csv(df_para_exibir)
                st.download_button(label="📥 Baixar lista em CSV", data=csv, file_name=f'clientes_{cluster_name.lower().replace(" ", "_")}.csv', mime='text/csv')
