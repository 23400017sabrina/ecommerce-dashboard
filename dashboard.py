import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style='dark')

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("all_data_sample.csv")
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

# =========================
# SIDEBAR (FILTER + GAMBAR)
# =========================
with st.sidebar:
    st.image(
        "https://img.freepik.com/vektor-premium/berbelanja-online-dengan-peralatan-teknologi-terhubung_24911-9127.jpg",
        use_container_width=True
    )
    
    st.header("Filter Dashboard")

    # FILTER TANGGAL
    min_date = df['order_purchase_timestamp'].min()
    max_date = df['order_purchase_timestamp'].max()

    start_date, end_date = st.date_input(
        "Rentang Waktu",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

    # FILTER STATE
    state = st.multiselect(
        "Pilih State",
        options=df["customer_state"].dropna().unique(),
        default=df["customer_state"].dropna().unique()
    )

    # FILTER KATEGORI
    category = st.multiselect(
        "Kategori Produk",
        options=df["product_category_name"].dropna().unique(),
        default=df["product_category_name"].dropna().unique()
    )

# =========================
# FILTER DATA
# =========================
main_df = df[
    (df['order_purchase_timestamp'] >= pd.to_datetime(start_date)) &
    (df['order_purchase_timestamp'] <= pd.to_datetime(end_date))
]

main_df = main_df[main_df["customer_state"].isin(state)]
main_df = main_df[main_df["product_category_name"].isin(category)]

# =========================
# HEADER
# =========================
st.title("Dashboard Analisis E-Commerce Olist")

# =========================
# METRIC UTAMA
# =========================
st.subheader("Ringkasan")

col1, col2, col3 = st.columns(3)

col1.metric("Total Order", main_df['order_id'].nunique())
col2.metric("Total Revenue", int(main_df['payment_value'].sum()))
col3.metric("Total Customer", main_df['customer_id'].nunique())

# =========================
# DAILY TREND
# =========================
st.subheader("Trend Order & Revenue Harian")

daily_orders = main_df.resample('D', on='order_purchase_timestamp').agg({
    'order_id': 'nunique',
    'payment_value': 'sum'
}).reset_index()

fig1, ax1 = plt.subplots()
ax1.plot(daily_orders['order_purchase_timestamp'], daily_orders['order_id'], marker='o')
ax1.set_ylabel("Order")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.plot(daily_orders['order_purchase_timestamp'], daily_orders['payment_value'], marker='o')
ax2.set_ylabel("Revenue")
st.pyplot(fig2)

# =========================
# TOP PRODUCT
# =========================
st.subheader("Top 5 Kategori Produk")

top_product = (
    main_df.groupby("product_category_name")["payment_value"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

fig3, ax3 = plt.subplots()
top_product.plot(kind='bar', ax=ax3)
st.pyplot(fig3)

# =========================
# CUSTOMER BY STATE
# =========================
st.subheader("Top Customer by State")

state_df = (
    main_df.groupby("customer_state")["customer_id"]
    .nunique()
    .sort_values(ascending=False)
    .head(5)
)

fig4, ax4 = plt.subplots()
state_df.plot(kind='bar', ax=ax4)
st.pyplot(fig4)

# =========================
# RFM ANALYSIS
# =========================
st.subheader("RFM Analysis")

rfm_df = main_df.groupby('customer_id').agg({
    'order_purchase_timestamp': 'max',
    'order_id': 'nunique',
    'payment_value': 'sum'
}).reset_index()

rfm_df.columns = ['customer_id', 'last_order', 'frequency', 'monetary']

recent_date = main_df['order_purchase_timestamp'].max()
rfm_df['recency'] = (recent_date - rfm_df['last_order']).dt.days

# NORMALISASI
rfm_df['r_rank_norm'] = (rfm_df['recency'].max() - rfm_df['recency']) / rfm_df['recency'].max()
rfm_df['f_rank_norm'] = rfm_df['frequency'] / rfm_df['frequency'].max()
rfm_df['m_rank_norm'] = rfm_df['monetary'] / rfm_df['monetary'].max()

# SCORE
rfm_df['RFM_score'] = (
    0.15 * rfm_df['r_rank_norm'] +
    0.28 * rfm_df['f_rank_norm'] +
    0.57 * rfm_df['m_rank_norm']
)

# SEGMENTASI
rfm_df['customer_segment'] = pd.cut(
    rfm_df['RFM_score'],
    bins=[0, 0.3, 0.5, 0.7, 0.85, 1],
    labels=[
        'Lost customers',
        'Low value customers',
        'Medium value customer',
        'High value customer',
        'Top customers'
    ]
)

# =========================
# CUSTOMER SEGMENTATION CHART
# =========================
st.subheader("Customer Segmentation")

segment_df = rfm_df.groupby("customer_segment")["customer_id"].nunique().reset_index()

segment_df['customer_segment'] = pd.Categorical(
    segment_df['customer_segment'],
    categories=[
        "Lost customers",
        "Low value customers",
        "Medium value customer",
        "High value customer",
        "Top customers"
    ],
    ordered=True
)

segment_df = segment_df.sort_values(by="customer_segment", ascending=False)

colors_ = ["#72BCD4", "#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

fig5, ax5 = plt.subplots(figsize=(10,5))

sns.barplot(
    x="customer_id",
    y="customer_segment",
    data=segment_df,
    palette=colors_,
    ax=ax5
)

for container in ax5.containers:
    ax5.bar_label(container)

st.pyplot(fig5)

# =========================
# FOOTER
# =========================
st.caption("Sabrina Dashboard 🚀")