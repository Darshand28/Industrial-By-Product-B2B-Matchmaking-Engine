import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Industrial By-Product Matchmaking",
    page_icon="♻️",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
.metric-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TITLE
# ------------------------------------------------
st.title("♻️ Industrial By-Product B2B Matchmaking Engine")
st.markdown("### AI-Powered Sustainability & Waste Exchange Platform")

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
df = pd.read_csv("factory_waste_ledgers.csv")

# ------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------
st.sidebar.header("Filters")

selected_product = st.sidebar.multiselect(
    "Select ByProduct",
    df["ByProduct_Name"].unique(),
    default=df["ByProduct_Name"].unique()
)

purity_range = st.sidebar.slider(
    "Purity %",
    int(df["Purity_Pct"].min()),
    int(df["Purity_Pct"].max()),
    (70, 100)
)

filtered_df = df[
    (df["ByProduct_Name"].isin(selected_product)) &
    (df["Purity_Pct"] >= purity_range[0]) &
    (df["Purity_Pct"] <= purity_range[1])
]

# ------------------------------------------------
# TOP METRICS
# ------------------------------------------------
total_waste = filtered_df["Monthly_Volume_Tons"].sum()

total_cost = (
    filtered_df["Monthly_Volume_Tons"] *
    filtered_df["Current_Disposal_Cost_Per_Ton"]
).sum()

avg_purity = filtered_df["Purity_Pct"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("Total Waste Volume", f"{total_waste:,.0f} Tons")
col2.metric("Disposal Cost Savings", f"${total_cost:,.0f}")
col3.metric("Average Purity", f"{avg_purity:.1f}%")

# ------------------------------------------------
# DATA PREPROCESSING
# ------------------------------------------------
filtered_df["Chemical_List"] = filtered_df[
    "Chemical_Composition_Primary"
].apply(lambda x: x.split(";"))

mlb = MultiLabelBinarizer()

encoded = pd.DataFrame(
    mlb.fit_transform(filtered_df["Chemical_List"]),
    columns=mlb.classes_
)

features = pd.concat([
    encoded,
    filtered_df[["Purity_Pct", "Monthly_Volume_Tons"]]
], axis=1)

# ------------------------------------------------
# MATCHMAKING ENGINE
# ------------------------------------------------
model = NearestNeighbors(n_neighbors=4, metric="euclidean")
model.fit(features)

st.subheader("🔍 AI Matchmaking Engine")

selected_index = st.selectbox(
    "Choose Industrial Waste Source",
    filtered_df.index
)

distances, indices = model.kneighbors(
    [features.loc[selected_index]]
)

matches = filtered_df.iloc[indices[0]]

st.dataframe(matches)

# ------------------------------------------------
# CHARTS SECTION
# ------------------------------------------------
st.subheader("📊 Industrial Analytics Dashboard")

col4, col5 = st.columns(2)

# Bar Chart
fig1 = px.bar(
    filtered_df,
    x="ByProduct_Name",
    y="Monthly_Volume_Tons",
    color="ByProduct_Name",
    title="Monthly Waste Generation"
)

col4.plotly_chart(fig1, use_container_width=True)

# Pie Chart
fig2 = px.pie(
    filtered_df,
    names="ByProduct_Name",
    values="Monthly_Volume_Tons",
    title="Waste Distribution"
)

col5.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------
# COST ANALYSIS
# ------------------------------------------------
st.subheader("💰 Disposal Cost Analysis")

filtered_df["Total_Disposal_Cost"] = (
    filtered_df["Monthly_Volume_Tons"] *
    filtered_df["Current_Disposal_Cost_Per_Ton"]
)

fig3 = px.scatter(
    filtered_df,
    x="Purity_Pct",
    y="Total_Disposal_Cost",
    size="Monthly_Volume_Tons",
    color="ByProduct_Name",
    hover_name="Plant_ID",
    title="Purity vs Disposal Cost"
)

st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------
# SUSTAINABILITY SCORECARD
# ------------------------------------------------
st.subheader("🌍 Sustainability Scorecard")

carbon_offset = total_waste * 0.42

st.success(f"""
✅ Estimated Landfill Diversion: {total_waste:,.0f} Tons

✅ Estimated Carbon Offset: {carbon_offset:,.0f} Tons CO₂

✅ Estimated Industrial Savings: ${total_cost:,.0f}
""")

# ------------------------------------------------
# RAW DATA
# ------------------------------------------------
with st.expander("View Raw Dataset"):
    st.dataframe(filtered_df)
