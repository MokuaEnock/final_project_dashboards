import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & SCIENTIFIC STYLING ---
st.set_page_config(
    page_title="Nairobi Veg Index",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for a professional "Financial Terminal" look
st.markdown("""
<style>
    /* Main Background & Font */
    .main {
        background-color: #f8f9fa;
        font-family: 'Roboto', sans-serif;
    }

    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #31333F; /* Force dark text color for visibility on white background */
    }

    /* Specific targeting for Metric Labels and Values to ensure contrast */
    div[data-testid="stMetric"] > div {
        color: #31333F !important;
    }

    div[data-testid="stMetricLabel"] > div {
        color: #000000 !important; /* Force title to Black as requested */
        font-weight: bold; /* Optional: Make it bold for better visibility */
    }

    div[data-testid="stMetricValue"] > div {
        color: #31333F !important; /* Dark for value */
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #ffffff;
        border-radius: 4px;
        color: #555;
        border: 1px solid #e0e0e0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        color: #0d47a1;
        font-weight: bold;
    }

    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. ROBUST DATA LOADING ENGINE ---

@st.cache_data
def load_data():
    """
    Robust data loader updated for the new schema.
    Expected Columns: description, date, source, name, price_per_kg, category, standard_name, is_weekend, etc.
    """
    file_path = './final_data.csv'

    try:
        # Load Data
        df = pd.read_csv(file_path)

        # 1. Date Conversion
        df['date'] = pd.to_datetime(df['date'])

        # 2. Schema Validation (ensure critical columns exist)
        # Based on your review, these columns SHOULD exist.
        # We fill NaNs just in case to prevent UI crashes.
        if 'category' in df.columns:
            df['category'] = df['category'].fillna('Uncategorized')

        if 'standard_name' in df.columns:
            df['standard_name'] = df['standard_name'].fillna(df['name'])

        return df

    except FileNotFoundError:
        st.error(f"⚠️ Critical Error: The file `{file_path}` was not found.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

df = load_data()

# --- 3. SIDEBAR CONTROLS ---

with st.sidebar:
    st.title("🥬 Control Panel")

    # --- UPDATED: HIERARCHICAL SELECTOR ---
    if not df.empty:
        st.markdown("### 🔍 Filter Strategy")

        # 1. Select View Mode
        # "Category Index" aggregates everything (e.g. All Vegetables)
        # "Specific Commodity" drills down (e.g. Just Sweet Potatoes)
        view_mode = st.radio(
            "Analysis Level:",
            ["Category Index", "Specific Commodity"],
            index=0,
            help="Category Index: Aggregates all items in the category (e.g. Vegetable Index). Specific Commodity: Drills down to standard names."
        )

        # 2. Category Selector (Always visible)
        # Logic to set "Vegetable" as default index
        available_cats = sorted(df['category'].dropna().unique())

        default_cat_index = 0
        if 'Vegetable' in available_cats:
            default_cat_index = available_cats.index('Vegetable')

        selected_cat = st.selectbox("Select Category:", available_cats, index=default_cat_index)

        # 3. Commodity Selector (Conditional)
        if view_mode == "Specific Commodity":
            # Filter standard names available within the selected category
            subset_commodities = df[df['category'] == selected_cat]['standard_name'].dropna().unique()
            selected_commodity = st.selectbox("Select Commodity:", sorted(subset_commodities))

            selected_filter = selected_commodity
            filter_col = 'standard_name'
            display_title = f"{selected_commodity}"
        else:
            # Default logic (Category Index)
            selected_filter = selected_cat
            filter_col = 'category'
            display_title = f"{selected_cat} Index (Composite)"

    else:
        selected_filter = "No Data"
        filter_col = 'category'
        display_title = "No Data"

    st.divider()

    # B. Time Horizon
    time_period = st.radio(
        "Analysis Window:",
        options=[7, 21, 30, 60, 90],
        index=2,
        format_func=lambda x: f"Last {x} Days"
    )

    # C. Advanced Options
    st.markdown("### 🛠 Analysis Tools")
    show_trendline = st.checkbox("Show Trendline (OLS)", value=True, help="Overlay a linear regression trend line to identify inflation/deflation.")
    exclude_outliers = st.checkbox("Exclude Outliers (IQR)", value=False, help="Remove prices > 1.5x IQR to clean extreme spikes.")

    # D. Data Export
    st.divider()
    if not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Dataset",
            data=csv,
            file_name='nairobi_veg_index_data.csv',
            mime='text/csv',
        )

# --- 4. DATA FILTERING & PROCESSING ---

if not df.empty:
    # 1. Time Filter
    cutoff_date = df['date'].max() - timedelta(days=int(time_period))

    # 2. Apply Hierarchical Filter
    filtered_df = df[
        (df['date'] > cutoff_date) &
        (df[filter_col] == selected_filter)
    ].copy()

    # 3. Outlier Removal
    if exclude_outliers:
        Q1 = filtered_df['price_per_kg'].quantile(0.25)
        Q3 = filtered_df['price_per_kg'].quantile(0.75)
        IQR = Q3 - Q1
        filtered_df = filtered_df[~((filtered_df['price_per_kg'] < (Q1 - 1.5 * IQR)) | (filtered_df['price_per_kg'] > (Q3 + 1.5 * IQR)))]

    # 4. Market Aggregates (Daily)
    daily_stats = filtered_df.groupby('date')['price_per_kg'].agg(
        min_price='min',
        max_price='max',
        avg_price='mean',
        std_dev='std'
    ).reset_index()
    daily_stats['spread'] = daily_stats['max_price'] - daily_stats['min_price']

else:
    filtered_df = pd.DataFrame()
    daily_stats = pd.DataFrame()


# --- 5. MAIN DASHBOARD UI ---

# Header
st.title(f"📊 {display_title}: Market Intelligence")
st.markdown(f"*Analysis Period: {cutoff_date.date()} to {datetime.today().date()}*")

# --- TOP ROW: KPI METRICS ---
if not filtered_df.empty:
    curr_avg = daily_stats.iloc[-1]['avg_price']
    prev_avg = daily_stats.iloc[0]['avg_price'] # Price at start of period
    pct_change = ((curr_avg - prev_avg) / prev_avg) * 100 if prev_avg != 0 else 0

    avg_spread = daily_stats['spread'].mean()
    volatility = filtered_df['price_per_kg'].std()

    # Find Cheapest Retailer (Mode)
    if not filtered_df.empty:
        min_price_df = filtered_df[filtered_df['price_per_kg'] == filtered_df['price_per_kg'].min()]
        if not min_price_df.empty:
             cheapest_source = min_price_df['source'].mode()[0]
        else:
             cheapest_source = "N/A"
    else:
        cheapest_source = "N/A"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Current Market Price", f"KES {curr_avg:.0f}/kg", f"{pct_change:.1f}%", delta_color="inverse")
    k2.metric("Market Inefficiency (Spread)", f"KES {avg_spread:.0f}", help="Avg difference between High and Low price")
    k3.metric("Volatility Index (StdDev)", f"{volatility:.2f}", help="Higher number = More unstable prices")
    k4.metric("Best Price Leader", cheapest_source, help="Retailer offering the lowest price most frequently")

st.divider()

# --- TABS SECTION ---
tab1, tab2, tab3 = st.tabs([
    "📈 Market Dynamics (Trends)",
    "🆚 Competitor Intelligence",
    "🔬 Statistical Hypothesis"
])

# === TAB 1: MARKET DYNAMICS ===
with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Price Dispersion & Trend Analysis")

        if not daily_stats.empty:
            fig = go.Figure()

            # A. The Corridor (Spread)
            fig.add_trace(go.Scatter(
                x=daily_stats['date'], y=daily_stats['max_price'],
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=daily_stats['date'], y=daily_stats['min_price'],
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(0, 100, 80, 0.1)',
                name='Arbitrage Zone (Spread)'
            ))

            # B. Average Price
            fig.add_trace(go.Scatter(
                x=daily_stats['date'], y=daily_stats['avg_price'],
                mode='lines', line=dict(color='#2E86C1', width=3),
                name='Market Weighted Avg'
            ))

            # C. Trendline (Scientific Feature)
            if show_trendline and len(daily_stats) > 1:
                x_nums = (daily_stats['date'] - daily_stats['date'].min()).dt.days
                coef = np.polyfit(x_nums, daily_stats['avg_price'], 1)
                poly1d_fn = np.poly1d(coef)

                fig.add_trace(go.Scatter(
                    x=daily_stats['date'], y=poly1d_fn(x_nums),
                    mode='lines', line=dict(color='red', width=2, dash='dot'),
                    name=f'Trend (Slope: {coef[0]:.2f})'
                ))

            fig.update_layout(height=450, template="plotly_white", hovermode="x unified",
                              yaxis_title="Price (KES/Kg)", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for chart.")

    with col2:
        st.subheader("Daily Volatility")
        if not daily_stats.empty:
            # Candle Chart for Daily Range
            fig_candle = go.Figure(data=[go.Candlestick(
                x=daily_stats['date'],
                open=daily_stats['avg_price'],
                high=daily_stats['max_price'],
                low=daily_stats['min_price'],
                close=daily_stats['avg_price'],
                name="Daily Spread"
            )])
            fig_candle.update_layout(height=450, template="plotly_white", showlegend=False,
                                     title="Daily High/Low Spread", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_candle, use_container_width=True)


# === TAB 2: COMPETITOR INTELLIGENCE ===
with tab2:
    if filtered_df.empty:
        st.warning("No data.")
    else:
        c1, c2 = st.columns([2, 1])

        # A. Line Chart: Strategy Comparison
        with c1:
            st.subheader("Retailer Pricing Strategy")
            pivot_df = filtered_df.groupby(['date', 'source'])['price_per_kg'].mean().reset_index()
            fig2 = px.line(
                pivot_df, x='date', y='price_per_kg', color='source',
                color_discrete_sequence=px.colors.qualitative.Bold,
                template="plotly_white"
            )
            fig2.update_layout(height=400, hovermode="x unified", yaxis_title="Price (KES/Kg)")
            st.plotly_chart(fig2, use_container_width=True)

        # B. Win Rate Analysis (Who is cheapest most often?)
        with c2:
            st.subheader("Market Dominance (Win Rate)")
            # Identify who had the min price for each day
            daily_mins = filtered_df.groupby('date')['price_per_kg'].min().reset_index()
            # Merge back to get the retailer name
            winners = pd.merge(filtered_df, daily_mins, on=['date', 'price_per_kg'])
            win_counts = winners['source'].value_counts().reset_index()
            win_counts.columns = ['Retailer', 'Days Cheapest']

            fig_pie = px.pie(win_counts, values='Days Cheapest', names='Retailer', hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Bold)
            fig_pie.update_layout(height=400, showlegend=True, legend=dict(orientation="h"))
            st.plotly_chart(fig_pie, use_container_width=True)


# === TAB 3: STATISTICAL INSIGHTS ===
with tab3:
    st.markdown("### 🔬 Hypothesis Testing & Distribution Analysis")

    col_a, col_b = st.columns(2)

    with col_a:
        # 1. The Weekend Effect
        # Ensure 'is_weekend' exists and handle boolean/numeric
        if 'is_weekend' in filtered_df.columns:
            st.markdown("#### Hypothesis 1: The 'Weekend Effect'")
            st.caption("Null Hypothesis ($H_0$): Mean prices on Weekends are equal to Weekdays.")

            # Map robustly (handles 1/0 or True/False)
            filtered_df['Day Type'] = filtered_df['is_weekend'].apply(lambda x: 'Weekend' if x in [True, 1, 'True'] else 'Weekday')

            fig3 = px.box(
                filtered_df, x="Day Type", y="price_per_kg", color="Day Type",
                points="all",
                color_discrete_map={'Weekend': '#FF5733', 'Weekday': '#3498DB'},
                template="plotly_white"
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Stats
            wknd_df = filtered_df[filtered_df['Day Type'] == 'Weekend']
            wkday_df = filtered_df[filtered_df['Day Type'] == 'Weekday']

            if not wknd_df.empty and not wkday_df.empty:
                wknd_mean = wknd_df['price_per_kg'].mean()
                wkday_mean = wkday_df['price_per_kg'].mean()
                premium = ((wknd_mean - wkday_mean) / wkday_mean) * 100
                st.info(f"**Insight:** Weekends trade at a **{premium:+.2f}%** premium compared to weekdays.")
            else:
                st.caption("Insufficient data for weekend calculation.")

        else:
            st.warning("⚠️ 'is_weekend' feature not found. Please check data engineering step.")

    with col_b:
        # 2. Distribution Analysis (Violin)
        st.markdown("#### Distribution Density")
        st.caption("Visualizing the spread and consistency of pricing strategies.")

        fig4 = px.violin(
            filtered_df, x="source", y="price_per_kg", color="source",
            box=True, points=False, # cleaner look
            title="Retailer Price Density",
            template="plotly_white"
        )
        st.plotly_chart(fig4, use_container_width=True)

    # 3. Descriptive Stats Table
    st.markdown("#### 📋 Descriptive Statistics Summary")
    if not filtered_df.empty:
        desc_stats = filtered_df.groupby('source')['price_per_kg'].describe()[['count', 'mean', 'std', 'min', 'max']].sort_values('mean')
        st.dataframe(desc_stats.style.format("{:.2f}"), use_container_width=True)
