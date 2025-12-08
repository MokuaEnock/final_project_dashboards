import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Nairobi Veg Price Tracker",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. IMPROVED STYLING ---
st.markdown("""
<style>
    .main {
        background-color: var(--background-color);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Welcome Banner */
    .welcome-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .welcome-banner h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }

    .welcome-banner p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* Info Cards */
    .info-card {
        background-color: var(--secondary-background-color);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .info-card h3 {
        color: var(--text-color);
        margin-top: 0;
    }

    .info-card p {
        color: var(--text-color);
        margin-bottom: 0;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: var(--background-color);
        border: 1px solid var(--secondary-background-color);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: var(--text-color) !important;
        font-weight: 500;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: var(--text-color) !important;
        font-weight: bold;
    }

    div[data-testid="stMetricDelta"] {
        color: var(--text-color) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
    }

    /* Help text */
    .help-text {
        background-color: rgba(33, 150, 243, 0.1);
        padding: 0.8rem;
        border-radius: 5px;
        border-left: 3px solid #2196f3;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: var(--text-color);
    }

    .help-text b {
        color: var(--text-color);
    }

    /* Step indicators */
    .step-indicator {
        background-color: #667eea;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. DATA LOADING ---
@st.cache_data
def load_data():
    file_path = './final_data.csv'
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])

        if 'category' in df.columns:
            df['category'] = df['category'].fillna('Uncategorized')
        if 'standard_name' in df.columns:
            df['standard_name'] = df['standard_name'].fillna(df['name'])

        return df
    except FileNotFoundError:
        st.error(f"⚠️ Data file not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# --- 4. WELCOME SECTION ---
if df.empty:
    st.error("📁 No data available. Please ensure 'final_data.csv' is in the correct location.")
    st.stop()

# Show welcome banner only on first load
if 'welcomed' not in st.session_state:
    st.markdown("""
    <div class="welcome-banner">
        <h1>🥬 Nairobi Vegetable Price Tracker</h1>
        <p>Track and compare vegetable prices across Nairobi markets in real-time. Make smarter shopping decisions!</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>📊 What This Does</h3>
            <p>Compare prices from different markets and retailers to find the best deals on vegetables.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>💡 How to Use</h3>
            <p>Use the sidebar to select a vegetable, choose a time period, and see price trends instantly.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Your Benefit</h3>
            <p>Save money by knowing which retailer offers the best price and when to buy.</p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🚀 Get Started", type="primary", use_container_width=True):
        st.session_state.welcomed = True
        st.rerun()

    st.stop()

# --- 5. SIDEBAR WITH GUIDED STEPS ---
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    st.markdown("Follow these simple steps:")

    st.markdown("---")

    # STEP 1: Choose what to track
    st.markdown('<span class="step-indicator">1</span> **Choose What to Track**', unsafe_allow_html=True)

    view_mode = st.radio(
        "I want to see:",
        ["📦 All vegetables in a category", "🥕 A specific vegetable"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown('<div class="help-text">💡 <b>Tip:</b> Start with a category to see the overall market, then drill down to specific vegetables.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # STEP 2: Select category/commodity
    st.markdown('<span class="step-indicator">2</span> **Select Your Item**', unsafe_allow_html=True)

    available_cats = sorted(df['category'].dropna().unique())
    default_cat_index = available_cats.index('Vegetable') if 'Vegetable' in available_cats else 0

    selected_cat = st.selectbox(
        "Category:",
        available_cats,
        index=default_cat_index,
        help="Choose a broad category like Vegetables, Fruits, etc."
    )

    if view_mode == "🥕 A specific vegetable":
        subset_commodities = df[df['category'] == selected_cat]['standard_name'].dropna().unique()
        selected_commodity = st.selectbox(
            "Specific vegetable:",
            sorted(subset_commodities),
            help="Pick the exact vegetable you want to track"
        )
        selected_filter = selected_commodity
        filter_col = 'standard_name'
        display_title = f"{selected_commodity}"
    else:
        selected_filter = selected_cat
        filter_col = 'category'
        display_title = f"{selected_cat} (All Items)"

    st.markdown("---")

    # STEP 3: Time period
    st.markdown('<span class="step-indicator">3</span> **Choose Time Period**', unsafe_allow_html=True)

    time_period = st.select_slider(
        "Show me the last:",
        options=[7, 14, 30, 60, 90],
        value=30,
        format_func=lambda x: f"{x} days",
        help="How far back do you want to see price history?"
    )

    st.markdown('<div class="help-text">💡 <b>Recommended:</b> Use 30 days for a balanced view of recent trends.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # STEP 4: Optional filters
    with st.expander("⚙️ Advanced Options (Optional)"):
        show_trendline = st.checkbox(
            "Show price trend line",
            value=True,
            help="Adds a line showing if prices are going up or down overall"
        )
        exclude_outliers = st.checkbox(
            "Remove unusual prices",
            value=False,
            help="Filters out extreme price spikes that might be data errors"
        )

    st.markdown("---")
    st.markdown("### 📥 Export Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full Dataset",
        data=csv,
        file_name='nairobi_veg_prices.csv',
        mime='text/csv',
        use_container_width=True
    )

# --- 6. DATA PROCESSING ---
cutoff_date = df['date'].max() - timedelta(days=int(time_period))

filtered_df = df[
    (df['date'] > cutoff_date) &
    (df[filter_col] == selected_filter)
].copy()

if exclude_outliers and not filtered_df.empty:
    Q1 = filtered_df['price_per_kg'].quantile(0.25)
    Q3 = filtered_df['price_per_kg'].quantile(0.75)
    IQR = Q3 - Q1
    filtered_df = filtered_df[
        ~((filtered_df['price_per_kg'] < (Q1 - 1.5 * IQR)) |
          (filtered_df['price_per_kg'] > (Q3 + 1.5 * IQR)))
    ]

if not filtered_df.empty:
    daily_stats = filtered_df.groupby('date')['price_per_kg'].agg(
        min_price='min',
        max_price='max',
        avg_price='mean',
        std_dev='std'
    ).reset_index()
    daily_stats['spread'] = daily_stats['max_price'] - daily_stats['min_price']
else:
    daily_stats = pd.DataFrame()

# --- 7. MAIN DASHBOARD ---
st.title(f"📊 {display_title}")
st.markdown(f"*Showing data from {cutoff_date.date()} to {datetime.today().date()}*")

if filtered_df.empty:
    st.warning("⚠️ No data available for your selection. Try choosing a different item or time period.")
    st.stop()

# --- KEY INSIGHTS (Top Metrics) ---
st.markdown("### 🎯 Key Insights at a Glance")
st.markdown("*Here's what you need to know right now:*")

curr_avg = daily_stats.iloc[-1]['avg_price']
prev_avg = daily_stats.iloc[0]['avg_price']
pct_change = ((curr_avg - prev_avg) / prev_avg) * 100 if prev_avg != 0 else 0

avg_spread = daily_stats['spread'].mean()
volatility = filtered_df['price_per_kg'].std()

min_price_df = filtered_df[filtered_df['price_per_kg'] == filtered_df['price_per_kg'].min()]
cheapest_source = min_price_df['source'].mode()[0] if not min_price_df.empty else "N/A"

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "Current Average Price",
        f"KES {curr_avg:.0f}/kg",
        f"{pct_change:+.1f}%",
        delta_color="inverse",
        help="The average price across all markets today vs. the start of the period"
    )

with k2:
    st.metric(
        "Price Difference",
        f"KES {avg_spread:.0f}",
        help="Average difference between the highest and lowest prices. Higher = more savings possible!"
    )

with k3:
    st.metric(
        "Price Stability",
        f"{'Stable' if volatility < 20 else 'Volatile'}",
        f"±{volatility:.0f} KES",
        help="How much prices fluctuate. Stable means prices are predictable"
    )

with k4:
    st.metric(
        "Best Place to Buy",
        cheapest_source,
        help="This retailer has the lowest prices most often"
    )

st.markdown("---")

# --- TABS ---
tab1, tab2, tab3 = st.tabs([
    "📈 Price Trends",
    "🏪 Compare Retailers",
    "📊 Detailed Analysis"
])

# TAB 1: Price Trends
with tab1:
    st.markdown("### 💰 How Prices Have Changed")
    st.markdown("*This chart shows you the price movement over time and helps you spot the best time to buy.*")

    if not daily_stats.empty:
        fig = go.Figure()

        # Shaded area showing price range
        fig.add_trace(go.Scatter(
            x=daily_stats['date'], y=daily_stats['max_price'],
            mode='lines', line=dict(width=0), showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=daily_stats['date'], y=daily_stats['min_price'],
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(102, 126, 234, 0.1)',
            name='Price Range (High to Low)',
            hovertemplate='Low: KES %{y:.0f}<extra></extra>'
        ))

        # Average price line
        fig.add_trace(go.Scatter(
            x=daily_stats['date'], y=daily_stats['avg_price'],
            mode='lines+markers',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6),
            name='Average Price',
            hovertemplate='Average: KES %{y:.0f}<extra></extra>'
        ))

        # Trend line
        if show_trendline and len(daily_stats) > 1:
            x_nums = (daily_stats['date'] - daily_stats['date'].min()).dt.days
            coef = np.polyfit(x_nums, daily_stats['avg_price'], 1)
            poly1d_fn = np.poly1d(coef)

            trend_direction = "↗️ Rising" if coef[0] > 0 else "↘️ Falling"

            fig.add_trace(go.Scatter(
                x=daily_stats['date'], y=poly1d_fn(x_nums),
                mode='lines',
                line=dict(color='#ff6b6b', width=2, dash='dash'),
                name=f'Overall Trend ({trend_direction})',
                hovertemplate='Trend: KES %{y:.0f}<extra></extra>'
            ))

        fig.update_layout(
            height=500,
            template="plotly_white",
            hovermode="x unified",
            yaxis_title="Price (KES per Kg)",
            xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Insights
        col1, col2 = st.columns(2)
        with col1:
            if pct_change > 5:
                st.info(f"📈 **Prices are UP {abs(pct_change):.1f}%** compared to {time_period} days ago. Consider waiting if possible.")
            elif pct_change < -5:
                st.success(f"📉 **Prices are DOWN {abs(pct_change):.1f}%** compared to {time_period} days ago. Good time to buy!")
            else:
                st.info(f"➡️ **Prices are STABLE** (changed only {abs(pct_change):.1f}%). Consistent pricing.")

        with col2:
            lowest_price = daily_stats['min_price'].min()
            highest_price = daily_stats['max_price'].max()
            st.metric("Lowest Price Seen", f"KES {lowest_price:.0f}/kg")
            st.metric("Highest Price Seen", f"KES {highest_price:.0f}/kg")

# TAB 2: Retailer Comparison
with tab2:
    st.markdown("### 🏪 Which Retailer Offers the Best Deal?")
    st.markdown("*Compare prices across different markets and stores to find where you should shop.*")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Price Comparison Over Time")
        pivot_df = filtered_df.groupby(['date', 'source'])['price_per_kg'].mean().reset_index()
        fig2 = px.line(
            pivot_df, x='date', y='price_per_kg', color='source',
            color_discrete_sequence=px.colors.qualitative.Bold,
            template="plotly_white"
        )
        fig2.update_layout(
            height=400,
            hovermode="x unified",
            yaxis_title="Price (KES/Kg)",
            xaxis_title="Date",
            legend_title="Retailer"
        )
        fig2.update_traces(hovertemplate='%{y:.0f} KES<extra></extra>')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("#### Who Has the Best Prices?")
        daily_mins = filtered_df.groupby('date')['price_per_kg'].min().reset_index()
        winners = pd.merge(filtered_df, daily_mins, on=['date', 'price_per_kg'])
        win_counts = winners['source'].value_counts().reset_index()
        win_counts.columns = ['Retailer', 'Days with Lowest Price']

        fig_pie = px.pie(
            win_counts,
            values='Days with Lowest Price',
            names='Retailer',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pie.update_layout(height=400, showlegend=True)
        fig_pie.update_traces(textinfo='percent+label', hovertemplate='%{label}: %{value} days<extra></extra>')
        st.plotly_chart(fig_pie, use_container_width=True)

    # Summary table
    st.markdown("#### 📋 Price Summary by Retailer")
    st.markdown("*Average, lowest, and highest prices from each retailer:*")

    retailer_stats = filtered_df.groupby('source')['price_per_kg'].agg([
        ('Average Price', 'mean'),
        ('Lowest Price', 'min'),
        ('Highest Price', 'max'),
        ('Sample Size', 'count')
    ]).round(2).sort_values('Average Price')

    st.dataframe(
        retailer_stats.style.format({
            'Average Price': 'KES {:.0f}',
            'Lowest Price': 'KES {:.0f}',
            'Highest Price': 'KES {:.0f}',
            'Sample Size': '{:.0f}'
        }).background_gradient(subset=['Average Price'], cmap='RdYlGn_r'),
        use_container_width=True
    )

# TAB 3: Detailed Analysis
with tab3:
    st.markdown("### 📊 Deep Dive into the Data")
    st.markdown("*For those who want more detailed insights and statistical information.*")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Weekend vs Weekday Prices")

        if 'is_weekend' in filtered_df.columns:
            filtered_df['Day Type'] = filtered_df['is_weekend'].apply(
                lambda x: 'Weekend' if x in [True, 1, 'True'] else 'Weekday'
            )

            fig3 = px.box(
                filtered_df, x="Day Type", y="price_per_kg", color="Day Type",
                points="outliers",
                color_discrete_map={'Weekend': '#ff6b6b', 'Weekday': '#4ecdc4'},
                template="plotly_white"
            )
            fig3.update_layout(showlegend=False, yaxis_title="Price (KES/Kg)")
            fig3.update_traces(hovertemplate='%{y:.0f} KES<extra></extra>')
            st.plotly_chart(fig3, use_container_width=True)

            wknd_df = filtered_df[filtered_df['Day Type'] == 'Weekend']
            wkday_df = filtered_df[filtered_df['Day Type'] == 'Weekday']

            if not wknd_df.empty and not wkday_df.empty:
                wknd_mean = wknd_df['price_per_kg'].mean()
                wkday_mean = wkday_df['price_per_kg'].mean()
                premium = ((wknd_mean - wkday_mean) / wkday_mean) * 100

                if abs(premium) > 2:
                    if premium > 0:
                        st.warning(f"⚠️ Weekend prices are **{premium:.1f}% higher** on average. Shop on weekdays to save!")
                    else:
                        st.success(f"💰 Weekend prices are **{abs(premium):.1f}% lower** on average. Weekend shopping might save you money!")
                else:
                    st.info(f"➡️ Prices are similar on weekends and weekdays (difference: {abs(premium):.1f}%)")
        else:
            st.info("Weekend data not available in this dataset.")

    with col_b:
        st.markdown("#### Price Distribution by Retailer")
        fig4 = px.violin(
            filtered_df, x="source", y="price_per_kg", color="source",
            box=True, points=False,
            template="plotly_white"
        )
        fig4.update_layout(showlegend=False, xaxis_title="Retailer", yaxis_title="Price (KES/Kg)")
        fig4.update_traces(hovertemplate='%{y:.0f} KES<extra></extra>')
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("*The wider the shape, the more variable that retailer's prices are.*")

    # Full statistics
    st.markdown("#### 📈 Complete Statistical Summary")
    desc_stats = filtered_df.groupby('source')['price_per_kg'].describe()[
        ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    ].round(2)
    desc_stats.columns = ['Samples', 'Average', 'Std Dev', 'Min', '25th %', 'Median', '75th %', 'Max']

    st.dataframe(
        desc_stats.style.format('{:.2f}').background_gradient(subset=['Average'], cmap='RdYlGn_r'),
        use_container_width=True
    )

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><b>💡 Tips for Using This Tool:</b></p>
    <p>• Check prices regularly to spot trends and patterns<br>
    • Use the retailer comparison to plan where to shop<br>
    • Look at the weekend analysis to time your shopping trips<br>
    • Download the data to keep your own records</p>
</div>
""", unsafe_allow_html=True)
