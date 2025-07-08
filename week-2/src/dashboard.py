import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sql_queries import *
import os
import sqlite3

# --- LOGIN FEATURE ---
def login():
    st.markdown(
        """
        <style>
        .stApp {background-color: #e3f0ff;}
        .login-box {
            background-color: #ffffffcc;
            border-radius: 12px;
            padding: 2rem 2rem 1.5rem 2rem;
            max-width: 350px;
            margin: 2rem auto;
            box-shadow: 0 2px 16px 0 #b3c6e0;
        }
        .login-footer {
            text-align: center;
            color: #888;
            font-size: 0.9rem;
            margin-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.image("download.png", width=120)
    st.markdown("<h2 style='text-align:center; color:#2d3a4a;'>PhonePe Transaction Insights</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; color:#4a5a6a;'>Unlock deep analytics on digital payments across India.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.info("Password hint: phonepe2024")
    password = st.text_input("Enter password", type="password")
    if st.button("Login"):
        if password == "phonepe2024":
            st.session_state["logged_in"] = True
        else:
            st.error("Incorrect password. Try again.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""<div class='login-footer'>© 2024 Vartika Rawat | Powered by Streamlit</div>""", unsafe_allow_html=True)
    st.stop()

# --- FILTERS ---
def get_filter_options():
    # Get available years, states, quarters from the database
    conn = None
    try:
        conn = sqlite3.connect('database/phonepe.db')
        years = pd.read_sql_query("SELECT DISTINCT year FROM aggregated_transaction ORDER BY year", conn)['year'].tolist()
        states = pd.read_sql_query("SELECT DISTINCT state FROM aggregated_transaction ORDER BY state", conn)['state'].tolist()
        quarters = [1, 2, 3, 4]
    except Exception:
        years, states, quarters = [], [], [1, 2, 3, 4]
    finally:
        if conn: conn.close()
    return years, states, quarters

def main():
    st.set_page_config(page_title="PhonePe Transaction Insights", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {background-color: #e3f0ff;}
        .css-1d391kg {background: #f7fafd;}
        .phonepe-bg {
            background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1200&q=80');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        </style>
        <div class='phonepe-bg'></div>
        """,
        unsafe_allow_html=True
    )
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        login()
    st.sidebar.title("PhonePe Insights")
    years, states, quarters = get_filter_options()
    # Sidebar filters
    selected_year = st.sidebar.selectbox("Year", options=["All"] + years, help="Filter by year")
    selected_state = st.sidebar.selectbox("State", options=["All"] + states, help="Filter by state")
    selected_quarter = st.sidebar.selectbox("Quarter", options=["All"] + quarters, help="Filter by quarter")
    page = st.sidebar.selectbox("Choose Analysis", [
        "📊 Overview",
        "🗺️ State Analysis", 
        "💳 Transaction Types",
        "📈 Time Trends",
        "🎯 Top Performers",
        "👥 User Analytics",
        "🛡️ Insurance Insights"
    ])
    filters = {"year": selected_year, "state": selected_state, "quarter": selected_quarter}
    if page == "📊 Overview":
        show_overview(filters)
    elif page == "🗺️ State Analysis":
        show_state_analysis(filters)
    elif page == "💳 Transaction Types":
        show_transaction_types(filters)
    elif page == "📈 Time Trends":
        show_time_trends(filters)
    elif page == "🎯 Top Performers":
        show_top_performers(filters)
    elif page == "👥 User Analytics":
        show_user_analytics(filters)
    elif page == "🛡️ Insurance Insights":
        show_insurance_insights(filters)

def apply_filters(df, filters, cols):
    if df.empty:
        return df
    if filters["year"] != "All" and "year" in cols:
        df = df[df["year"] == filters["year"]]
    if filters["state"] != "All" and "state" in cols:
        df = df[df["state"] == filters["state"]]
    if filters["quarter"] != "All" and "quarter" in cols:
        df = df[df["quarter"] == filters["quarter"]]
    return df

def show_overview(filters):
    st.title("📊 PhonePe Transaction Overview")
    top_states = get_top_states_by_transaction()
    yearly = get_yearly_growth()
    top_states = apply_filters(top_states, filters, ["state"])
    yearly = apply_filters(yearly, filters, ["year"])
    total_txn = int(yearly['total_amount'].sum()) if not yearly.empty else 0
    total_states = len(top_states) if not top_states.empty else 0
    col1, col2 = st.columns(2)
    col1.metric("Total Transaction Amount", f"₹{total_txn:,.0f}", help="Sum of all transactions in the selected period.")
    col2.metric("States Covered", total_states, help="Number of states with transactions.")
    if not yearly.empty:
        fig = px.line(yearly, x='year', y='total_amount', markers=True, title="Yearly Transaction Growth")
        st.plotly_chart(fig, use_container_width=True)
    if not top_states.empty:
        st.subheader("Top 10 States by Transaction Amount")
        st.dataframe(top_states)
        st.download_button("Download CSV", top_states.to_csv(index=False), file_name="top_states.csv")

def show_state_analysis(filters):
    st.title("🗺️ State-wise Analysis")
    df = get_top_states_by_transaction()
    df = apply_filters(df, filters, ["state"])
    if not df.empty:
        fig = px.bar(df, x='state', y='total_amount', title="Top States by Transaction Amount", color='total_amount')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), file_name="state_analysis.csv")
    comp = get_comparative_analysis()
    comp = apply_filters(comp, filters, ["state"])
    if not comp.empty:
        st.subheader("State Comparison: Transactions vs Users")
        fig2 = px.scatter(comp, x='total_users', y='total_amount', text='state',
                          labels={'total_users': 'Registered Users', 'total_amount': 'Transaction Amount'},
                          title="State Comparison")
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(comp)
        st.download_button("Download CSV", comp.to_csv(index=False), file_name="state_comparison.csv")

def show_transaction_types(filters):
    st.title("💳 Transaction Type Analysis")
    df = get_transaction_type_distribution()
    df = apply_filters(df, filters, ["transaction_type"])
    if not df.empty:
        fig = px.pie(df, names='transaction_type', values='total_amount', title="Transaction Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), file_name="transaction_types.csv")

def show_time_trends(filters):
    st.title("📈 Time Trends Analysis")
    df = get_quarterly_trends()
    df = apply_filters(df, filters, ["year", "quarter"])
    if not df.empty:
        df['period'] = df['year'].astype(str) + ' Q' + df['quarter'].astype(str)
        fig = px.line(df, x='period', y='total_amount', markers=True, title="Quarterly Transaction Trends")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), file_name="quarterly_trends.csv")

def show_top_performers(filters):
    st.title("🎯 Top Performers")
    st.subheader("Top 20 Districts by Transaction Amount")
    districts = get_top_districts()
    districts = apply_filters(districts, filters, ["district"])
    if not districts.empty:
        fig = px.bar(districts, x='district', y='total_amount', title="Top Districts", color='total_amount')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(districts)
        st.download_button("Download CSV", districts.to_csv(index=False), file_name="top_districts.csv")
    st.subheader("Top 20 Pincodes by Transaction Amount")
    pincodes = get_top_pincodes()
    pincodes = apply_filters(pincodes, filters, ["pincode"])
    if not pincodes.empty:
        fig2 = px.bar(pincodes, x='pincode', y='total_amount', title="Top Pincodes", color='total_amount')
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(pincodes)
        st.download_button("Download CSV", pincodes.to_csv(index=False), file_name="top_pincodes.csv")

def show_user_analytics(filters):
    st.title("👥 User Analytics")
    users = get_state_wise_users()
    users = apply_filters(users, filters, ["state"])
    if not users.empty:
        fig = px.bar(users, x='state', y='total_users', title="State-wise Registered Users", color='total_users')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(users)
        st.download_button("Download CSV", users.to_csv(index=False), file_name="user_analytics.csv")

def show_insurance_insights(filters):
    st.title("🛡️ Insurance Insights")
    ins = get_insurance_analysis()
    ins = apply_filters(ins, filters, ["year", "quarter"])
    if ins.empty:
        st.warning("No insurance data found for the selected filters.")
        st.write(ins)  # Debug: show the DataFrame
    else:
        ins['period'] = ins['year'].astype(str) + ' Q' + ins['quarter'].astype(str)
        fig = px.line(ins, x='period', y='total_insurance', markers=True, title="Insurance Transaction Trends")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ins)
        st.download_button("Download CSV", ins.to_csv(index=False), file_name="insurance_insights.csv")

def run_data_extraction():
    print("🔄 Starting data extraction from pulse directory...")
    os.makedirs("database", exist_ok=True)
    import sys
    sys.path.append('src')
    import data_extractor
    data_extractor.load_all_data()
    print("✅ Data extraction completed!")

if __name__ == "__main__":
    main() 