# 📱 PhonePe Transaction Insights

## 📋 Project Overview
Analysis and visualization of PhonePe transaction data using SQL and Streamlit.

## 🚀 Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run the project: `python main.py`
3. Choose option 3 to extract data and launch dashboard
4. **Login password:** `phonepe2024`

## 📊 Features
- State-wise transaction analysis
- Time trend analysis
- Transaction type distribution
- Top performers identification
- User analytics
- Insurance insights
- **Login-protected dashboard**
- **Sidebar filters** (year, state, quarter)
- **Download CSV** for all tables

## 🗂️ Project Structure
- `src/data_extractor.py` - Data extraction from JSON files
- `src/sql_queries.py` - SQL query functions
- `src/dashboard.py` - Streamlit dashboard
- `database/phonepe.db` - SQLite database
- `main.py` - Main execution script

## 📈 Dashboard Pages
1. **Overview** - Key metrics and summary
2. **State Analysis** - State-wise performance
3. **Transaction Types** - Category analysis
4. **Time Trends** - Quarterly/yearly trends
5. **Top Performers** - Best performing regions
6. **User Analytics** - User registration insights
7. **Insurance Insights** - Insurance transaction analysis

## 🖼️ Example Screenshots
- **Login Page:**
  ![Login Screenshot](screenshots/login.png)
- **Main Dashboard:**
  ![Dashboard Screenshot](screenshots/dashboard.png)
- **Filters and Download:**
  ![Filters Screenshot](screenshots/filters.png)

> _Replace the above image links with your own screenshots after running the app._

## 🔧 Technical Details
- **Database**: SQLite with 9 tables
- **Frontend**: Streamlit dashboard
- **Visualization**: Plotly charts
- **Data Processing**: Pandas

## 📊 Key Insights Generated
- Top performing states and districts
- Transaction growth trends
- Popular payment categories
- Geographic distribution patterns
- User engagement metrics

## ❓ FAQ
**Q: What is the dashboard login password?**  
A: `phonepe2024`

**Q: How do I filter by year, state, or quarter?**  
A: Use the sidebar filters on the left of the dashboard.

**Q: How do I download the data?**  
A: Click the "Download CSV" button below any table.

**Q: How do I update the data?**  
A: Run `python main.py` and choose option 1 or 3 to re-extract data.

## ⚠️ Known Issues
- If the database is empty or missing, you must extract data before launching the dashboard.
- Some filters may return empty tables if no data is available for the selection.
- The login feature is for demonstration only and not secure for production use.

## 📝 Contributing
Pull requests welcome! Please follow PEP 8 and add docstrings/type hints.

## 📄 License
MIT License 