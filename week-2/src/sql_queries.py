import sqlite3
import pandas as pd

def get_top_states_by_transaction():
    conn = sqlite3.connect('database/phonepe.db')
    query = '''
        SELECT state, SUM(transaction_amount) as total_amount
        FROM aggregated_transaction
        GROUP BY state
        ORDER BY total_amount DESC
        LIMIT 10
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_yearly_growth():
    conn = sqlite3.connect('database/phonepe.db')
    query = '''
        SELECT year, SUM(transaction_amount) as total_amount
        FROM aggregated_transaction
        GROUP BY year
        ORDER BY year
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_transaction_type_distribution():
    conn = sqlite3.connect('database/phonepe.db')
    query = '''
        SELECT transaction_type, SUM(transaction_amount) as total_amount
        FROM aggregated_transaction
        GROUP BY transaction_type
        ORDER BY total_amount DESC
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_quarterly_trends():
    conn = sqlite3.connect('database/phonepe.db')
    query = '''
        SELECT year, quarter, SUM(transaction_amount) as total_amount
        FROM aggregated_transaction
        GROUP BY year, quarter
        ORDER BY year, quarter
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_top_districts():
    conn = sqlite3.connect('database/phonepe.db')
    query = '''
        SELECT district, SUM(transaction_amount) as total_amount
        FROM map_transaction
        GROUP BY district
        ORDER BY total_amount DESC
        LIMIT 20
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_state_wise_users():
    conn = sqlite3.connect('database/phonepe.db')
    query = '''
        SELECT state, SUM(registered_users) as total_users
        FROM aggregated_user
        GROUP BY state
        ORDER BY total_users DESC
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_top_pincodes():
    conn = sqlite3.connect('database/phonepe.db')
    query = '''
        SELECT pincode, SUM(transaction_amount) as total_amount
        FROM top_transaction
        GROUP BY pincode
        ORDER BY total_amount DESC
        LIMIT 20
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_insurance_analysis():
    conn = sqlite3.connect('database/phonepe.db')
    query = '''
        SELECT year, quarter, SUM(insurance_amount) as total_insurance
        FROM aggregated_insurance
        GROUP BY year, quarter
        ORDER BY year, quarter
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_monthly_trends():
    # If monthly data is not available, fallback to quarterly trends
    return get_quarterly_trends()

def get_comparative_analysis():
    conn = sqlite3.connect('database/phonepe.db')
    query = '''
        SELECT a.state, SUM(a.transaction_amount) as total_amount, SUM(u.registered_users) as total_users
        FROM aggregated_transaction a
        LEFT JOIN aggregated_user u ON a.state = u.state AND a.year = u.year AND a.quarter = u.quarter
        GROUP BY a.state
        ORDER BY total_amount DESC
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df 