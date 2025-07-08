import pandas as pd
import sqlite3
import json
import os
from pathlib import Path
from glob import glob
import sys

sys.path.append('src')
import data_extractor

def create_database():
    conn = sqlite3.connect('database/phonepe.db')
    c = conn.cursor()
    # Aggregated tables
    c.execute('''CREATE TABLE IF NOT EXISTS aggregated_transaction (
        state TEXT, year INTEGER, quarter INTEGER, transaction_type TEXT, transaction_count INTEGER, transaction_amount REAL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS aggregated_user (
        state TEXT, year INTEGER, quarter INTEGER, registered_users INTEGER, app_opens INTEGER
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS aggregated_insurance (
        state TEXT, year INTEGER, quarter INTEGER, insurance_count INTEGER, insurance_amount REAL
    )''')
    # Map tables
    c.execute('''CREATE TABLE IF NOT EXISTS map_transaction (
        state TEXT, year INTEGER, quarter INTEGER, district TEXT, transaction_count INTEGER, transaction_amount REAL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS map_user (
        state TEXT, year INTEGER, quarter INTEGER, district TEXT, registered_users INTEGER, app_opens INTEGER
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS map_insurance (
        state TEXT, year INTEGER, quarter INTEGER, district TEXT, insurance_count INTEGER, insurance_amount REAL
    )''')
    # Top tables
    c.execute('''CREATE TABLE IF NOT EXISTS top_transaction (
        state TEXT, year INTEGER, quarter INTEGER, pincode TEXT, transaction_count INTEGER, transaction_amount REAL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS top_user (
        state TEXT, year INTEGER, quarter INTEGER, pincode TEXT, registered_users INTEGER
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS top_insurance (
        state TEXT, year INTEGER, quarter INTEGER, pincode TEXT, insurance_count INTEGER, insurance_amount REAL
    )''')
    conn.commit()
    conn.close()

def extract_aggregated_transaction():
    records = []
    base = Path('pulse/data/aggregated/transaction/country/india/state')
    for state_dir in base.iterdir():
        if state_dir.is_dir():
            state = state_dir.name
            for year_dir in state_dir.iterdir():
                year = int(year_dir.name)
                for file in year_dir.glob('*.json'):
                    quarter = int(file.stem)
                    with open(file, 'r') as f:
                        data = json.load(f)
                    for t in data.get('data', {}).get('transactionData', []):
                        records.append({
                            'state': state,
                            'year': year,
                            'quarter': quarter,
                            'transaction_type': t.get('name'),
                            'transaction_count': t.get('paymentInstruments', [{}])[0].get('count', 0),
                            'transaction_amount': t.get('paymentInstruments', [{}])[0].get('amount', 0.0)
                        })
    df = pd.DataFrame(records)
    return df

def extract_aggregated_user():
    records = []
    base = Path('pulse/data/aggregated/user/country/india/state')
    for state_dir in base.iterdir():
        if state_dir.is_dir():
            state = state_dir.name
            for year_dir in state_dir.iterdir():
                year = int(year_dir.name)
                for file in year_dir.glob('*.json'):
                    quarter = int(file.stem)
                    with open(file, 'r') as f:
                        data = json.load(f)
                    summary = data.get('data', {}).get('aggregated', {})
                    records.append({
                        'state': state,
                        'year': year,
                        'quarter': quarter,
                        'registered_users': summary.get('registeredUsers', 0),
                        'app_opens': summary.get('appOpens', 0)
                    })
    df = pd.DataFrame(records)
    return df

def extract_aggregated_insurance():
    records = []
    base = Path('pulse/data/aggregated/insurance/country/india/state')
    for state_dir in base.iterdir():
        if state_dir.is_dir():
            state = state_dir.name
            for year_dir in state_dir.iterdir():
                year = int(year_dir.name)
                for file in year_dir.glob('*.json'):
                    quarter = int(file.stem)
                    with open(file, 'r') as f:
                        data = json.load(f)
                    for t in data.get('data', {}).get('transactionData', []):
                        if t.get('name') == 'Insurance':
                            pi = t.get('paymentInstruments', [{}])[0]
                            record = {
                                'state': state,
                                'year': year,
                                'quarter': quarter,
                                'insurance_count': pi.get('count', 0),
                                'insurance_amount': pi.get('amount', 0.0)
                            }
                            records.append(record)
                            print("Extracted insurance record:", record)
    print(f"Total insurance records extracted: {len(records)}")
    df = pd.DataFrame(records)
    return df

def extract_map_transaction():
    records = []
    base = Path('pulse/data/map/transaction/hover/country/india/state')
    for state_dir in base.iterdir():
        if state_dir.is_dir():
            state = state_dir.name
            for year_dir in state_dir.iterdir():
                year = int(year_dir.name)
                for file in year_dir.glob('*.json'):
                    quarter = int(file.stem)
                    with open(file, 'r') as f:
                        data = json.load(f)
                    for d in data.get('data', {}).get('hoverDataList', []):
                        records.append({
                            'state': state,
                            'year': year,
                            'quarter': quarter,
                            'district': d.get('name'),
                            'transaction_count': d.get('metric', [{}])[0].get('count', 0),
                            'transaction_amount': d.get('metric', [{}])[0].get('amount', 0.0)
                        })
    df = pd.DataFrame(records)
    return df

def extract_map_user():
    records = []
    base = Path('pulse/data/map/user/hover/country/india/state')
    for state_dir in base.iterdir():
        if state_dir.is_dir():
            state = state_dir.name
            for year_dir in state_dir.iterdir():
                year = int(year_dir.name)
                for file in year_dir.glob('*.json'):
                    quarter = int(file.stem)
                    with open(file, 'r') as f:
                        data = json.load(f)
                    for d in data.get('data', {}).get('hoverDataList', []):
                        records.append({
                            'state': state,
                            'year': year,
                            'quarter': quarter,
                            'district': d.get('name'),
                            'registered_users': d.get('metric', [{}])[0].get('count', 0),
                            'app_opens': d.get('metric', [{}])[0].get('appOpens', 0)
                        })
    df = pd.DataFrame(records)
    return df

def extract_map_insurance():
    records = []
    base = Path('pulse/data/map/insurance/hover/country/india/state')
    for state_dir in base.iterdir():
        if state_dir.is_dir():
            state = state_dir.name
            for year_dir in state_dir.iterdir():
                year = int(year_dir.name)
                for file in year_dir.glob('*.json'):
                    quarter = int(file.stem)
                    with open(file, 'r') as f:
                        data = json.load(f)
                    for d in data.get('data', {}).get('hoverDataList', []):
                        records.append({
                            'state': state,
                            'year': year,
                            'quarter': quarter,
                            'district': d.get('name'),
                            'insurance_count': d.get('metric', [{}])[0].get('count', 0),
                            'insurance_amount': d.get('metric', [{}])[0].get('amount', 0.0)
                        })
    df = pd.DataFrame(records)
    return df

def extract_top_transaction():
    records = []
    base = Path('pulse/data/top/transaction/country/india/state')
    for state_dir in base.iterdir():
        if state_dir.is_dir():
            state = state_dir.name
            for year_dir in state_dir.iterdir():
                year = int(year_dir.name)
                for file in year_dir.glob('*.json'):
                    quarter = int(file.stem)
                    with open(file, 'r') as f:
                        data = json.load(f)
                    for t in data.get('data', {}).get('pincodes', []):
                        metric = t.get('metric', {})
                        if isinstance(metric, list):
                            metric = metric[0] if metric else {}
                        records.append({
                            'state': state,
                            'year': year,
                            'quarter': quarter,
                            'pincode': t.get('entityName'),
                            'transaction_count': metric.get('count', 0),
                            'transaction_amount': metric.get('amount', 0.0)
                        })
    df = pd.DataFrame(records)
    return df

def extract_top_user():
    records = []
    base = Path('pulse/data/top/user/country/india/state')
    for state_dir in base.iterdir():
        if state_dir.is_dir():
            state = state_dir.name
            for year_dir in state_dir.iterdir():
                year = int(year_dir.name)
                for file in year_dir.glob('*.json'):
                    quarter = int(file.stem)
                    with open(file, 'r') as f:
                        data = json.load(f)
                    for t in data.get('data', {}).get('pincodes', []):
                        metric = t.get('metric', {})
                        if isinstance(metric, list):
                            metric = metric[0] if metric else {}
                        records.append({
                            'state': state,
                            'year': year,
                            'quarter': quarter,
                            'pincode': t.get('entityName'),
                            'registered_users': metric.get('count', 0)
                        })
    df = pd.DataFrame(records)
    return df

def extract_top_insurance():
    records = []
    base = Path('pulse/data/top/insurance/country/india/state')
    for state_dir in base.iterdir():
        if state_dir.is_dir():
            state = state_dir.name
            for year_dir in state_dir.iterdir():
                year = int(year_dir.name)
                for file in year_dir.glob('*.json'):
                    quarter = int(file.stem)
                    with open(file, 'r') as f:
                        data = json.load(f)
                    for t in data.get('data', {}).get('pincodes', []):
                        metric = t.get('metric', {})
                        if isinstance(metric, list):
                            metric = metric[0] if metric else {}
                        records.append({
                            'state': state,
                            'year': year,
                            'quarter': quarter,
                            'pincode': t.get('entityName'),
                            'insurance_count': metric.get('count', 0),
                            'insurance_amount': metric.get('amount', 0.0)
                        })
    df = pd.DataFrame(records)
    return df

def load_all_data():
    create_database()
    conn = sqlite3.connect('database/phonepe.db')
    # Aggregated
    print('Extracting aggregated_transaction...')
    df = extract_aggregated_transaction()
    if not df.empty:
        df.to_sql('aggregated_transaction', conn, if_exists='replace', index=False)
    print('Extracting aggregated_user...')
    df = extract_aggregated_user()
    if not df.empty:
        df.to_sql('aggregated_user', conn, if_exists='replace', index=False)
    print('Extracting aggregated_insurance...')
    df = extract_aggregated_insurance()
    if not df.empty:
        df.to_sql('aggregated_insurance', conn, if_exists='replace', index=False)
    # Map
    print('Extracting map_transaction...')
    df = extract_map_transaction()
    if not df.empty:
        df.to_sql('map_transaction', conn, if_exists='replace', index=False)
    print('Extracting map_user...')
    df = extract_map_user()
    if not df.empty:
        df.to_sql('map_user', conn, if_exists='replace', index=False)
    print('Extracting map_insurance...')
    df = extract_map_insurance()
    if not df.empty:
        df.to_sql('map_insurance', conn, if_exists='replace', index=False)
    # Top
    print('Extracting top_transaction...')
    df = extract_top_transaction()
    if not df.empty:
        df.to_sql('top_transaction', conn, if_exists='replace', index=False)
    print('Extracting top_user...')
    df = extract_top_user()
    if not df.empty:
        df.to_sql('top_user', conn, if_exists='replace', index=False)
    print('Extracting top_insurance...')
    df = extract_top_insurance()
    if not df.empty:
        df.to_sql('top_insurance', conn, if_exists='replace', index=False)
    conn.close()
    print('✅ All data loaded into database!')

def run_data_extraction():
    """Run data extraction and database creation"""
    print("🔄 Starting data extraction from pulse directory...")
    os.makedirs("database", exist_ok=True)
    sys.path.append('src')
    import data_extractor
    data_extractor.load_all_data()
    print("✅ Data extraction completed!")

if __name__ == "__main__":
    run_data_extraction() 