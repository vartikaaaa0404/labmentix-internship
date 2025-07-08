import subprocess
import os
import sys

def check_database_exists():
    """Check if database exists"""
    return os.path.exists("database/phonepe.db")

def run_data_extraction():
    """Run data extraction and database creation"""
    print("🔄 Starting data extraction from pulse directory...")
    # Create database directory if it doesn't exist
    os.makedirs("database", exist_ok=True)
    # Run data extraction
    exec(open('src/data_extractor.py').read())
    print("✅ Data extraction completed!")

def launch_dashboard():
    """Launch Streamlit dashboard"""
    print("🚀 Launching PhonePe Insights Dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard.py"])

def main():
    print("=" * 50)
    print("📱 PhonePe Transaction Insights")
    print("=" * 50)
    if not check_database_exists():
        print("⚠️  Database not found. Running data extraction first...")
        run_data_extraction()
    print("\nChoose an option:")
    print("1. 🔄 Re-extract Data")
    print("2. 🚀 Launch Dashboard")
    print("3. 🔄 Extract Data & Launch Dashboard")
    print("4. ❌ Exit")
    choice = input("\nEnter your choice (1-4): ")
    if choice == "1":
        run_data_extraction()
    elif choice == "2":
        if check_database_exists():
            launch_dashboard()
        else:
            print("⚠️  Database not found. Please extract data first.")
    elif choice == "3":
        run_data_extraction()
        launch_dashboard()
    elif choice == "4":
        print("👋 Goodbye!")
        sys.exit(0)
    else:
        print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 