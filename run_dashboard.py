#!/usr/bin/env python3
"""
Launcher script for the Home Credit Risk Analysis Dashboard
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard"""
    
    print("🏦 Home Credit Risk Analysis Dashboard")
    print("=" * 50)
    print("Starting the interactive dashboard...")
    print("This will open in your default web browser.")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "streamlit_requirements.txt"])
        print("✅ Requirements installed")
    
    # Launch the dashboard
    try:
        print("\n🚀 Launching dashboard...")
        print("📊 Dashboard will be available at: http://localhost:8501")
        print("💡 Press Ctrl+C to stop the dashboard")
        print("-" * 50)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("💡 Try running: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main() 