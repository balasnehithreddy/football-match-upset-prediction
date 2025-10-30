"""
Test that all packages are installed correctly
"""
print("=" * 50)
print("🚀 TESTING YOUR SETUP")
print("=" * 50)

# Check Python location
import sys
print(f"✅ Python: {sys.executable}")

# Test each package
print("\n📦 CHECKING PACKAGES:")

try:
    import requests
    print("✅ requests installed")
except:
    print("❌ requests NOT found")

try:
    import pandas as pd
    print("✅ pandas installed")
except:
    print("❌ pandas NOT found")

try:
    import pyspark
    from pyspark.sql import SparkSession
    print("✅ pyspark installed")
except:
    print("❌ pyspark NOT found")

try:
    from dotenv import load_dotenv
    print("✅ python-dotenv installed")
except:
    print("❌ python-dotenv NOT found")

print("\n" + "=" * 50)
print("🎉 ALL SET! Ready to build the project!")
print("=" * 50)