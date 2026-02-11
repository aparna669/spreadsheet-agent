"""
Test script for AI Spreadsheet Agent
Run this after starting the backend server to test functionality
"""

import requests
import pandas as pd
import json

API_BASE_URL = "http://localhost:8000"

def create_sample_data():
    """Create a sample CSV file for testing"""
    data = {
        'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Laptop', 
                   'Mouse', 'Keyboard', 'Monitor', 'Headphones', 'Webcam'],
        'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 
                    'Electronics', 'Accessories', 'Accessories', 'Electronics', 
                    'Accessories', 'Electronics'],
        'Price': [999.99, 29.99, 79.99, 299.99, 1099.99, 
                 24.99, 89.99, 349.99, 149.99, 79.99],
        'Quantity': [5, 50, 30, 10, 3, 
                    45, 25, 8, 20, 15],
        'Sales': [4999.95, 1499.50, 2399.70, 2999.90, 3299.97,
                 1124.55, 2249.75, 2799.92, 2999.80, 1199.85]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('sample_sales.csv', index=False)
    print("✅ Created sample_sales.csv")
    return 'sample_sales.csv'


def test_upload(filename):
    """Test file upload"""
    print("\n📤 Testing file upload...")
    
    with open(filename, 'rb') as f:
        files = {'file': (filename, f, 'text/csv')}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Upload successful! Session ID: {result['session_id']}")
        print(f"   Rows: {result['info']['rows']}, Columns: {result['info']['columns']}")
        return result['session_id']
    else:
        print(f"❌ Upload failed: {response.text}")
        return None


def test_query(session_id):
    """Test natural language query"""
    print("\n💬 Testing natural language query...")
    
    queries = [
        "What are the top 3 products by sales?",
        "Calculate the total revenue",
        "What is the average price by category?"
    ]
    
    for query in queries:
        print(f"\n   Query: {query}")
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"session_id": session_id, "query": query}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Response: {result['result']['response']}")
        else:
            print(f"   ❌ Query failed: {response.text}")


def test_clean(session_id):
    """Test data cleaning"""
    print("\n🧹 Testing data cleaning...")
    
    response = requests.post(
        f"{API_BASE_URL}/clean",
        json={"session_id": session_id}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Cleaning successful!")
        print(f"   Steps performed: {len(result['cleaning_steps'])}")
        for step in result['cleaning_steps']:
            print(f"   - {step}")
    else:
        print(f"❌ Cleaning failed: {response.text}")


def test_formula(session_id):
    """Test formula suggestion"""
    print("\n📐 Testing formula suggestion...")
    
    description = "Calculate total revenue (Price × Quantity) for each product"
    
    response = requests.post(
        f"{API_BASE_URL}/suggest-formula",
        json={
            "session_id": session_id,
            "description": description
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Formula suggestion successful!")
        formula_result = result['result']
        print(f"   Formula: {formula_result.get('formula', 'N/A')}")
        print(f"   Explanation: {formula_result.get('explanation', 'N/A')[:100]}...")
    else:
        print(f"❌ Formula suggestion failed: {response.text}")


def test_chart(session_id):
    """Test chart generation"""
    print("\n📊 Testing chart generation...")
    
    response = requests.post(
        f"{API_BASE_URL}/generate-chart",
        json={
            "session_id": session_id,
            "chart_type": "bar",
            "description": "Show total sales by category"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Chart generation successful!")
        print(f"   Chart type: {result['chart_type']}")
        print(f"   Configuration: {result['config']}")
        print(f"   Image data length: {len(result['image'])} characters")
    else:
        print(f"❌ Chart generation failed: {response.text}")


def test_preview(session_id):
    """Test data preview"""
    print("\n👁️ Testing data preview...")
    
    response = requests.get(f"{API_BASE_URL}/preview/{session_id}?rows=5")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Preview successful!")
        print(f"   Showing {len(result['preview']['head'])} rows")
        print(f"   Total rows: {result['preview']['total_rows']}")
    else:
        print(f"❌ Preview failed: {response.text}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("AI Spreadsheet Agent - Test Suite")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code != 200:
            print("❌ Backend server is not running!")
            print("   Please start the server with: python backend/main.py")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend server!")
        print("   Please start the server with: python backend/main.py")
        return
    
    print("✅ Backend server is running\n")
    
    # Create sample data
    filename = create_sample_data()
    
    # Test upload
    session_id = test_upload(filename)
    if not session_id:
        print("\n❌ Cannot proceed without valid session")
        return
    
    # Run all tests
    test_query(session_id)
    test_clean(session_id)
    test_formula(session_id)
    test_chart(session_id)
    test_preview(session_id)
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()