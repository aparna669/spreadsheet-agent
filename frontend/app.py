import streamlit as st
import requests
import pandas as pd
import json
from typing import Optional
import base64
from io import BytesIO

# Configuration
API_BASE_URL = "http://localhost:8001"

# Page config
st.set_page_config(
    page_title="AI Spreadsheet Agent",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'file_info' not in st.session_state:
    st.session_state.file_info = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def upload_file(file):
    """Upload file to backend"""
    files = {"file": (file.name, file, file.type)}
    response = requests.post(f"{API_BASE_URL}/upload", files=files)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Upload failed: {response.text}")
        return None


def query_data(session_id: str, query: str):
    """Send natural language query"""
    response = requests.post(
        f"{API_BASE_URL}/query",
        json={"session_id": session_id, "query": query}
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Query failed: {response.text}")
        return None


def clean_data(session_id: str, instructions: Optional[str] = None):
    """Clean data"""
    response = requests.post(
        f"{API_BASE_URL}/clean",
        json={"session_id": session_id, "instructions": instructions}
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Cleaning failed: {response.text}")
        return None


def suggest_formula(session_id: str, description: str, context: Optional[str] = None):
    """Get formula suggestions"""
    response = requests.post(
        f"{API_BASE_URL}/suggest-formula",
        json={
            "session_id": session_id,
            "description": description,
            "context": context
        }
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Formula suggestion failed: {response.text}")
        return None


def generate_chart(session_id: str, chart_type: str, description: str):
    """Generate chart"""
    response = requests.post(
        f"{API_BASE_URL}/generate-chart",
        json={
            "session_id": session_id,
            "chart_type": chart_type,
            "description": description
        }
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Chart generation failed: {response.text}")
        return None


def get_preview(session_id: str, rows: int = 10):
    """Get data preview"""
    response = requests.get(f"{API_BASE_URL}/preview/{session_id}?rows={rows}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Preview failed: {response.text}")
        return None


# Main UI
st.title("🤖 AI Spreadsheet Agent")
st.markdown("Upload your spreadsheet and interact with it using natural language!")

# Sidebar
with st.sidebar:
    st.header("📁 File Upload")
    
    uploaded_file = st.file_uploader(
        "Upload spreadsheet",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file"
    )
    
    if uploaded_file is not None and st.session_state.session_id is None:
        with st.spinner("Uploading file..."):
            result = upload_file(uploaded_file)
            if result:
                st.session_state.session_id = result['session_id']
                st.session_state.file_info = result['info']
                st.success("File uploaded successfully!")
    
    if st.session_state.file_info:
        st.markdown("---")
        st.subheader("📊 Dataset Info")
        info = st.session_state.file_info
        st.metric("Rows", info['rows'])
        st.metric("Columns", info['columns'])
        
        with st.expander("Column Details"):
            for col, dtype in info['dtypes'].items():
                st.text(f"{col}: {dtype}")
        
        with st.expander("Missing Values"):
            missing = info['missing_values']
            if any(missing.values()):
                for col, count in missing.items():
                    if count > 0:
                        st.text(f"{col}: {count}")
            else:
                st.text("No missing values")
        
        if st.button("🗑️ Clear Session"):
            st.session_state.session_id = None
            st.session_state.file_info = None
            st.session_state.chat_history = []
            st.rerun()

# Main content
if st.session_state.session_id:
    
    # Tabs for different features
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Chat Query",
        "🧹 Data Cleaning",
        "📐 Formula Helper",
        "📈 Chart Generator",
        "👁️ Data Preview"
    ])
    
    # Tab 1: Natural Language Query
    with tab1:
        st.header("Ask Questions About Your Data")
        st.markdown("Use natural language to query, analyze, and manipulate your spreadsheet data.")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "data" in message and message["data"]:
                    if isinstance(message["data"], pd.DataFrame):
                        st.dataframe(message["data"])
        
        # Query input
        query = st.chat_input("Ask a question about your data...")
        
        if query:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": query
            })
            
            with st.chat_message("user"):
                st.markdown(query)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = query_data(st.session_state.session_id, query)
                    if result and result.get('success'):
                        response_text = result['result']['response']
                        st.markdown(response_text)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response_text,
                            "data": None
                        })
                    else:
                        error_msg = "Sorry, I couldn't process that query."
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg,
                            "data": None
                        })
    
    # Tab 2: Data Cleaning
    with tab2:
        st.header("Automated Data Cleaning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Automatic Cleaning")
            st.markdown("""
            Automatically clean your data by:
            - Removing duplicate rows
            - Handling missing values
            - Stripping whitespace
            - Standardizing formats
            """)
            
            if st.button("🧹 Auto Clean Data", type="primary"):
                with st.spinner("Cleaning data..."):
                    result = clean_data(st.session_state.session_id)
                    if result and result.get('success'):
                        st.success("Data cleaned successfully!")
                        
                        st.subheader("Cleaning Summary")
                        st.json(result['cleaning_steps'])
                        
                        st.metric("Rows Before", result.get('before_rows', 'N/A'))
                        st.metric("Rows After", result.get('after_rows', 'N/A'))
        
        with col2:
            st.subheader("Custom Cleaning")
            st.markdown("Provide specific instructions for data cleaning.")
            
            instructions = st.text_area(
                "Cleaning Instructions",
                placeholder="e.g., Remove rows where Age is less than 0, convert all email addresses to lowercase",
                height=100
            )
            
            if st.button("🎯 Clean with Instructions"):
                if instructions:
                    with st.spinner("Processing..."):
                        result = clean_data(st.session_state.session_id, instructions)
                        if result and result.get('success'):
                            st.success("Data cleaned with custom instructions!")
                            st.json(result['cleaning_steps'])
                else:
                    st.warning("Please provide cleaning instructions")
    
    # Tab 3: Formula Helper
    with tab3:
        st.header("Formula Suggestion Helper")
        st.markdown("Describe what you want to calculate, and get Excel/spreadsheet formula suggestions.")
        
        description = st.text_area(
            "What do you want to calculate?",
            placeholder="e.g., Calculate the total sales for each product category",
            height=100
        )
        
        context = st.text_input(
            "Additional Context (optional)",
            placeholder="e.g., Use columns A, B, and C"
        )
        
        if st.button("💡 Get Formula Suggestion", type="primary"):
            if description:
                with st.spinner("Generating formula..."):
                    result = suggest_formula(
                        st.session_state.session_id,
                        description,
                        context if context else None
                    )
                    if result and result.get('success'):
                        formula_result = result['result']
                        
                        st.success("Formula suggestion generated!")
                        
                        st.subheader("📝 Formula")
                        st.code(formula_result.get('formula', 'N/A'), language="excel")
                        
                        st.subheader("📖 Explanation")
                        st.markdown(formula_result.get('explanation', 'N/A'))
                        
                        if formula_result.get('example'):
                            st.subheader("💡 Example")
                            st.code(formula_result['example'], language="excel")
                        
                        if formula_result.get('alternatives'):
                            st.subheader("🔄 Alternatives")
                            for alt in formula_result['alternatives']:
                                st.markdown(f"- {alt}")
                        
                        if formula_result.get('notes'):
                            st.subheader("⚠️ Notes")
                            st.info(formula_result['notes'])
            else:
                st.warning("Please describe what you want to calculate")
    
    # Tab 4: Chart Generator
    with tab4:
        st.header("Chart & Visualization Generator")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            chart_type = st.selectbox(
                "Chart Type",
                ["bar", "line", "scatter", "pie", "histogram", "box"]
            )
            
            chart_description = st.text_area(
                "Describe what to visualize",
                placeholder="e.g., Show sales trends over time by product category",
                height=150
            )
            
            if st.button("📊 Generate Chart", type="primary"):
                if chart_description:
                    with st.spinner("Creating visualization..."):
                        result = generate_chart(
                            st.session_state.session_id,
                            chart_type,
                            chart_description
                        )
                        if result and result.get('success'):
                            st.session_state.generated_chart = result
                        else:
                            st.error("Failed to generate chart")
                else:
                    st.warning("Please describe what to visualize")
        
        with col2:
            if 'generated_chart' in st.session_state:
                chart_data = st.session_state.generated_chart
                st.subheader("Generated Chart")
                
                # Display image
                if 'image' in chart_data:
                    st.image(chart_data['image'], use_container_width=True)
                
                # Show configuration
                with st.expander("Chart Configuration"):
                    st.json(chart_data.get('config', {}))
    
    # Tab 5: Data Preview
    with tab5:
        st.header("Data Preview")
        
        rows = st.slider("Number of rows to display", 5, 50, 10)
        
        if st.button("🔄 Refresh Preview"):
            preview = get_preview(st.session_state.session_id, rows)
            if preview:
                st.session_state.preview_data = preview
        
        if 'preview_data' in st.session_state or st.button("👁️ Show Preview"):
            preview = get_preview(st.session_state.session_id, rows)
            if preview:
                st.subheader(f"First {rows} rows")
                df_preview = pd.DataFrame(preview['preview']['head'])
                st.dataframe(df_preview, use_container_width=True)
                
                st.caption(f"Total rows: {preview['preview']['total_rows']}")

else:
    # Welcome screen
    st.info("👈 Please upload a spreadsheet file to get started!")
    
    st.markdown("### Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **💬 Natural Language Queries**
        - Ask questions about your data
        - Get insights and analysis
        - Perform calculations
        
        **🧹 Data Cleaning**
        - Automatic cleaning
        - Custom instructions
        - Handle missing values
        """)
    
    with col2:
        st.markdown("""
        **📐 Formula Helper**
        - Get Excel formula suggestions
        - Context-aware recommendations
        - Multiple alternatives
        
        **📈 Chart Generation**
        - Create visualizations
        - Multiple chart types
        - AI-powered insights
        """)

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit, FastAPI, and LangChain",
    unsafe_allow_html=True
)