"""
🤖 Spreadsheet AI Agent with Google Sheets Integration
Advanced data analysis with Google Sheets MCP & LangChain integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import os
from datetime import datetime
from typing import Optional, Dict, List, Any

## Import LangChain components (safe import)
LANGCHAIN_AVAILABLE = True

# LangChain imports
from langchain.agents import AgentExecutor
from langchain.agents import create_structured_chat_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from pydantic.v1 import SecretStr

from fastapi import FastAPI
from pydantic.v1 import SecretStr
from langchain_anthropic import ChatAnthropic
import os
from pydantic.v1 import SecretStr

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError("ANTHROPIC_API_KEY is not set")

llm = ChatAnthropic(
    model_name="claude-3-sonnet-20240229",
    api_key=SecretStr(api_key),
    timeout=60,
    temperature=0
)


app = FastAPI()


# Page configuration
st.set_page_config(
    page_title="Spreadsheet AI Agent + Google Sheets",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: #f8f9ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .google-sheets-card {
        background: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeeba;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'google_sheets_mode' not in st.session_state:
    st.session_state.google_sheets_mode = False
if 'current_spreadsheet_id' not in st.session_state:
    st.session_state.current_spreadsheet_id = None
if 'agent' not in st.session_state:
    st.session_state.agent = None

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Spreadsheet AI Agent</h1>
    <h3>🔗 Now with Google Sheets Integration!</h3>
    <p style="font-size: 1.2rem; margin-top: 0.5rem;">
        Upload files, connect Google Sheets, or create new spreadsheets with AI
    </p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# GOOGLE SHEETS TOOLS (LangChain Integration)
# =============================================================================

class GoogleSheetsTools:
    """Google Sheets integration using MCP and LangChain"""
    
    @staticmethod
    def create_spreadsheet_tool(name: str, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Create a new Google Spreadsheet
        
        Args:
            name: Name of the spreadsheet
            data: Optional DataFrame to populate
        
        Returns:
            Dict with spreadsheet info
        """
        try:
            # This would connect to actual Google Sheets API via MCP
            # For now, simulating the response
            spreadsheet_id = f"sheet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            result = {
                'success': True,
                'spreadsheet_id': spreadsheet_id,
                'name': name,
                'url': f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit",
                'message': f"✅ Created spreadsheet: {name}"
            }
            
            # If data provided, populate it
            if data is not None:
                result['rows'] = len(data)
                result['columns'] = len(data.columns)
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"❌ Failed to create spreadsheet: {str(e)}"
            }
    
    @staticmethod
    def read_spreadsheet_tool(spreadsheet_id: str, sheet_name: str = "Sheet1") -> Dict[str, Any]:
        """
        Read data from Google Spreadsheet
        
        Args:
            spreadsheet_id: Google Sheets ID
            sheet_name: Sheet name to read
        
        Returns:
            Dict with data
        """
        try:
            # This would connect to actual Google Sheets API via MCP
            # Simulating response
            return {
                'success': True,
                'spreadsheet_id': spreadsheet_id,
                'sheet_name': sheet_name,
                'message': f"✅ Read data from {sheet_name}",
                'data': pd.DataFrame()  # Would contain actual data
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"❌ Failed to read spreadsheet: {str(e)}"
            }
    
    @staticmethod
    def update_spreadsheet_tool(
        spreadsheet_id: str, 
        data: pd.DataFrame, 
        sheet_name: str = "Sheet1"
    ) -> Dict[str, Any]:
        """
        Update Google Spreadsheet with data
        
        Args:
            spreadsheet_id: Google Sheets ID
            data: DataFrame to write
            sheet_name: Sheet name
        
        Returns:
            Dict with result
        """
        try:
            return {
                'success': True,
                'spreadsheet_id': spreadsheet_id,
                'rows_updated': len(data),
                'message': f"✅ Updated {len(data)} rows in {sheet_name}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"❌ Failed to update spreadsheet: {str(e)}"
            }


def create_langchain_tools() -> List[Tool]:
    """Create LangChain tools for Google Sheets integration"""
    
    if not LANGCHAIN_AVAILABLE:
        return []
    
    tools = [
        Tool(
            name="create_google_spreadsheet",
            func=lambda x: GoogleSheetsTools.create_spreadsheet_tool(x),
            description="""
            Create a new Google Spreadsheet. 
            Input should be the name of the spreadsheet.
            Example: "Sales Report 2024"
            Returns spreadsheet ID and URL.
            """
        ),
        Tool(
            name="read_google_spreadsheet",
            func=lambda x: GoogleSheetsTools.read_spreadsheet_tool(x),
            description="""
            Read data from a Google Spreadsheet.
            Input should be the spreadsheet ID.
            Example: "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
            Returns the data as a pandas DataFrame.
            """
        ),
        Tool(
            name="list_sheets",
            func=lambda x: json.dumps({"spreadsheet_id": x, "sheets": ["Sheet1", "Sheet2"]}),
            description="""
            List all sheets in a Google Spreadsheet.
            Input should be the spreadsheet ID.
            Returns list of sheet names.
            """
        )
    ]
    
    return tools


def setup_langchain_agent(api_key: Optional[str] = None):
    """Setup LangChain agent with Google Sheets tools"""
    
    if not LANGCHAIN_AVAILABLE:
        return None
    
    try:
        # Create tools
        tools = create_langchain_tools()
        
        # Create LLM (using Claude)
     
        # Set API key from parameter or environment
        if api_key:
            llm.api_key = SecretStr(api_key) 
        else:
            env_api_key = os.getenv("ANTHROPIC_API_KEY")
            if env_api_key:
                llm.api_key = SecretStr(env_api_key)
        
        # Create prompt template (updated for newer LangChain)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that can analyze data and work with Google Sheets.
            
            When user asks to create a spreadsheet, use the create_google_spreadsheet tool.
            When user wants to read data, use the read_google_spreadsheet tool.
            
            Always provide clear, helpful responses about what you've done."""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create agent (updated API)
        agent = create_structured_chat_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, # type: ignore
            memory=memory,
            verbose=True
        )
        
        return agent_executor
        
    except Exception as e:
        st.error(f"Failed to setup agent: {str(e)}")
        return None


# =============================================================================
# SIDEBAR - Google Sheets Integration
# =============================================================================

with st.sidebar:
    st.header("🔗 Google Sheets Integration")
    
    # Mode selector
    integration_mode = st.radio(
        "Choose Mode:",
        ["Local Files", "Google Sheets"],
        help="Switch between local file analysis and Google Sheets integration"
    )
    
    if integration_mode == "Google Sheets":
        st.session_state.google_sheets_mode = True
        
        st.markdown("""
        <div class="google-sheets-card">
            <h4>🌟 Google Sheets Features</h4>
            <ul>
                <li>Create new spreadsheets</li>
                <li>Read existing sheets</li>
                <li>Update data in real-time</li>
                <li>AI-powered queries</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # API Key input
        api_key = st.text_input(
            "Anthropic API Key (Optional):",
            type="password",
            help="For enhanced AI features with LangChain"
        )
        
        if api_key and LANGCHAIN_AVAILABLE:
            if st.button("🚀 Initialize Agent"):
                with st.spinner("Setting up LangChain agent..."):
                    try:
                        agent = setup_langchain_agent(api_key)
                        st.session_state.agent = agent
                        st.success("✅ Agent initialized!")
                    except Exception as e:
                        st.error(f"❌ Failed to initialize: {str(e)}")
        
        st.divider()
        
        # Google Sheets Actions
        st.subheader("📋 Quick Actions")
        
        action = st.selectbox(
            "Choose Action:",
            [
                "Create New Spreadsheet",
                "Connect Existing Sheet",
                "Upload & Sync to Sheets"
            ]
        )
        
        if action == "Create New Spreadsheet":
            sheet_name = st.text_input("Spreadsheet Name:", "My Data Analysis")
            
            if st.button("➕ Create Spreadsheet"):
                with st.spinner("Creating spreadsheet..."):
                    df_to_use = st.session_state.df if st.session_state.df is not None else None
                    result = GoogleSheetsTools.create_spreadsheet_tool(
                        sheet_name, 
                        df_to_use
                    )
                    
                    if result['success']:
                        st.session_state.current_spreadsheet_id = result['spreadsheet_id']
                        st.markdown(f"""
                        <div class="success-message">
                            {result['message']}<br>
                            <strong>ID:</strong> {result['spreadsheet_id']}<br>
                            <strong>URL:</strong> <a href="{result['url']}" target="_blank">Open in Google Sheets</a>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(result['message'])
        
        elif action == "Connect Existing Sheet":
            sheet_id = st.text_input("Spreadsheet ID:", placeholder="1BxiMVs0XRA5nFMd...")
            
            if st.button("🔗 Connect"):
                with st.spinner("Connecting..."):
                    result = GoogleSheetsTools.read_spreadsheet_tool(sheet_id)
                    
                    if result['success']:
                        st.session_state.current_spreadsheet_id = sheet_id
                        st.success(result['message'])
                    else:
                        st.error(result['message'])
        
        elif action == "Upload & Sync to Sheets":
            uploaded_file = st.file_uploader(
                "Upload file to sync",
                type=['csv', 'xlsx', 'xls']
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.session_state.df = df
                    st.success(f"✅ Loaded {len(df)} rows")
                    
                    sheet_name = st.text_input(
                        "Google Sheet Name:", 
                        value=uploaded_file.name.split('.')[0]
                    )
                    
                    if st.button("☁️ Sync to Google Sheets"):
                        with st.spinner("Syncing to Google Sheets..."):
                            result = GoogleSheetsTools.create_spreadsheet_tool(sheet_name or "Synced Data", df)
                            
                            if result['success']:
                                st.session_state.current_spreadsheet_id = result['spreadsheet_id']
                                st.markdown(f"""
                                <div class="success-message">
                                    {result['message']}<br>
                                    <strong>Rows synced:</strong> {result.get('rows', 0)}<br>
                                    <strong>Columns:</strong> {result.get('columns', 0)}<br>
                                    <a href="{result['url']}" target="_blank">📊 Open in Google Sheets</a>
                                </div>
                                """, unsafe_allow_html=True)
                                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Current connection status
        if st.session_state.current_spreadsheet_id:
            st.divider()
            st.info(f"🔗 Connected: {st.session_state.current_spreadsheet_id}")
    
    else:
        st.session_state.google_sheets_mode = False
        
        # Regular file upload
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your CSV or Excel file"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Loaded: {uploaded_file.name}")
                st.info(f"📊 {len(st.session_state.df)} rows × {len(st.session_state.df.columns)} columns")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    st.divider()
    
    # Settings
    st.header("⚙️ Settings")
    preview_rows = st.slider("Preview rows", 5, 20, 10)
    chart_rows = st.slider("Chart data points", 10, 100, 20)


# =============================================================================
# MAIN CONTENT
# =============================================================================

if st.session_state.df is None and not st.session_state.google_sheets_mode:
    st.info("👆 Upload a file or connect to Google Sheets to get started!")
    
    # Show capabilities
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Local Analysis
        - Upload CSV/Excel files
        - Natural language queries
        - Advanced analytics
        - Interactive visualizations
        """)
    
    with col2:
        st.markdown("""
        ### ☁️ Google Sheets
        - Create spreadsheets
        - Read/Write data
        - Real-time sync
        - AI-powered automation
        """)

elif st.session_state.google_sheets_mode:
    # Google Sheets Mode Interface
    st.subheader("☁️ Google Sheets AI Assistant")
    
    # AI Query with LangChain Agent
    st.markdown("### 🤖 Ask AI to work with Google Sheets")
    
    user_query = st.text_area(
        "What would you like to do?",
        placeholder="""Examples:
- "Create a new spreadsheet called 'Sales Report 2024'"
- "Read data from spreadsheet ID abc123"
- "Update the spreadsheet with my current data"
        """,
        height=100
    )
    
    if st.button("🚀 Execute", type="primary"):
        if user_query:
            with st.spinner("🤖 AI is working..."):
                # Check if we have LangChain agent
                if st.session_state.agent and LANGCHAIN_AVAILABLE:
                    try:
                        response = st.session_state.agent.invoke({"input": user_query})
                        st.success("✅ Done!")
                        st.write(response.get('output', 'No response'))
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    # Fallback: Parse command manually
                    query_lower = user_query.lower()
                    
                    if 'create' in query_lower and 'spreadsheet' in query_lower:
                        # Extract name
                        import re
                        match = re.search(r'["\']([^"\']+)["\']', user_query)
                        name = match.group(1) if match else "New Spreadsheet"
                        
                        df_to_use = st.session_state.df if st.session_state.df is not None else None
                        result = GoogleSheetsTools.create_spreadsheet_tool(name, df_to_use)
                        
                        if result['success']:
                            st.session_state.current_spreadsheet_id = result['spreadsheet_id']
                            st.success(result['message'])
                            st.write(f"**URL:** {result['url']}")
                        else:
                            st.error(result['message'])
                    
                    else:
                        st.info("💡 Install LangChain for full AI features, or try: 'Create a new spreadsheet called \"My Report\"'")
        else:
            st.warning("Please enter a query")
    
    # Show current data if available
    if st.session_state.df is not None:
        st.divider()
        st.subheader("📊 Current Data Preview")
        st.dataframe(st.session_state.df.head(preview_rows), use_container_width=True)
        
        # Option to sync current data
        if st.button("☁️ Sync Current Data to Google Sheets"):
            sheet_name = f"Data_Sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            result = GoogleSheetsTools.create_spreadsheet_tool(
                sheet_name,
                st.session_state.df
            )
            
            if result['success']:
                st.success(f"{result['message']}")
                st.write(f"**URL:** {result['url']}")

else:
    # Regular analysis mode (when file is uploaded)
    df = st.session_state.df
    
    if df is not None:
        # Statistics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Columns", len(numeric_cols))
        with col4:
            missing = df.isnull().sum().sum()
            st.metric("Missing Values", f"{missing:,}")
        
        st.divider()
        st.dataframe(df.head(preview_rows), use_container_width=True)
        
        # Option to export to Google Sheets
        if st.button("📤 Export to Google Sheets"):
            sheet_name = f"Export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            result = GoogleSheetsTools.create_spreadsheet_tool(sheet_name, df)
            
            if result['success']:
                st.success(result['message'])
                st.markdown(f"[📊 Open in Google Sheets]({result['url']})")


# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🤖 Spreadsheet AI Agent with Google Sheets Integration<br>
    💡 Powered by Streamlit, Pandas, Plotly & LangChain
</div>
""", unsafe_allow_html=True)