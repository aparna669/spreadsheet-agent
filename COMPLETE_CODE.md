# AI Spreadsheet Agent - Complete Code Reference

## Table of Contents
1. [Backend Code](#backend-code)
2. [Frontend Code](#frontend-code)
3. [Configuration Files](#configuration-files)
4. [Test Suite](#test-suite)
5. [Setup Instructions](#setup-instructions)

---

## Backend Code

### 1. backend/main.py
```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import os
from pathlib import Path
import uuid
from typing import Optional, Dict, Any
import json

from agent.spreadsheet_agent import SpreadsheetAgent

app = FastAPI(title="AI Spreadsheet Agent API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Store active sessions
sessions: Dict[str, SpreadsheetAgent] = {}


class QueryRequest(BaseModel):
    session_id: str
    query: str


class CleanDataRequest(BaseModel):
    session_id: str
    instructions: Optional[str] = None


class FormulaRequest(BaseModel):
    session_id: str
    description: str
    context: Optional[str] = None


class ChartRequest(BaseModel):
    session_id: str
    chart_type: str
    description: str


@app.get("/")
async def root():
    return {"message": "AI Spreadsheet Agent API", "status": "running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a spreadsheet file and create a session"""
    try:
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(400, "Only CSV and Excel files are supported")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Save file
        file_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Create agent session
        agent = SpreadsheetAgent(str(file_path))
        sessions[session_id] = agent
        
        # Get basic info
        df_info = agent.get_dataframe_info()
        
        return {
            "session_id": session_id,
            "filename": file.filename,
            "info": df_info,
            "message": "File uploaded successfully"
        }
    
    except Exception as e:
        raise HTTPException(500, f"Error uploading file: {str(e)}")


@app.post("/query")
async def query_data(request: QueryRequest):
    """Process natural language query on spreadsheet data"""
    try:
        if request.session_id not in sessions:
            raise HTTPException(404, "Session not found")
        
        agent = sessions[request.session_id]
        result = agent.process_query(request.query)
        
        return {
            "session_id": request.session_id,
            "query": request.query,
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(500, f"Error processing query: {str(e)}")


@app.post("/clean")
async def clean_data(request: CleanDataRequest):
    """Automatically clean and format spreadsheet data"""
    try:
        if request.session_id not in sessions:
            raise HTTPException(404, "Session not found")
        
        agent = sessions[request.session_id]
        result = agent.clean_data(request.instructions)
        
        return {
            "session_id": request.session_id,
            "result": result,
            "message": "Data cleaning completed"
        }
    
    except Exception as e:
        raise HTTPException(500, f"Error cleaning data: {str(e)}")


@app.post("/suggest-formula")
async def suggest_formula(request: FormulaRequest):
    """Suggest Excel/spreadsheet formulas based on description"""
    try:
        if request.session_id not in sessions:
            raise HTTPException(404, "Session not found")
        
        agent = sessions[request.session_id]
        result = agent.suggest_formula(request.description, request.context)
        
        return {
            "session_id": request.session_id,
            "description": request.description,
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(500, f"Error suggesting formula: {str(e)}")


@app.post("/generate-chart")
async def generate_chart(request: ChartRequest):
    """Generate chart/visualization based on description"""
    try:
        if request.session_id not in sessions:
            raise HTTPException(404, "Session not found")
        
        agent = sessions[request.session_id]
        result = agent.generate_chart(request.chart_type, request.description)
        
        return {
            "session_id": request.session_id,
            "chart_type": request.chart_type,
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(500, f"Error generating chart: {str(e)}")


@app.get("/preview/{session_id}")
async def preview_data(session_id: str, rows: int = 10):
    """Preview the current state of the spreadsheet"""
    try:
        if session_id not in sessions:
            raise HTTPException(404, "Session not found")
        
        agent = sessions[session_id]
        preview = agent.get_preview(rows)
        
        return {
            "session_id": session_id,
            "preview": preview
        }
    
    except Exception as e:
        raise HTTPException(500, f"Error previewing data: {str(e)}")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its associated file"""
    try:
        if session_id in sessions:
            # Remove from sessions
            agent = sessions.pop(session_id)
            
            # Delete file
            if os.path.exists(agent.file_path):
                os.remove(agent.file_path)
            
            return {"message": "Session deleted successfully"}
        else:
            raise HTTPException(404, "Session not found")
    
    except Exception as e:
        raise HTTPException(500, f"Error deleting session: {str(e)}")


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "sessions": [
            {
                "session_id": sid,
                "info": agent.get_dataframe_info()
            }
            for sid, agent in sessions.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. backend/agent/spreadsheet_agent.py
```python
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import json
import os
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import matplotlib.pyplot as plt
import io
import base64


class SpreadsheetAgent:
    """AI agent for spreadsheet operations using LangChain"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = self._load_dataframe(file_path)
        self.original_df = self.df.copy()
        
        # Initialize Claude with API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            anthropic_api_key=api_key,
            temperature=0
        )
        
        # Create pandas dataframe agent
        self.pandas_agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            agent_type="tool-calling",
            verbose=True,
            allow_dangerous_code=True
        )
    
    def _load_dataframe(self, file_path: str) -> pd.DataFrame:
        """Load spreadsheet into pandas DataFrame"""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def get_dataframe_info(self) -> Dict[str, Any]:
        """Get basic information about the dataframe"""
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "column_names": self.df.columns.tolist(),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "memory_usage": f"{self.df.memory_usage(deep=True).sum() / 1024:.2f} KB",
            "missing_values": self.df.isnull().sum().to_dict()
        }
    
    def get_preview(self, rows: int = 10) -> Dict[str, Any]:
        """Get a preview of the dataframe"""
        return {
            "head": self.df.head(rows).to_dict(orient='records'),
            "columns": self.df.columns.tolist(),
            "total_rows": len(self.df)
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query on the spreadsheet data"""
        try:
            # Update the agent's dataframe reference
            self.pandas_agent = create_pandas_dataframe_agent(
                self.llm,
                self.df,
                agent_type="tool-calling",
                verbose=True,
                allow_dangerous_code=True
            )
            
            # Execute query
            response = self.pandas_agent.invoke({"input": query})
            
            return {
                "success": True,
                "query": query,
                "response": response["output"],
                "type": "query_result"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def clean_data(self, instructions: Optional[str] = None) -> Dict[str, Any]:
        """Automatically clean and format spreadsheet data"""
        try:
            cleaning_steps = []
            
            if instructions:
                # Use AI to clean based on instructions
                prompt = f"""
                You are helping clean a spreadsheet dataset. Here's the current state:
                
                Columns: {self.df.columns.tolist()}
                Shape: {self.df.shape}
                Data types: {self.df.dtypes.to_dict()}
                Missing values: {self.df.isnull().sum().to_dict()}
                
                User instructions: {instructions}
                
                Provide specific Python pandas code to clean this data according to the instructions.
                Return only the code, with each operation on a new line.
                Use 'df' as the dataframe variable name.
                """
                
                response = self.llm.invoke(prompt)
                code = response.content
                
                # Execute the cleaning code
                local_vars = {"df": self.df, "pd": pd, "np": np}
                exec(code, {}, local_vars)
                self.df = local_vars["df"]
                
                cleaning_steps.append({
                    "type": "custom",
                    "description": instructions,
                    "code": code
                })
            
            else:
                # Automatic cleaning
                initial_rows = len(self.df)
                
                # Remove duplicate rows
                duplicates = self.df.duplicated().sum()
                if duplicates > 0:
                    self.df = self.df.drop_duplicates()
                    cleaning_steps.append({
                        "type": "remove_duplicates",
                        "removed": duplicates
                    })
                
                # Handle missing values
                for col in self.df.columns:
                    missing = self.df[col].isnull().sum()
                    if missing > 0:
                        if self.df[col].dtype in ['int64', 'float64']:
                            # Fill numeric with median
                            self.df[col].fillna(self.df[col].median(), inplace=True)
                            cleaning_steps.append({
                                "type": "fill_missing",
                                "column": col,
                                "method": "median",
                                "count": missing
                            })
                        else:
                            # Fill categorical with mode or 'Unknown'
                            if not self.df[col].mode().empty:
                                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                            else:
                                self.df[col].fillna('Unknown', inplace=True)
                            cleaning_steps.append({
                                "type": "fill_missing",
                                "column": col,
                                "method": "mode/unknown",
                                "count": missing
                            })
                
                # Strip whitespace from string columns
                for col in self.df.select_dtypes(include=['object']).columns:
                    self.df[col] = self.df[col].str.strip()
                    cleaning_steps.append({
                        "type": "strip_whitespace",
                        "column": col
                    })
            
            return {
                "success": True,
                "cleaning_steps": cleaning_steps,
                "before_rows": initial_rows if not instructions else len(self.original_df),
                "after_rows": len(self.df),
                "message": "Data cleaning completed successfully"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def suggest_formula(self, description: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Suggest Excel/spreadsheet formulas based on description"""
        try:
            column_info = "\n".join([
                f"- {col} ({dtype})" 
                for col, dtype in self.df.dtypes.items()
            ])
            
            sample_data = self.df.head(3).to_string()
            
            prompt = f"""
            You are an expert in Excel and spreadsheet formulas. 
            
            Dataset information:
            Columns:
            {column_info}
            
            Sample data:
            {sample_data}
            
            User request: {description}
            {f"Additional context: {context}" if context else ""}
            
            Provide:
            1. The Excel formula(s) needed
            2. Step-by-step explanation
            3. Example usage with actual column names from the dataset
            4. Alternative approaches if applicable
            
            Format your response as JSON with these keys:
            - formula: the main formula
            - explanation: detailed explanation
            - example: example with real column names
            - alternatives: list of alternative formulas (if any)
            - notes: any important notes or caveats
            """
            
            response = self.llm.invoke(prompt)
            
            # Try to parse as JSON, otherwise return as text
            try:
                result = json.loads(response.content)
            except:
                result = {
                    "formula": "See explanation",
                    "explanation": response.content,
                    "example": "",
                    "alternatives": [],
                    "notes": ""
                }
            
            return {
                "success": True,
                "description": description,
                "result": result
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_chart(self, chart_type: str, description: str) -> Dict[str, Any]:
        """Generate chart/visualization based on description"""
        try:
            # Ask AI to determine appropriate columns and parameters
            prompt = f"""
            You are a data visualization expert. Based on this dataset and request, 
            determine what chart to create.
            
            Dataset columns: {self.df.columns.tolist()}
            Data types: {self.df.dtypes.to_dict()}
            Sample data:
            {self.df.head(5).to_string()}
            
            Chart type requested: {chart_type}
            Description: {description}
            
            Provide a JSON response with:
            - chart_type: specific matplotlib chart type (bar, line, scatter, pie, hist, box)
            - x_column: column for x-axis (if applicable)
            - y_column: column(s) for y-axis
            - title: appropriate chart title
            - xlabel: x-axis label
            - ylabel: y-axis label
            - additional_params: any other parameters needed
            
            Respond with ONLY valid JSON, no other text.
            """
            
            response = self.llm.invoke(prompt)
            chart_config = json.loads(response.content)
            
            # Generate the chart
            plt.figure(figsize=(10, 6))
            
            if chart_config['chart_type'] == 'bar':
                x_data = self.df[chart_config['x_column']].value_counts()
                plt.bar(x_data.index, x_data.values)
            
            elif chart_config['chart_type'] == 'line':
                plt.plot(self.df[chart_config['x_column']], 
                        self.df[chart_config['y_column']])
            
            elif chart_config['chart_type'] == 'scatter':
                plt.scatter(self.df[chart_config['x_column']], 
                           self.df[chart_config['y_column']])
            
            elif chart_config['chart_type'] == 'pie':
                data = self.df[chart_config['y_column']].value_counts()
                plt.pie(data.values, labels=data.index, autopct='%1.1f%%')
            
            elif chart_config['chart_type'] == 'hist':
                plt.hist(self.df[chart_config['y_column']], bins=20)
            
            elif chart_config['chart_type'] == 'box':
                self.df.boxplot(column=chart_config['y_column'])
            
            plt.title(chart_config['title'])
            if 'xlabel' in chart_config:
                plt.xlabel(chart_config['xlabel'])
            if 'ylabel' in chart_config:
                plt.ylabel(chart_config['ylabel'])
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return {
                "success": True,
                "chart_type": chart_config['chart_type'],
                "config": chart_config,
                "image": f"data:image/png;base64,{image_base64}",
                "description": description
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "description": description
            }
```

### 3. backend/agent/__init__.py
```python
from .spreadsheet_agent import SpreadsheetAgent

__all__ = ['SpreadsheetAgent']
```

### 4. backend/requirements.txt
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
pandas==2.2.0
numpy==1.26.3
openpyxl==3.1.2
python-multipart==0.0.6
langchain==0.1.4
langchain-anthropic==0.1.4
langchain-experimental==0.0.50
anthropic==0.18.1
matplotlib==3.8.2
pydantic==2.5.3
python-dotenv==1.0.1
```

### 5. backend/.env.example
```
# Anthropic API Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Specify model (default: claude-sonnet-4-20250514)
# ANTHROPIC_MODEL=claude-sonnet-4-20250514

# Optional: API Base URL (default: http://localhost:8000)
# API_BASE_URL=http://localhost:8000
```

---

## Frontend Code

### frontend/app.py
```python
import streamlit as st
import requests
import pandas as pd
import json
from typing import Optional
import base64
from io import BytesIO

# Configuration
API_BASE_URL = "http://localhost:8000"

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
    "<div style='text-align: center'>Built with ❤️ using Streamlit, FastAPI, and LangChain</div>",
    unsafe_allow_html=True
)
```

### frontend/requirements.txt
```
streamlit==1.30.0
requests==2.31.0
pandas==2.2.0
```

---

## Configuration Files

### .gitignore
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# Environment Variables
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Uploads
backend/uploads/*
!backend/uploads/.gitkeep

# Logs
*.log

# Streamlit
.streamlit/
```

---

## Test Suite

### test_agent.py
```python
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
```

---

## Setup Instructions

### 1. Environment Setup

Create `.env` file in `backend/` directory:
```bash
ANTHROPIC_API_KEY=your_actual_api_key_here
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run Application

**Terminal 1 (Backend):**
```bash
cd backend
source venv/bin/activate
python main.py
```

**Terminal 2 (Frontend):**
```bash
cd frontend
source venv/bin/activate
streamlit run app.py
```

### 5. Run Tests (Optional)
```bash
python test_agent.py
```

---

## API Endpoints

- `POST /upload` - Upload spreadsheet
- `POST /query` - Natural language query
- `POST /clean` - Clean data
- `POST /suggest-formula` - Get formula suggestions
- `POST /generate-chart` - Generate charts
- `GET /preview/{session_id}` - Preview data
- `DELETE /session/{session_id}` - Delete session
- `GET /sessions` - List all sessions

## Access Points

- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Frontend UI: http://localhost:8501