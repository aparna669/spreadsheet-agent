# 🔗 Google Sheets Integration Guide

Complete guide to set up and use Google Sheets integration with the Spreadsheet AI Agent.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Google Cloud Setup](#google-cloud-setup)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [LangChain Integration](#langchain-integration)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The Google Sheets integration adds powerful capabilities:

✅ **Create spreadsheets** - AI creates Google Sheets automatically
✅ **Read data** - Import data from existing sheets
✅ **Write data** - Export analysis results to sheets
✅ **Update cells** - Modify specific cells programmatically
✅ **AI queries** - Natural language commands via LangChain

### Two Integration Methods:

1. **MCP (Model Context Protocol)** - Direct API integration
2. **LangChain Tools** - AI-powered natural language interface

---

## 📦 Prerequisites

### Required:
- Python 3.8+
- Google Account
- Google Cloud Project (free tier works!)

### Skills Needed:
- Basic Python knowledge
- Understanding of Google Sheets
- (Optional) LangChain basics for AI features

---

## ☁️ Google Cloud Setup

### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Create Project"
3. Name it: "Spreadsheet AI Agent"
4. Click "Create"

### Step 2: Enable APIs

1. In Cloud Console, go to "APIs & Services" → "Library"
2. Enable these APIs:
   - **Google Sheets API**
   - **Google Drive API**

### Step 3: Create Service Account

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "Service Account"
3. Name: "spreadsheet-agent"
4. Click "Create and Continue"
5. Role: "Editor" (or "Owner" for full access)
6. Click "Done"

### Step 4: Create Service Account Key

1. Click on your service account
2. Go to "Keys" tab
3. Click "Add Key" → "Create New Key"
4. Choose "JSON"
5. Click "Create"
6. **IMPORTANT:** Save the downloaded JSON file securely!

### Step 5: Enable Domain-Wide Delegation (Optional)

Only needed if working with organization G Suite:
1. In service account settings
2. Check "Enable G Suite Domain-wide Delegation"
3. Save

---

## 🔧 Installation

### Step 1: Install Dependencies

```bash
cd spreadsheet-ai-agent/google-sheets-version
pip install -r requirements_google.txt
```

This installs:
- `google-auth` - Google authentication
- `google-api-python-client` - Sheets API
- `gspread` - Easy Sheets manipulation
- `langchain` - AI agent framework
- `langchain-anthropic` - Claude integration

### Step 2: Setup Credentials

**Option A: Using JSON file**
```bash
# Place your service account JSON file
cp ~/Downloads/your-service-account.json ./credentials.json
```

**Option B: Using environment variable**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

**Option C: In code**
```python
mcp = GoogleSheetsMCP(credentials_path='credentials.json')
```

---

## ⚙️ Configuration

### Create Config File

Create `config.py`:
```python
import os

# Google Sheets Configuration
GOOGLE_CREDENTIALS_PATH = os.getenv(
    'GOOGLE_APPLICATION_CREDENTIALS',
    'credentials.json'
)

# Anthropic API Key (for LangChain)
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')

# Default Settings
DEFAULT_SHARE_WITH = [
    'your-email@example.com'
]
```

### Environment Variables

Create `.env` file:
```bash
GOOGLE_APPLICATION_CREDENTIALS=credentials.json
ANTHROPIC_API_KEY=your_api_key_here
```

---

## 🚀 Usage Examples

### Basic MCP Usage

```python
from google_sheets_mcp import GoogleSheetsMCP
import pandas as pd

# Initialize MCP client
mcp = GoogleSheetsMCP(credentials_path='credentials.json')

# Create new spreadsheet
result = mcp.create_spreadsheet('My Sales Report')
print(result['spreadsheet_url'])  # Open this in browser!

# Write data
df = pd.DataFrame({
    'Product': ['A', 'B', 'C'],
    'Sales': [100, 200, 150]
})

mcp.write_dataframe(
    spreadsheet_id=result['spreadsheet_id'],
    data=df,
    sheet_name='Sheet1'
)

# Read data back
data = mcp.read_spreadsheet(result['spreadsheet_id'])
print(data['data'])  # Returns pandas DataFrame
```

### In Streamlit App

```python
import streamlit as st
from google_sheets_mcp import GoogleSheetsMCP

# Initialize in sidebar
with st.sidebar:
    st.header("🔗 Google Sheets")
    
    # Initialize MCP
    if 'mcp' not in st.session_state:
        st.session_state.mcp = GoogleSheetsMCP(
            credentials_path='credentials.json'
        )
    
    # Create spreadsheet button
    if st.button("Create New Sheet"):
        result = st.session_state.mcp.create_spreadsheet(
            'AI Generated Report',
            data=st.session_state.df
        )
        st.success(result['message'])
        st.markdown(f"[Open Sheet]({result['spreadsheet_url']})")
```

### AI-Powered Commands

```python
from google_sheets_mcp import GoogleSheetsMCP
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate

# Initialize
mcp = GoogleSheetsMCP(credentials_path='credentials.json')
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# Create tools
tools = create_langchain_google_sheets_tools(mcp)

# Create agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can work with Google Sheets."),
    ("human", "{input}")
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Use natural language!
response = agent_executor.invoke({
    "input": "Create a new spreadsheet called 'Q1 Sales Report'"
})
print(response)
```

---

## 🤖 LangChain Integration

### Setup AI Agent

```python
# google_sheets_agent.py
from google_sheets_mcp import GoogleSheetsMCP, create_langchain_google_sheets_tools
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

def setup_google_sheets_agent(credentials_path: str, api_key: str):
    """Setup LangChain agent with Google Sheets tools"""
    
    # Initialize MCP
    mcp = GoogleSheetsMCP(credentials_path=credentials_path)
    
    # Create tools
    tools = create_langchain_google_sheets_tools(mcp)
    
    # Create LLM
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        anthropic_api_key=api_key
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant that can work with Google Sheets.
        
        Available actions:
        - Create new spreadsheets
        - Read existing spreadsheets
        - List sheets in a spreadsheet
        - Get spreadsheet information
        
        Always provide the spreadsheet URL when creating sheets.
        When reading data, summarize key insights.
        Be helpful and clear in your responses."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=memory,
        verbose=True
    )
    
    return agent_executor

# Usage
agent = setup_google_sheets_agent('credentials.json', 'your_api_key')

# Natural language commands!
responses = [
    agent.invoke({"input": "Create a spreadsheet called 'Monthly Budget'"}),
    agent.invoke({"input": "Read the spreadsheet with ID abc123"}),
    agent.invoke({"input": "What sheets are in that spreadsheet?"})
]
```

### Example AI Queries

```python
# The agent can understand these natural language commands:

"Create a new spreadsheet called 'Sales Dashboard 2024'"
→ Creates spreadsheet, returns URL

"Read data from spreadsheet ID 1BxiMVs0XRA5nFMd..."
→ Reads data, shows preview

"What's in that spreadsheet?"
→ Lists all sheets with info

"Add a new sheet called 'Q1 Data'"
→ Creates new sheet in existing spreadsheet

"Share this spreadsheet with john@example.com"
→ Grants access to user

"Get info about spreadsheet ID abc123"
→ Returns metadata
```

---

## 🎯 Complete Workflow Example

```python
import streamlit as st
import pandas as pd
from google_sheets_mcp import GoogleSheetsMCP
from google_sheets_agent import setup_google_sheets_agent

# Streamlit app
st.title("🤖 AI Spreadsheet Agent")

# Initialize
if 'mcp' not in st.session_state:
    st.session_state.mcp = GoogleSheetsMCP(credentials_path='credentials.json')

if 'agent' not in st.session_state:
    st.session_state.agent = setup_google_sheets_agent(
        'credentials.json',
        st.secrets['ANTHROPIC_API_KEY']
    )

# Upload data
uploaded_file = st.file_uploader("Upload CSV")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
    
    # AI Command
    command = st.text_input(
        "What would you like to do?",
        placeholder="e.g., Create a spreadsheet with this data"
    )
    
    if st.button("Execute"):
        # If command involves current data, pass it
        if 'data' in command.lower() or 'this' in command.lower():
            # Create spreadsheet with data
            result = st.session_state.mcp.create_spreadsheet(
                'AI Upload',
                data=df
            )
            st.success(result['message'])
            st.markdown(f"[Open Sheet]({result['spreadsheet_url']})")
        else:
            # Let AI handle it
            response = st.session_state.agent.invoke({"input": command})
            st.write(response['output'])
```

---

## 🐛 Troubleshooting

### Error: "Credentials not found"

**Solution:**
```bash
# Check file exists
ls -la credentials.json

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/credentials.json"

# Or use absolute path
GOOGLE_APPLICATION_CREDENTIALS="/full/path/to/credentials.json"
```

### Error: "Permission denied"

**Cause:** Service account doesn't have access to spreadsheet

**Solutions:**
1. **Share manually:** Share sheet with service account email
   ```
   your-service-account@project-id.iam.gserviceaccount.com
   ```

2. **Auto-share when creating:**
   ```python
   mcp.create_spreadsheet(
       'My Sheet',
       share_with=['your-email@example.com']
   )
   ```

### Error: "API not enabled"

**Solution:**
1. Go to Google Cloud Console
2. Enable "Google Sheets API"
3. Enable "Google Drive API"
4. Wait 1-2 minutes for propagation

### Error: "Quota exceeded"

**Cause:** Hitting API rate limits

**Solutions:**
1. Reduce request frequency
2. Batch operations together
3. Use exponential backoff
4. Request quota increase in Cloud Console

### Error: "LangChain import failed"

**Solution:**
```bash
pip install langchain langchain-anthropic anthropic
```

### Slow Performance

**Optimizations:**
1. **Batch updates:**
   ```python
   # ❌ Slow: Multiple calls
   for row in rows:
       mcp.update_cell(sheet_id, f'A{i}', row)
   
   # ✅ Fast: Single call
   mcp.write_dataframe(sheet_id, df)
   ```

2. **Cache reads:**
   ```python
   @st.cache_data(ttl=300)  # Cache for 5 minutes
   def read_sheet(sheet_id):
       return mcp.read_spreadsheet(sheet_id)
   ```

---

## 📊 Testing Your Setup

### Test Script

Create `test_google_sheets.py`:
```python
from google_sheets_mcp import GoogleSheetsMCP
import pandas as pd

def test_google_sheets():
    """Test Google Sheets integration"""
    
    print("🧪 Testing Google Sheets MCP...")
    
    # Initialize
    mcp = GoogleSheetsMCP(credentials_path='credentials.json')
    print("✅ MCP initialized")
    
    # Create test spreadsheet
    result = mcp.create_spreadsheet('Test Spreadsheet')
    if result['success']:
        print(f"✅ Created spreadsheet: {result['spreadsheet_url']}")
        sheet_id = result['spreadsheet_id']
    else:
        print(f"❌ Failed: {result['message']}")
        return
    
    # Write test data
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Score': [95, 87, 92]
    })
    
    result = mcp.write_dataframe(sheet_id, df)
    if result['success']:
        print(f"✅ Wrote data: {result['rows_written']} rows")
    else:
        print(f"❌ Failed: {result['message']}")
        return
    
    # Read data back
    result = mcp.read_spreadsheet(sheet_id)
    if result['success']:
        print(f"✅ Read data: {len(result['data'])} rows")
        print(result['data'])
    else:
        print(f"❌ Failed: {result['message']}")
    
    print("\n🎉 All tests passed!")
    print(f"📊 Check your spreadsheet: {result.get('spreadsheet_url', 'N/A')}")

if __name__ == "__main__":
    test_google_sheets()
```

Run test:
```bash
python test_google_sheets.py
```

---

## 🎓 Advanced Features

### Custom Formatting

```python
def format_spreadsheet(mcp, spreadsheet_id):
    """Apply custom formatting"""
    
    # Bold headers
    requests = [{
        'repeatCell': {
            'range': {
                'sheetId': 0,
                'startRowIndex': 0,
                'endRowIndex': 1
            },
            'cell': {
                'userEnteredFormat': {
                    'textFormat': {'bold': True},
                    'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
                }
            },
            'fields': 'userEnteredFormat(textFormat,backgroundColor)'
        }
    }]
    
    mcp.service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={'requests': requests}
    ).execute()
```

### Scheduled Updates

```python
import schedule
import time

def sync_data():
    """Sync data to Google Sheets every hour"""
    df = fetch_latest_data()  # Your data source
    mcp.write_dataframe(SPREADSHEET_ID, df)
    print(f"✅ Synced at {datetime.now()}")

# Schedule
schedule.every().hour.do(sync_data)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 🔒 Security Best Practices

1. **Never commit credentials:**
   ```bash
   echo "credentials.json" >> .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use environment variables:**
   ```python
   credentials_path = os.getenv('GOOGLE_CREDENTIALS')
   ```

3. **Rotate keys regularly**

4. **Use least privilege:**
   - Service account: Editor role only
   - Share sheets: Reader/Writer, not Owner

5. **Encrypt at rest:**
   ```bash
   # Encrypt credentials
   gpg -c credentials.json
   # Decrypt when needed
   gpg credentials.json.gpg
   ```

---

## 📚 Resources

- [Google Sheets API Docs](https://developers.google.com/sheets/api)
- [gspread Documentation](https://docs.gspread.org/)
- [LangChain Documentation](https://python.langchain.com/)
- [Service Account Guide](https://cloud.google.com/iam/docs/service-accounts)

---

**You're ready to integrate Google Sheets! 🎉**

Start with the test script, then integrate into your Streamlit app.

