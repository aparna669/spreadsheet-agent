# AI Spreadsheet Agent - Project Structure

```
ai_spreadsheet_agent/
│
├── README.md                          # Main documentation
├── QUICKSTART.md                      # Quick start guide
├── .gitignore                         # Git ignore rules
├── test_agent.py                      # Test suite
│
├── backend/                           # FastAPI Backend
│   ├── main.py                        # API endpoints & server
│   ├── requirements.txt               # Python dependencies
│   ├── .env.example                   # Environment variables template
│   ├── .env                           # Your API key (create this!)
│   │
│   ├── agent/                         # AI Agent Logic
│   │   ├── __init__.py
│   │   └── spreadsheet_agent.py       # LangChain agent implementation
│   │
│   └── uploads/                       # Uploaded files storage
│       └── .gitkeep
│
└── frontend/                          # Streamlit Frontend
    ├── app.py                         # Web UI application
    └── requirements.txt               # Python dependencies
```

## File Descriptions

### Root Level
- **README.md**: Comprehensive documentation with architecture, setup, and usage
- **QUICKSTART.md**: 5-minute setup guide for quick start
- **.gitignore**: Prevents committing sensitive files (API keys, uploads, etc.)
- **test_agent.py**: Automated test suite to verify all features

### Backend (`backend/`)
- **main.py**: FastAPI application with REST API endpoints
  - `/upload`: Upload spreadsheet files
  - `/query`: Natural language queries
  - `/clean`: Data cleaning
  - `/suggest-formula`: Formula suggestions
  - `/generate-chart`: Chart generation
  - `/preview`: Data preview

- **agent/spreadsheet_agent.py**: Core AI agent logic
  - LangChain integration
  - Claude API communication
  - Data processing with pandas
  - Chart generation with matplotlib

- **requirements.txt**: Backend dependencies
  - FastAPI, Uvicorn (web framework)
  - LangChain, Anthropic (AI agent)
  - Pandas, NumPy (data processing)
  - Matplotlib (visualization)

- **.env.example**: Template for environment configuration
- **.env**: Your actual API key (you create this)

### Frontend (`frontend/`)
- **app.py**: Streamlit web interface
  - File upload widget
  - Chat interface for queries
  - Data cleaning controls
  - Formula helper
  - Chart generator
  - Data preview

- **requirements.txt**: Frontend dependencies
  - Streamlit (web UI)
  - Requests (API client)
  - Pandas (data display)

## Data Flow

```
User Upload File → Streamlit UI → FastAPI Backend → Store in uploads/
                                                    ↓
User Query → Streamlit → FastAPI → SpreadsheetAgent → LangChain → Claude API
                                                                        ↓
Response ← Streamlit ← FastAPI ← Process Result ← ← ← ← ← ← ← ← ← ← ←
```

## Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for Python
- **LangChain**: Framework for building AI agents
- **Claude AI**: Anthropic's language model (via API)
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Plotting and visualization

### Frontend  
- **Streamlit**: Fast web app framework for data apps
- **Python Requests**: HTTP library for API calls

## Key Features by File

### Natural Language Queries (`spreadsheet_agent.py`)
- Creates pandas dataframe agent
- Executes natural language queries
- Returns formatted results

### Data Cleaning (`spreadsheet_agent.py`)
- Automatic: Removes duplicates, handles missing values
- Custom: AI-powered instructions execution
- Returns cleaning summary

### Formula Suggestions (`spreadsheet_agent.py`)
- Analyzes dataset structure
- Generates Excel formulas
- Provides explanations and examples

### Chart Generation (`spreadsheet_agent.py`)
- AI determines appropriate visualization
- Creates charts with matplotlib
- Returns base64 encoded images

## Environment Variables

Create `backend/.env`:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Port Usage
- Backend: `8001` (FastAPI)
- Frontend: `8501` (Streamlit)

## Session Management
- Each upload creates a unique session ID
- Sessions store agent instances in memory
- Sessions can be deleted to cleanup resources