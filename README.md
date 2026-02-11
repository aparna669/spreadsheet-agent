spreadsheet-agent
# AI Spreadsheet Agent 🤖📊

An intelligent spreadsheet assistant powered by Claude AI and LangChain that allows you to interact with your spreadsheets using natural language.

## Features

### 💬 Natural Language Queries
- Ask questions about your data in plain English
- Get insights, perform calculations, and analyze trends
- Conversational interface for data exploration

### 🧹 Automated Data Cleaning
- Automatic cleaning (duplicates, missing values, whitespace)
- Custom cleaning with natural language instructions
- Detailed cleaning reports

### 📐 Formula Suggestion
- Describe what you want to calculate
- Get Excel/spreadsheet formulas with explanations
- Alternative approaches and best practices

### 📈 Chart Generation
- Create visualizations from natural language descriptions
- Support for bar, line, scatter, pie, histogram, and box plots
- AI-powered column selection and configuration

## Architecture

```
ai_spreadsheet_agent/
│
├── backend/                 # FastAPI backend
│   ├── main.py             # API endpoints
│   ├── agent/              # LangChain agent logic
│   │   ├── __init__.py
│   │   └── spreadsheet_agent.py
│   ├── uploads/            # Uploaded files storage
│   └── requirements.txt
│
├── frontend/               # Streamlit frontend
│   ├── app.py             # UI application
│   └── requirements.txt
│
└── README.md
```

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- Anthropic API key

### 1. Clone or Create Project Structure

```bash
mkdir ai_spreadsheet_agent
cd ai_spreadsheet_agent
```

### 2. Set Up Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the `backend` directory:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

### 3. Set Up Frontend

```bash
cd ../frontend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Application

### Start Backend (Terminal 1)

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```

The API will be available at `http://localhost:8001`

### Start Frontend (Terminal 2)

```bash
cd frontend
source venv/bin/activate  # On Windows: venv\Scripts\activate
streamlit run app.py
```

The UI will open at `http://localhost:8501`

## Usage Guide

### 1. Upload Your Spreadsheet
- Click "Browse files" in the sidebar
- Upload CSV or Excel (.xlsx, .xls) file
- View dataset information and statistics

### 2. Natural Language Queries

Example queries:
- "What are the top 5 products by sales?"
- "Calculate the average price for each category"
- "Show me all rows where quantity is greater than 100"
- "What's the correlation between price and sales?"
- "Find duplicate customer emails"

### 3. Data Cleaning

**Automatic Cleaning:**
- Click "Auto Clean Data" for standard cleaning
- Removes duplicates, handles missing values, strips whitespace

**Custom Cleaning:**
- Provide specific instructions like:
  - "Remove rows where age is negative"
  - "Convert all email addresses to lowercase"
  - "Fill missing prices with the average price"

### 4. Formula Helper

Describe what you want to calculate:
- "Sum of sales for each month"
- "Calculate profit margin percentage"
- "Count unique customers per region"

Get:
- Excel formulas
- Step-by-step explanations
- Examples with your column names
- Alternative approaches

### 5. Chart Generation

Choose chart type and describe:
- "Show sales trends over the last 6 months"
- "Compare revenue by product category"
- "Distribution of customer ages"

## API Endpoints

### POST `/upload`
Upload a spreadsheet file
- **Request:** multipart/form-data with file
- **Response:** Session ID and dataset info

### POST `/query`
Natural language query on data
- **Body:** `{"session_id": "...", "query": "..."}`
- **Response:** Query results

### POST `/clean`
Clean spreadsheet data
- **Body:** `{"session_id": "...", "instructions": "..."}`
- **Response:** Cleaning summary

### POST `/suggest-formula`
Get formula suggestions
- **Body:** `{"session_id": "...", "description": "...", "context": "..."}`
- **Response:** Formula with explanations

### POST `/generate-chart`
Generate visualizations
- **Body:** `{"session_id": "...", "chart_type": "...", "description": "..."}`
- **Response:** Chart image and configuration

### GET `/preview/{session_id}`
Preview spreadsheet data
- **Response:** First N rows of data

### DELETE `/session/{session_id}`
Delete session and file
- **Response:** Confirmation message

## Technical Details

### LangChain Integration
- Uses `create_pandas_dataframe_agent` for natural language queries
- Tool-calling agent type for structured outputs
- Claude Sonnet 4 for intelligent processing

### Data Processing
- Pandas for data manipulation
- Matplotlib for chart generation
- Support for CSV and Excel formats

### Security Considerations
- File uploads are isolated per session
- Session-based access control
- Automatic cleanup on session deletion

## Troubleshooting

### Backend won't start
- Check that your `ANTHROPIC_API_KEY` is set in `.env`
- Ensure port 8000 is not in use
- Verify all dependencies are installed

### Frontend won't connect
- Ensure backend is running on port 8000
- Check `API_BASE_URL` in `frontend/app.py`
- Verify no firewall blocking

### Queries failing
- Check that file was uploaded successfully
- Verify API key is valid and has credits
- Review backend logs for errors

### Charts not generating
- Ensure matplotlib is installed correctly
- Check that data has appropriate columns for chart type
- Review column names and data types

## Future Enhancements

- [ ] Support for multiple sheet Excel files
- [ ] Export cleaned/modified data
- [ ] Scheduled data refreshes
- [ ] Custom visualization templates
- [ ] Collaborative features
- [ ] Data transformation pipelines
- [ ] Integration with Google Sheets
- [ ] Advanced statistical analysis
- [ ] Machine learning predictions
- [ ] Real-time data streaming

## Dependencies

### Backend
- FastAPI - Web framework
- LangChain - AI agent framework
- Anthropic - Claude AI API
- Pandas - Data manipulation
- Matplotlib - Visualization

### Frontend
- Streamlit - Web UI framework
- Requests - HTTP client

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - feel free to use this project for learning and development.

## Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation at `http://localhost:8000/docs`
- Open an issue on GitHub

---

Built with ❤️ using Claude AI, LangChain, FastAPI, and Streamlit