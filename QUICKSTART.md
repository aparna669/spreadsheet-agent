# Quick Start Guide - AI Spreadsheet Agent

## 🚀 Quick Setup (5 minutes)

### Step 1: Get Your Anthropic API Key
1. Visit https://console.anthropic.com/
2. Create an account or sign in
3. Go to API Keys section
4. Create a new API key and copy it

### Step 2: Set Up Backend
```bash
cd backend
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

Create `.env` file in `backend/` directory:
```
ANTHROPIC_API_KEY=your_actual_api_key_here
```

### Step 3: Set Up Frontend
```bash
cd frontend
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### Step 4: Run the Application

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python main.py
```
Wait for: `Uvicorn running on http://0.0.0.0:8000`

**Terminal 2 - Frontend:**
```bash
cd frontend
source venv/bin/activate  # or venv\Scripts\activate on Windows
streamlit run app.py
```
Browser will open automatically at `http://localhost:8501`

### Step 5: Test It!
```bash
# Optional: Run the test suite (in a third terminal)
python test_agent.py
```

## 🎯 First Tasks to Try

1. **Upload a spreadsheet** (CSV or Excel)
2. **Ask a question**: "What are the top 5 rows?"
3. **Clean data**: Click "Auto Clean Data"
4. **Get a formula**: "Calculate sum of column A"
5. **Create a chart**: "Bar chart of sales by category"

## 📝 Example Queries

### Data Analysis
- "What's the average value in column X?"
- "Show me rows where price > 100"
- "How many unique values are in column Y?"
- "What's the correlation between columns A and B?"

### Data Manipulation
- "Sort by sales descending"
- "Filter rows where status is 'Active'"
- "Group by category and sum sales"
- "Find duplicate emails"

### Calculations
- "Calculate total revenue"
- "What's the percentage of each category?"
- "Find the median price by region"
- "Show quarterly totals"

## 🔧 Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is in use
# On macOS/Linux:
lsof -i :8000

# On Windows:
netstat -ano | findstr :8000

# Kill the process if needed
```

### Missing dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### API key issues
- Make sure `.env` file is in `backend/` directory
- No spaces around the `=` sign
- No quotes around the API key
- Check that the key is valid at console.anthropic.com

### Frontend can't connect
- Ensure backend is running first
- Check that backend shows "Uvicorn running on http://0.0.0.0:8000"
- Try accessing http://localhost:8000/docs in your browser

## 🎨 Using the Interface

### Tab 1: Chat Query
Type natural language questions about your data. The AI will:
- Analyze your data
- Perform calculations
- Filter and sort
- Provide insights

### Tab 2: Data Cleaning
- **Auto Clean**: One-click cleanup
- **Custom**: Describe what you want cleaned

### Tab 3: Formula Helper
- Describe your calculation need
- Get Excel-compatible formulas
- See examples with your actual column names

### Tab 4: Chart Generator
- Select chart type
- Describe what to visualize
- Get publication-ready charts

### Tab 5: Data Preview
- View your current data
- See changes after cleaning
- Adjust number of rows displayed

## 📚 Learn More

- Full documentation: See `README.md`
- API documentation: http://localhost:8000/docs (when backend is running)
- Test examples: Run `python test_agent.py`

## 💡 Pro Tips

1. **Be specific**: Instead of "analyze this", try "what are the top 5 products by revenue?"
2. **Iterate**: Clean data first, then run queries for better results
3. **Save formulas**: Copy suggested formulas to use in Excel
4. **Export charts**: Right-click charts to save images
5. **Use context**: Reference column names in your queries

## 🆘 Get Help

- Backend API docs: http://localhost:8000/docs
- Check logs in terminal windows
- Review error messages carefully
- Ensure your data is in CSV or Excel format

Happy analyzing! 🎉