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
import logging

from agent.spreadsheet_agent import SpreadsheetAgent
from dotenv import load_dotenv
import os

load_dotenv()  # This loads the .env file

print("API KEY:", os.getenv("ANTHROPIC_API_KEY"))  # Temporary test


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Health check endpoint"""
    return {"message": "AI Spreadsheet Agent API", "status": "running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a spreadsheet file and create a session"""
    try:
        # Validate file exists and has correct extension
        if not file.filename:
            raise HTTPException(400, "No file provided")
            
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(400, "Only CSV and Excel files are supported")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Save file
        file_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
        
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                if not content:
                    raise HTTPException(400, "Empty file uploaded")
                f.write(content)
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise HTTPException(500, f"Error saving file: {str(e)}")
        
        # Create agent session
        try:
            agent = SpreadsheetAgent(str(file_path))
            sessions[session_id] = agent
        except Exception as e:
            # Clean up file if agent creation fails
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.error(f"Error creating agent: {str(e)}")
            raise HTTPException(500, f"Error processing file: {str(e)}")
        
        # Get basic info
        df_info = agent.get_dataframe_info()
        
        logger.info(f"File uploaded successfully. Session: {session_id}")
        
        return {
            "session_id": session_id,
            "filename": file.filename,
            "info": df_info,
            "message": "File uploaded successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload: {str(e)}")
        raise HTTPException(500, f"Error uploading file: {str(e)}")


@app.post("/query")
async def query_data(request: QueryRequest):
    """Process natural language query on spreadsheet data"""
    try:
        if request.session_id not in sessions:
            raise HTTPException(404, "Session not found")
        
        if not request.query or not request.query.strip():
            raise HTTPException(400, "Query cannot be empty")
        
        agent = sessions[request.session_id]
        result = agent.process_query(request.query)
        
        logger.info(f"Query processed for session {request.session_id}")
        
        return {
            "session_id": request.session_id,
            "query": request.query,
            "result": result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(500, f"Error processing query: {str(e)}")


@app.post("/clean")
async def clean_data(request: CleanDataRequest):
    """Automatically clean and format spreadsheet data"""
    try:
        if request.session_id not in sessions:
            raise HTTPException(404, "Session not found")
        
        agent = sessions[request.session_id]
        result = agent.clean_data(request.instructions)
        
        logger.info(f"Data cleaned for session {request.session_id}")
        
        return {
            "session_id": request.session_id,
            "result": result,
            "message": "Data cleaning completed"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise HTTPException(500, f"Error cleaning data: {str(e)}")


@app.post("/suggest-formula")
async def suggest_formula(request: FormulaRequest):
    """Suggest Excel/spreadsheet formulas based on description"""
    try:
        if request.session_id not in sessions:
            raise HTTPException(404, "Session not found")
        
        if not request.description or not request.description.strip():
            raise HTTPException(400, "Description cannot be empty")
        
        agent = sessions[request.session_id]
        result = agent.suggest_formula(request.description, request.context)
        
        logger.info(f"Formula suggested for session {request.session_id}")
        
        return {
            "session_id": request.session_id,
            "description": request.description,
            "result": result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error suggesting formula: {str(e)}")
        raise HTTPException(500, f"Error suggesting formula: {str(e)}")


@app.post("/generate-chart")
async def generate_chart(request: ChartRequest):
    """Generate chart/visualization based on description"""
    try:
        if request.session_id not in sessions:
            raise HTTPException(404, "Session not found")
        
        if not request.description or not request.description.strip():
            raise HTTPException(400, "Description cannot be empty")
        
        valid_chart_types = ["bar", "line", "scatter", "pie", "histogram", "box"]
        if request.chart_type not in valid_chart_types:
            raise HTTPException(400, f"Invalid chart type. Must be one of: {', '.join(valid_chart_types)}")
        
        agent = sessions[request.session_id]
        result = agent.generate_chart(request.chart_type, request.description)
        
        logger.info(f"Chart generated for session {request.session_id}")
        
        return {
            "session_id": request.session_id,
            "chart_type": request.chart_type,
            "result": result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating chart: {str(e)}")
        raise HTTPException(500, f"Error generating chart: {str(e)}")


@app.get("/preview/{session_id}")
async def preview_data(session_id: str, rows: int = 10):
    """Preview the current state of the spreadsheet"""
    try:
        if session_id not in sessions:
            raise HTTPException(404, "Session not found")
        
        # Validate rows parameter
        if rows < 1 or rows > 100:
            raise HTTPException(400, "Rows must be between 1 and 100")
        
        agent = sessions[session_id]
        preview = agent.get_preview(rows)
        
        return {
            "session_id": session_id,
            "preview": preview
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error previewing data: {str(e)}")
        raise HTTPException(500, f"Error previewing data: {str(e)}")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its associated file"""
    try:
        if session_id not in sessions:
            raise HTTPException(404, "Session not found")
        
        # Remove from sessions
        agent = sessions.pop(session_id)
        
        # Delete file
        try:
            if os.path.exists(agent.file_path):
                os.remove(agent.file_path)
                logger.info(f"Deleted file for session {session_id}")
        except Exception as e:
            logger.warning(f"Could not delete file: {str(e)}")
        
        return {"message": "Session deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(500, f"Error deleting session: {str(e)}")


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    try:
        session_list = []
        for sid, agent in sessions.items():
            try:
                info = agent.get_dataframe_info()
                session_list.append({
                    "session_id": sid,
                    "info": info
                })
            except Exception as e:
                logger.warning(f"Error getting info for session {sid}: {str(e)}")
                continue
        
        return {"sessions": session_list}
    
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(500, f"Error listing sessions: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("AI Spreadsheet Agent API starting up...")
    logger.info(f"Upload directory: {UPLOAD_DIR.absolute()}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("AI Spreadsheet Agent API shutting down...")
    # Clean up any remaining sessions
    for session_id in list(sessions.keys()):
        try:
            agent = sessions.pop(session_id)
            if os.path.exists(agent.file_path):
                os.remove(agent.file_path)
        except Exception as e:
            logger.warning(f"Error cleaning up session {session_id}: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # For production, use: host="0.0.0.0"
    # For local development, can use: host="127.0.0.1"
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Listen on all interfaces
        port=8001,        # Port number
        log_level="info"  # Logging level
    )