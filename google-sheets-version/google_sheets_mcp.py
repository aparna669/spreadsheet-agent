"""
Google Sheets MCP Wrapper
Direct integration with Google Sheets API using MCP (Model Context Protocol)
"""
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import gspread
from gspread_dataframe import set_with_dataframe
from langchain_core.tools import Tool

import pandas as pd
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

try:
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    import gspread
    from gspread_dataframe import set_with_dataframe
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    print("Google API libraries not installed. Run: pip install -r requirements_google.txt")


class GoogleSheetsMCP:
    """
    Model Context Protocol wrapper for Google Sheets API
    Provides standardized interface for spreadsheet operations
    """
    
    def __init__(self, credentials_path: Optional[str] = None, credentials_dict: Optional[dict] = None):
        """
        Initialize Google Sheets MCP client
        
        Args:
            credentials_path: Path to service account JSON file
            credentials_dict: Service account credentials as dictionary
        """
        if not GOOGLE_API_AVAILABLE:
            raise ImportError(
                "Google API libraries not installed. "
                "Run: pip install google-auth google-api-python-client gspread gspread-dataframe"
            )
        
        self.credentials = None
        self.service = None
        self.gc = None  # gspread client
        
        if credentials_path:
            self.credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/drive'
                ]
            )
        elif credentials_dict:
            self.credentials = service_account.Credentials.from_service_account_info(
                credentials_dict,
                scopes=[
                    'https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/drive'
                ]
            )
        
        if self.credentials:
            self.service = build('sheets', 'v4', credentials=self.credentials)
            self.gc = gspread.authorize(self.credentials)
    
    def create_spreadsheet(
        self, 
        title: str, 
        data: Optional[pd.DataFrame] = None,
        share_with: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new Google Spreadsheet
        
        Args:
            title: Spreadsheet title
            data: Optional DataFrame to populate
            share_with: List of email addresses to share with
        
        Returns:
            Dict containing spreadsheet info
        """
        try:
            if not self.service:
                return {
                    'success': False,
                    'error': 'Service not initialized',
                    'message': '❌ Google Sheets service not initialized'
                }
            
            # Create spreadsheet
            spreadsheet = {
                'properties': {
                    'title': title
                }
            }
            
            spreadsheet = self.service.spreadsheets().create(
                body=spreadsheet,
                fields='spreadsheetId,spreadsheetUrl'
            ).execute()
            
            spreadsheet_id = spreadsheet.get('spreadsheetId')
            spreadsheet_url = spreadsheet.get('spreadsheetUrl')
            
            # Populate with data if provided
            if data is not None and not data.empty:
                self.write_dataframe(spreadsheet_id, data, 'Sheet1')
            
            # Share with users if specified
            if share_with:
                for email in share_with:
                    self._share_spreadsheet(spreadsheet_id, email)
            
            return {
                'success': True,
                'spreadsheet_id': spreadsheet_id,
                'spreadsheet_url': spreadsheet_url,
                'title': title,
                'rows': len(data) if data is not None else 0,
                'columns': len(data.columns) if data is not None else 0,
                'message': f"✅ Created spreadsheet: {title}"
            }
            
        except HttpError as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"❌ Failed to create spreadsheet: {str(e)}"
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"❌ Error: {str(e)}"
            }
    
    def read_spreadsheet(
        self, 
        spreadsheet_id: str, 
        sheet_name: str = 'Sheet1',
        range_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Read data from Google Spreadsheet
        
        Args:
            spreadsheet_id: Google Sheets ID
            sheet_name: Sheet name to read
            range_name: Specific range (e.g., 'A1:D10')
        
        Returns:
            Dict with DataFrame
        """
        try:
            if not self.gc:
                return {
                    'success': False,
                    'error': 'Client not initialized',
                    'data': pd.DataFrame(),
                    'message': '❌ gspread client not initialized'
                }
            
            # Use gspread for easier DataFrame conversion
            spreadsheet = self.gc.open_by_key(spreadsheet_id)
            worksheet = spreadsheet.worksheet(sheet_name)
            
            # Get all values
            values = worksheet.get_all_values()
            
            if not values:
                return {
                    'success': True,
                    'data': pd.DataFrame(),
                    'message': "⚠️ Sheet is empty"
                }
            
            # Convert to DataFrame (first row as header)
            df = pd.DataFrame(values[1:], columns=values[0])
            
            # Try to infer numeric types
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
            
            return {
                'success': True,
                'data': df,
                'spreadsheet_id': spreadsheet_id,
                'sheet_name': sheet_name,
                'rows': len(df),
                'columns': len(df.columns),
                'message': f"✅ Read {len(df)} rows from {sheet_name}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': pd.DataFrame(),
                'message': f"❌ Failed to read spreadsheet: {str(e)}"
            }
    
    def write_dataframe(
        self, 
        spreadsheet_id: str, 
        data: pd.DataFrame, 
        sheet_name: str = 'Sheet1',
        start_cell: str = 'A1'
    ) -> Dict[str, Any]:
        """
        Write DataFrame to Google Spreadsheet
        
        Args:
            spreadsheet_id: Google Sheets ID
            data: DataFrame to write
            sheet_name: Sheet name
            start_cell: Starting cell (default A1)
        
        Returns:
            Dict with result
        """
        try:
            if not self.gc:
                return {
                    'success': False,
                    'error': 'Client not initialized',
                    'message': '❌ gspread client not initialized'
                }
            
            spreadsheet = self.gc.open_by_key(spreadsheet_id)
            
            # Get or create worksheet
            try:
                worksheet = spreadsheet.worksheet(sheet_name)
            except:
                worksheet = spreadsheet.add_worksheet(
                    title=sheet_name, 
                    rows=len(data) + 1, 
                    cols=len(data.columns)
                )
            
            # Write DataFrame
            set_with_dataframe(worksheet, data, include_index=False, include_column_header=True)
            
            return {
                'success': True,
                'spreadsheet_id': spreadsheet_id,
                'sheet_name': sheet_name,
                'rows_written': len(data),
                'columns_written': len(data.columns),
                'message': f"✅ Wrote {len(data)} rows to {sheet_name}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"❌ Failed to write data: {str(e)}"
            }
    
    def append_dataframe(
        self, 
        spreadsheet_id: str, 
        data: pd.DataFrame, 
        sheet_name: str = 'Sheet1'
    ) -> Dict[str, Any]:
        """
        Append DataFrame to existing data
        
        Args:
            spreadsheet_id: Google Sheets ID
            data: DataFrame to append
            sheet_name: Sheet name
        
        Returns:
            Dict with result
        """
        try:
            if not self.gc:
                return {
                    'success': False,
                    'error': 'Client not initialized',
                    'message': '❌ gspread client not initialized'
                }
            
            spreadsheet = self.gc.open_by_key(spreadsheet_id)
            worksheet = spreadsheet.worksheet(sheet_name)
            
            # Convert DataFrame to list of lists
            values = data.values.tolist()
            
            # Append rows
            worksheet.append_rows(values)
            
            return {
                'success': True,
                'spreadsheet_id': spreadsheet_id,
                'sheet_name': sheet_name,
                'rows_appended': len(data),
                'message': f"✅ Appended {len(data)} rows to {sheet_name}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"❌ Failed to append data: {str(e)}"
            }
    
    def update_cell(
        self, 
        spreadsheet_id: str, 
        cell: str, 
        value: Any, 
        sheet_name: str = 'Sheet1'
    ) -> Dict[str, Any]:
        """
        Update a single cell
        
        Args:
            spreadsheet_id: Google Sheets ID
            cell: Cell reference (e.g., 'A1')
            value: Value to set
            sheet_name: Sheet name
        
        Returns:
            Dict with result
        """
        try:
            if not self.gc:
                return {
                    'success': False,
                    'error': 'Client not initialized',
                    'message': '❌ gspread client not initialized'
                }
            
            spreadsheet = self.gc.open_by_key(spreadsheet_id)
            worksheet = spreadsheet.worksheet(sheet_name)
            
            worksheet.update(cell, value)
            
            return {
                'success': True,
                'spreadsheet_id': spreadsheet_id,
                'cell': cell,
                'value': value,
                'message': f"✅ Updated {cell} = {value}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"❌ Failed to update cell: {str(e)}"
            }
    
    def list_sheets(self, spreadsheet_id: str) -> Dict[str, Any]:
        """
        List all sheets in a spreadsheet
        
        Args:
            spreadsheet_id: Google Sheets ID
        
        Returns:
            Dict with sheet names
        """
        try:
            if not self.gc:
                return {
                    'success': False,
                    'error': 'Client not initialized',
                    'message': '❌ gspread client not initialized'
                }
            
            spreadsheet = self.gc.open_by_key(spreadsheet_id)
            worksheets = spreadsheet.worksheets()
            
            sheet_info = [
                {
                    'title': ws.title,
                    'id': ws.id,
                    'row_count': ws.row_count,
                    'col_count': ws.col_count
                }
                for ws in worksheets
            ]
            
            return {
                'success': True,
                'spreadsheet_id': spreadsheet_id,
                'sheets': sheet_info,
                'count': len(sheet_info),
                'message': f"✅ Found {len(sheet_info)} sheets"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"❌ Failed to list sheets: {str(e)}"
            }
    
    def create_sheet(
        self, 
        spreadsheet_id: str, 
        sheet_name: str, 
        rows: int = 1000, 
        cols: int = 26
    ) -> Dict[str, Any]:
        """
        Create a new sheet in existing spreadsheet
        
        Args:
            spreadsheet_id: Google Sheets ID
            sheet_name: Name for new sheet
            rows: Number of rows
            cols: Number of columns
        
        Returns:
            Dict with result
        """
        try:
            if not self.gc:
                return {
                    'success': False,
                    'error': 'Client not initialized',
                    'message': '❌ gspread client not initialized'
                }
            
            spreadsheet = self.gc.open_by_key(spreadsheet_id)
            worksheet = spreadsheet.add_worksheet(
                title=sheet_name, 
                rows=rows, 
                cols=cols
            )
            
            return {
                'success': True,
                'spreadsheet_id': spreadsheet_id,
                'sheet_name': sheet_name,
                'sheet_id': worksheet.id,
                'message': f"✅ Created sheet: {sheet_name}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"❌ Failed to create sheet: {str(e)}"
            }
    
    def delete_sheet(self, spreadsheet_id: str, sheet_name: str) -> Dict[str, Any]:
        """
        Delete a sheet from spreadsheet
        
        Args:
            spreadsheet_id: Google Sheets ID
            sheet_name: Sheet name to delete
        
        Returns:
            Dict with result
        """
        try:
            if not self.gc:
                return {
                    'success': False,
                    'error': 'Client not initialized',
                    'message': '❌ gspread client not initialized'
                }
            
            spreadsheet = self.gc.open_by_key(spreadsheet_id)
            worksheet = spreadsheet.worksheet(sheet_name)
            spreadsheet.del_worksheet(worksheet)
            
            return {
                'success': True,
                'spreadsheet_id': spreadsheet_id,
                'sheet_name': sheet_name,
                'message': f"✅ Deleted sheet: {sheet_name}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"❌ Failed to delete sheet: {str(e)}"
            }
    
    def _share_spreadsheet(self, spreadsheet_id: str, email: str, role: str = 'writer'):
        """
        Share spreadsheet with email address
        
        Args:
            spreadsheet_id: Google Sheets ID
            email: Email address to share with
            role: Permission role (reader, writer, owner)
        """
        try:
            if self.gc:
                spreadsheet = self.gc.open_by_key(spreadsheet_id)
                spreadsheet.share(email, perm_type='user', role=role)
        except Exception as e:
            print(f"Warning: Could not share with {email}: {str(e)}")
    
    def get_spreadsheet_info(self, spreadsheet_id: str) -> Dict[str, Any]:
        """
        Get spreadsheet metadata
        
        Args:
            spreadsheet_id: Google Sheets ID
        
        Returns:
            Dict with spreadsheet info
        """
        try:
            if not self.service:
                return {
                    'success': False,
                    'error': 'Service not initialized',
                    'message': '❌ Service not initialized'
                }
            
            spreadsheet = self.service.spreadsheets().get(
                spreadsheetId=spreadsheet_id
            ).execute()
            
            return {
                'success': True,
                'spreadsheet_id': spreadsheet_id,
                'title': spreadsheet['properties']['title'],
                'locale': spreadsheet['properties'].get('locale'),
                'time_zone': spreadsheet['properties'].get('timeZone'),
                'sheet_count': len(spreadsheet['sheets']),
                'url': f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit",
                'message': "✅ Retrieved spreadsheet info"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"❌ Failed to get spreadsheet info: {str(e)}"
            }


# Convenience functions for LangChain integration
def create_langchain_google_sheets_tools(mcp_client: GoogleSheetsMCP):
    """
    Create LangChain tools from GoogleSheetsMCP client
    
    Args:
        mcp_client: Initialized GoogleSheetsMCP client
    
    Returns:
        List of LangChain Tool objects
    """
    try:
        tools = [
            Tool(
                name="create_google_spreadsheet",
                func=lambda x: mcp_client.create_spreadsheet(x),
                description="Create a new Google Spreadsheet. Input: spreadsheet title"
            ),
            Tool(
                name="read_google_spreadsheet",
                func=lambda x: mcp_client.read_spreadsheet(x),
                description="Read data from Google Spreadsheet. Input: spreadsheet ID"
            ),
            Tool(
                name="list_sheets",
                func=lambda x: mcp_client.list_sheets(x),
                description="List all sheets in a spreadsheet. Input: spreadsheet ID"
            ),
            Tool(
                name="get_spreadsheet_info",
                func=lambda x: mcp_client.get_spreadsheet_info(x),
                description="Get spreadsheet metadata. Input: spreadsheet ID"
            )
        ]
        
        return tools
    except ImportError:
        print("LangChain not available - tools not created")
        return []