import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import json
import os
from langchain_anthropic import ChatAnthropic
from pydantic.v1 import SecretStr

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import matplotlib.pyplot as plt
import io
import base64
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class SpreadsheetAgent:
    """AI agent for spreadsheet operations using LangChain and Claude"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = self._load_dataframe(file_path)
        self.original_df = self.df.copy()
        
        # Initialize Claude with API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.llm = ChatAnthropic(
                model_name="claude-sonnet-4-20250514",
                api_key=SecretStr(api_key),
                temperature=0,
                timeout=60
)



        
        # Create pandas dataframe agent
        try:
            self.pandas_agent = create_pandas_dataframe_agent(
                self.llm,
                self.df,
                agent_type="tool-calling",
                verbose=True,
                allow_dangerous_code=True,
                handle_parsing_errors=True
            )
            logger.info("✅ Pandas agent created successfully with Claude")
        except Exception as e:
            logger.error(f"❌ Error creating pandas agent: {str(e)}")
            raise
    
    def _load_dataframe(self, file_path: str) -> pd.DataFrame:
        """Load spreadsheet into pandas DataFrame"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Loaded dataframe with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise
    
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
            logger.info(f"🔍 Processing query: {query}")
            
            # Update the agent's dataframe reference
            self.pandas_agent = create_pandas_dataframe_agent(
                self.llm,
                self.df,
                agent_type="tool-calling",
                verbose=True,
                allow_dangerous_code=True,
                handle_parsing_errors=True
            )
            
            # Execute query
            response = self.pandas_agent.invoke({"input": query})
            
            logger.info(f"✅ Query successful")
            
            return {
                "success": True,
                "query": query,
                "response": response["output"],
                "type": "query_result"
            }
        
        except Exception as e:
            logger.error(f"❌ Error processing query: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def clean_data(self, instructions: Optional[str] = None) -> Dict[str, Any]:
        """Automatically clean and format spreadsheet data"""
        try:
            cleaning_steps = []
            initial_rows = len(self.df)
            
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
                
                # Ensure code is a string
                if isinstance(code, (list, dict)):
                    code = json.dumps(code) if isinstance(code, dict) else "\n".join(str(item) for item in code)
                
                logger.info(f"Generated cleaning code: {code[:200]}")
                
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
                
                # Remove duplicate rows
                duplicates = self.df.duplicated().sum()
                if duplicates > 0:
                    self.df = self.df.drop_duplicates()
                    cleaning_steps.append({
                        "type": "remove_duplicates",
                        "removed": int(duplicates)
                    })
                
                # Handle missing values
                for col in self.df.columns:
                    missing = self.df[col].isnull().sum()
                    if missing > 0:
                        if self.df[col].dtype in ['int64', 'float64']:
                            # Fill numeric with median
                            median_val = self.df[col].median()
                            self.df[col].fillna(median_val, inplace=True)
                            cleaning_steps.append({
                                "type": "fill_missing",
                                "column": col,
                                "method": "median",
                                "count": int(missing)
                            })
                        else:
                            # Fill categorical with mode or 'Unknown'
                            if not self.df[col].mode().empty:
                                mode_val = self.df[col].mode()[0]
                                self.df[col].fillna(mode_val, inplace=True)
                            else:
                                self.df[col].fillna('Unknown', inplace=True)
                            cleaning_steps.append({
                                "type": "fill_missing",
                                "column": col,
                                "method": "mode/unknown",
                                "count": int(missing)
                            })
                
                # Strip whitespace from string columns
                for col in self.df.select_dtypes(include=['object']).columns:
                    try:
                        self.df[col] = self.df[col].str.strip()
                        cleaning_steps.append({
                            "type": "strip_whitespace",
                            "column": col
                        })
                    except Exception as e:
                        logger.warning(f"Could not strip whitespace from {col}: {e}")
            
            return {
                "success": True,
                "cleaning_steps": cleaning_steps,
                "before_rows": initial_rows,
                "after_rows": len(self.df),
                "message": "Data cleaning completed successfully"
            }
        
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}", exc_info=True)
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
                content = response.content if isinstance(response.content, str) else str(response.content)
                # Remove markdown code blocks if present
                content = content.replace('```json', '').replace('```', '').strip()
                result = json.loads(content)
            except Exception as parse_error:
                logger.warning(f"Could not parse JSON response: {parse_error}")
                result = {
                    "formula": "See explanation",
                    "explanation": response.content if hasattr(response, 'content') else str(response),
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
            logger.error(f"Error suggesting formula: {str(e)}", exc_info=True)
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
            - x_column: column for x-axis (if applicable, otherwise null)
            - y_column: column(s) for y-axis
            - title: appropriate chart title
            - xlabel: x-axis label (if applicable)
            - ylabel: y-axis label
            - additional_params: any other parameters needed
            
            Respond with ONLY valid JSON, no other text or markdown.
            """
            
            response = self.llm.invoke(prompt)
            content = response.content if isinstance(response.content, str) else str(response)
            # Remove markdown code blocks if present
            content = content.replace('```json', '').replace('```', '').strip()
            chart_config = json.loads(content)
            
            logger.info(f"Chart config: {chart_config}")
            
            # Generate the chart
            plt.figure(figsize=(10, 6))
            
            chart_type_lower = chart_config.get('chart_type', '').lower()
            
            if chart_type_lower == 'bar':
                if chart_config.get('x_column'):
                    x_data = self.df[chart_config['x_column']].value_counts()
                    plt.bar(x_data.index.astype(str), x_data.values)
                else:
                    y_data = self.df[chart_config['y_column']].value_counts()
                    plt.bar(y_data.index.astype(str), y_data.values)
            
            elif chart_type_lower == 'line':
                plt.plot(self.df[chart_config['x_column']], 
                        self.df[chart_config['y_column']])
            
            elif chart_type_lower == 'scatter':
                plt.scatter(self.df[chart_config['x_column']], 
                           self.df[chart_config['y_column']])
            
            elif chart_type_lower == 'pie':
                data = self.df[chart_config['y_column']].value_counts()
                plt.pie(data.values, labels=data.index.astype(str), autopct='%1.1f%%')
            
            elif chart_type_lower in ['hist', 'histogram']:
                plt.hist(self.df[chart_config['y_column']].dropna(), bins=20)
            
            elif chart_type_lower == 'box':
                self.df.boxplot(column=chart_config['y_column'])
            
            plt.title(chart_config.get('title', 'Chart'))
            if chart_config.get('xlabel'):
                plt.xlabel(chart_config['xlabel'])
            if chart_config.get('ylabel'):
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
                "chart_type": chart_config.get('chart_type'),
                "config": chart_config,
                "image": f"data:image/png;base64,{image_base64}",
                "description": description
            }
        
        except Exception as e:
            logger.error(f"Error generating chart: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "description": description
            }