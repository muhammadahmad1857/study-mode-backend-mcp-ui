import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MCP_SERVER_URL = "http://127.0.0.1:5000/mcp"

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set")
