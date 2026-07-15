import os

MOSS_API_URL = os.environ.get("MOSS_API_URL", "http://localhost:8000")
MUCKROCK_API_URL = os.environ.get("MUCKROCK_API_URL", "https://www.muckrock.com/api_v2")

SQUARELET_BASE = os.environ.get("SQUARELET_BASE", "https://accounts.muckrock.com")
SQUARELET_CLIENT_ID = os.environ.get("SQUARELET_CLIENT_ID", "")
SQUARELET_CLIENT_SECRET = os.environ.get("SQUARELET_CLIENT_SECRET", "")

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8001")
MCP_PORT = int(os.environ.get("PORT", "8001"))
