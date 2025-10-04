from fastapi import APIRouter
from fastapi.responses import JSONResponse
from agents.mcp import MCPServerStreamableHttp
from agents import SQLiteSession
from chat_agents.agent import create_study_agent, run_agent
from config.settings import MCP_SERVER_URL

router = APIRouter()

@router.post("/chat")
async def chat(query: str,
               session_id: str):

    session = SQLiteSession(session_id)

    async with MCPServerStreamableHttp(
        name="StudyMode StreamableHttp Server",
        params={"url": MCP_SERVER_URL},
        cache_tools_list=True
    ) as mcp_server:
        try:
            # Create agent
            agent = create_study_agent(instructions=query, mcp_server=mcp_server, session=session)
            result = await run_agent(agent, session=session)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
