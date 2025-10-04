from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import Body
from agents.mcp import MCPServerStreamableHttp
from agents import SQLiteSession
from chat_agents.agent import create_study_agent, run_agent
from config.settings import MCP_SERVER_URL
from pydantic_schemas.schemas import ChatRequest  # import your Pydantic model

router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):
    session = SQLiteSession(request.session_id)

    async with MCPServerStreamableHttp(
        name="StudyMode StreamableHttp Server",
        params={"url": MCP_SERVER_URL},
        cache_tools_list=True
    ) as mcp_server:
        try:
            prompt_result = await mcp_server.get_prompt("prompt-v1")
        # print("prompt_result", prompt_result)
        # Extract the actual prompt text from the GetPromptResult object
            if prompt_result.messages and len(prompt_result.messages) > 0:
            # Get the first message's content
                first_message = prompt_result.messages[0]
                if hasattr(first_message.content, 'text'):
                    instruction_text = first_message.content.text
                elif isinstance(first_message.content, str):
                    instruction_text = first_message.content
                else:
                    instruction_text = str(first_message.content)
            else:
                instruction_text = "No prompt text found"
       
            # Create agent
            agent = create_study_agent(instructions=instruction_text, mcp_server=mcp_server, session=session)
            result = await run_agent(agent,query=request.query, session=session)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
