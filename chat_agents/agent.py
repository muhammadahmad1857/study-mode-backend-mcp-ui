from agents import Agent, OpenAIChatCompletionsModel, AsyncOpenAI, Runner, SQLiteSession, gen_trace_id
from agents.mcp import MCPServer
from pydantic_schemas.schemas import AgentResponse
from config.settings import GEMINI_API_KEY

# Initialize OpenAI client
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# ======================
# Formatting Agent
# ======================
formatting_agent = Agent(
    name="FormattingAgent",
    instructions="""
You are a professional FormattingAgent. Format all outputs strictly according to the AgentResponse schema.
Refer to the schema rules: text, reasoning, ui-resource, content summarization, no extra fields.
""",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",
        openai_client=client,
    ),
    output_type=AgentResponse,
)

# ======================
# Function to create main StudyMode agent
# ======================
def create_study_agent(instructions: str, mcp_server: MCPServer, session: SQLiteSession | None = None):
    agent = Agent(
        name="StudyMode",
        instructions=instructions,
        model=OpenAIChatCompletionsModel(
            model="gemini-2.5-flash",
            openai_client=client,
        ),
        mcp_servers=[mcp_server],
        handoffs=[formatting_agent],
    )
    return agent

# ======================
# Runner helper
# ======================
async def run_agent(agent: Agent, session: SQLiteSession | None = None) -> dict:
    trace_id = gen_trace_id()
    print(f"\nView trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
    result = await Runner.run(agent, "User query executed", session=session)
    return result.final_output.model_dump()
