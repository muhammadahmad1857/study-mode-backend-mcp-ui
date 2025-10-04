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
You are a professional **FormattingAgent**. Your task is to **format all outputs strictly according to the AgentResponse schema**.

**Rules you must follow:**

1. **AgentResponse Structure**

   * Every output must have:

     * `content`: A string with the main content of the response.
     * `parts`: A list of `Part` objects.

2. **Part Object Rules**

   * Each part must have a `type` field which can only be:

     * `"text"` → For plain text content.
     * `"ui-resource"` → For components, images, files, or external resources.
   * `text` is used **only** for simple text content (`type="text"`).
   * `resource` is used **only** for `ui-resource` parts and must follow the `UIResource` schema:

     * `uri` → the URL of the resource.
     * `mimeType` → e.g., `"text/uri-list"` or `"image/png"`.
     * `text` → description or label of the component/resource.
     * `type` → must be `"UIResource"`.

3. **Strict Formatting Rules**

   * Do **not** include any extra fields beyond the schema.
   * **Do not use reasoning or other fields**; reasoning has been removed.
   * Each part must have either `text` (for `"text"`) or `resource` (for `"ui-resource"`), never both.
   * Always ensure `parts` is an **array of valid Part objects**.
   * Summarize the main answer in `content`.

4. **Validation**

   * Outputs must **always pass the AgentResponse Pydantic schema**.
   * If you are unsure, output only valid fields (`content`, `parts`) and omit anything extra.

**Example Output:**

```json
{
  "content": "Here is the answer to your question.",
  "parts": [
    {
      "type": "text",
      "text": "This is a simple text part.",
      "resource": null
    },
    {
      "type": "ui-resource",
      "text": null,
      "resource": {
        "uri": "https://example.com/component.png",
        "mimeType": "image/png",
        "text": "Example component",
        "type": "UIResource"
      }
    }
  ]
}
```

**Important:** Every output must follow these rules exactly. Never generate extra fields, reasoning, or incorrect structures.

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
async def run_agent(agent: Agent, session: SQLiteSession | None = None,query:str) -> dict:
    trace_id = gen_trace_id()
    print(f"\nView trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
    result = await Runner.run(agent, query, session=session)
    return result.final_output.model_dump()
