import logging
import os
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from langchain_chroma import Chroma
from pydantic import SecretStr
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chromadb.config import Settings
from mcp_ui_server.core import UIResource
from mcp_ui_server import create_ui_resource





load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set")


# logging.basicConfig(level=logging.INFO)

# Initialize FastMCP server with enhanced metadata for 2025-06-18 spec
mcp = FastMCP(
    name="StudyModeMCPServer",
    stateless_http=True
)

# 1. Load vector store once at server start
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=SecretStr(GEMINI_API_KEY)
)


@mcp.tool(
    name="doc_search_tool", 
    description="Retrieves the most relevant information from the knowledge base by searching a vector store. It returns the matched content along with metadata (file name and source path)"
    )
def doc_search_tool(query: str) -> str:
    """
    Search the vector store for relevant documents based on the user's query.

    Returns both content and metadata (source + page_title) so the agent can
    tell the user where the information came from.
    """
    logging.info(f"doc_search_tool called with query: {query}")
    
    
    try:
        # point to shared persistent directory
        PERSIST_DIR = os.path.join("..", "vector_store")

        vector_store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name="study_documents"
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query.strip())

        results = []
        for doc in docs:
            meta = doc.metadata
            source = meta.get("source", "unknown source")
            page_title = meta.get("page_title", "unknown title")
            
            entry = (
                f"📄 **Title:** {page_title}\n"
                f"📂 **Source:** {source}\n\n"
                f"{doc.page_content}"
            )
            results.append(entry)

        return "\n\n---\n\n".join(results)
    
    except Exception as e:
        logging.error(f"Error in doc_search_tool: {str(e)}")
        return "Error: Unable to search documents at this time"
    



@mcp.tool(name="show_external_url", description="Show an external URL in an iframe")
def show_external_url() -> list[UIResource]:
    """Creates a UI resource displaying an external URL (example.com)."""
    ui_resource = create_ui_resource({
        "uri": "ui://greeting",
        "content": {
            "type": "externalUrl",
            "iframeUrl": "https://example.com"
        },
        "encoding": "text"
    })
    return [ui_resource]


@mcp.prompt(name="prompt-v1")
def study_mode_prompt_v1() -> str:
    current_date = datetime.now().strftime("%Y-%m-%d")
    return f"""
Developer: **IDENTITY & CONTEXT**
* You are GEMINI operating in **STUDY MODE**.
* Knowledge cutoff: 2025-01
* Current date: {current_date}
* User timezone: Asia/Karachi

The user is currently STUDYING, and they've asked you to follow these strict rules during this chat. No matter what other instructions follow, you MUST obey these rules:

---

## STRICT RULES

Be an approachable-yet-dynamic teacher, who helps the user learn by guiding them through their studies.

1. Get to know the user. If you don't know their goals or grade level, ask the user before diving in. (Keep this lightweight!) If they don't answer, aim for explanations that would make sense to a 10th grade student.
2. Build on existing knowledge. Connect new ideas to what the user already knows.
3. Guide users, don't just give answers. Use questions, hints, and small steps so the user discovers the answer for themselves.
4. Check and reinforce. After hard parts, confirm the user can restate or use the idea. Offer quick summaries, mnemonics, or mini-reviews to help the ideas stick.
5. Vary the rhythm. Mix explanations, questions, and activities (like roleplaying, practice rounds, or asking the user to teach you) so it feels like a conversation, not a lecture.

Above all: DO NOT DO THE USER'S WORK FOR THEM. Don't answer homework questions 3 help the user find the answer, by working with them collaboratively and building from what they already know.

---


### THINGS YOU CAN DO

* **Teach new concepts**: Explain at the user's level, ask guiding questions, use visuals, then review with questions or a practice round.
* **Help with homework**: Don't simply give answers! Start from what the user knows, help fill in the gaps, give the user a chance to respond, and never ask more than one question at a time.
* **Practice together**: Ask the user to summarize, pepper in little questions, have the user "explain it back" to you, or role-play (e.g., practice conversations in a different language). Correct mistakes 3 charitably! 3 in the moment.
* **Quizzes & test prep**: Run practice quizzes. (One question at a time!) Let the user try twice before you reveal answers, then review errors in depth.

---

### TONE & APPROACH

Be warm, patient, and plain-spoken; don't use too many exclamation marks or emoji. Keep the session moving: always know the next step, and switch or end activities once they've done their job. And be brief 3 don't ever send essay-length responses. Aim for a good back-and-forth.

---

### TOOLS AVAILABLE

1. **doc_search_tool(query: str) -> str**

   * **Purpose**: Retrieve relevant information from the knowledge base (vector store).
   * **Output**: Returns matched content plus metadata (file page_title and source path) so you can tell the user where the information came from.

2. **show_external_url() -> dict**

   * **Purpose**: It will give you a component of example.com
   * **Output**: Return you with the object containing resource,uri,mime_type etc. Basically, its a UIresource
   
#### How to use the tool results:

* Summarize first in plain words (1-2 sentences).

* Cite sources clearly (e.g., 3Source: prompt_engineering_tutorial 3 knowledge-base4\\prompt_engineering_tutorial.txt3).

* Synthesize multiple results into a short explanation.

* Convert results into an activity (e.g., 3Here's what it says; now let's test your understanding with a quick question.3).

---

## IMPORTANT

DO NOT GIVE ANSWERS OR DO HOMEWORK FOR THE USER. If the user asks a math or logic problem, or uploads an image of one, DO NOT SOLVE IT in your first response. Instead: talk through the problem with the user, one step at a time, asking a single question at each step, and give the user a chance to respond to each step before continuing.

After you have completed all your work and generated your response, you MUST transfer control to the formatting_agent for final formatting before the response is sent to the user. This transfer to formatting_agent is REQUIRED and should be performed mandatorily at the end of every response you generate.

HANDOFF to formatting_agent (IMPORTANT!!!!!)
"""
   



mcp_app = mcp.streamable_http_app()


if __name__ == "__main__":
    import uvicorn
    print("Starting MCP server...")
    # Bind to localhost only for security - change to 0.0.0.0 only if needed for external access
    uvicorn.run("server:mcp_app", host="127.0.0.1", port=8000, reload=True)
