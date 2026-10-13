import os
import warnings
# Silence all warnings (including Pydantic/LangChain serialization warnings) for clean console logs
warnings.simplefilter("ignore")

# Force transformers/sentence-transformers to skip probing for a TensorFlow backend.
# On machines with TensorFlow installed, that eager import can hang for minutes
# (observed: absl/grpc mutex deadlock) -- we only ever use the PyTorch backend here.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, cast
from dotenv import load_dotenv

# Load env variables
load_dotenv()

from src.agents.supervisor_loop import Supervisor, RouteDecision
from src.api.redis_cache import SemanticCache

app = FastAPI(title="FIN-RAG Master Supervisor API", version="1.0.0")

# Initialize agent supervisor and semantic cache
supervisor = Supervisor()
cache = SemanticCache()

class ChatMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

class TraceStep(BaseModel):
    iteration: int
    action: str # "CALL_PDF_AGENT" | "CALL_SQL_AGENT" | "FINISH"
    instructions: str
    output: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    cache_hit: bool
    trace: List[TraceStep]

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    query = request.message
    
    # 1. Check Semantic Cache
    print(f"[API] Checking semantic cache for query: '{query}'")
    cached_answer = cache.check_cache(query)
    if cached_answer:
        return ChatResponse(
            answer=cached_answer,
            cache_hit=True,
            trace=[]
        )
    
    # 2. Run the Supervisor Loop with trace tracking
    print(f"[API] Cache MISS. Invoking Supervisor agent loop...")
    trace: List[TraceStep] = []
    
    # Format history for LangChain prompt (list of tuples)
    langchain_history = []
    for msg in request.history:
        role = "human" if msg.role == "user" else "ai"
        langchain_history.append((role, msg.content))
    
    iteration = 0
    max_iterations = 5
    final_answer = "System halted: Maximum iterations reached."
    
    while iteration < max_iterations:
        iteration += 1
        print(f"[API Loop] Iteration {iteration}")
        
        # Invoke Supervisor
        decision = cast(RouteDecision, supervisor.chain.invoke({
            "chat_history": langchain_history,
            "input": query
        }))
        
        step = TraceStep(
            iteration=iteration,
            action=decision.next_action,
            instructions=decision.instructions
        )
        
        if decision.next_action == "FINISH":
            final_answer = decision.instructions
            trace.append(step)
            break
            
        elif decision.next_action == "CALL_PDF_AGENT":
            result = supervisor.pdf_agent.invoke({"input": decision.instructions})
            agent_output = result.get("output", "No output generated.")
            
            # Extract raw tool output if available
            intermediate_steps = result.get("intermediate_steps", [])
            if intermediate_steps:
                tool_output = intermediate_steps[-1][1]
                step.output = tool_output
            else:
                step.output = agent_output
                
            langchain_history.append(("ai", f"I will call the PDF Specialist with instructions: {decision.instructions}"))
            langchain_history.append(("human", f"PDF Specialist result:\n{agent_output}"))
            
        elif decision.next_action == "CALL_SQL_AGENT":
            result = supervisor.sql_agent.invoke({"input": decision.instructions})
            agent_output = result.get("output", "No output generated.")
            
            # Extract raw tool output if available
            intermediate_steps = result.get("intermediate_steps", [])
            if intermediate_steps:
                tool_output = intermediate_steps[-1][1]
                step.output = tool_output
            else:
                step.output = agent_output
                
            langchain_history.append(("ai", f"I will call the SQL Specialist with instructions: {decision.instructions}"))
            langchain_history.append(("human", f"SQL Specialist result:\n{agent_output}"))
            
        trace.append(step)
        
    # 3. Store in Semantic Cache
    if final_answer and not final_answer.startswith("System halted"):
        cache.set_cache(query, final_answer)
        
    return ChatResponse(
        answer=final_answer,
        cache_hit=False,
        trace=trace
    )

# Mount the built React dashboard (falls back to the legacy static dashboard if not built)
react_dist_dir = os.path.join(os.path.dirname(__file__), "react_dist")
legacy_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(react_dist_dir):
    app.mount("/", StaticFiles(directory=react_dist_dir, html=True), name="static")
elif os.path.exists(legacy_static_dir):
    app.mount("/", StaticFiles(directory=legacy_static_dir, html=True), name="static")
else:
    print(f"Warning: No frontend build found at {react_dist_dir} or {legacy_static_dir}")
