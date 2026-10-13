import os
import warnings
# Silence all warnings (including Pydantic/LangChain serialization warnings) for clean console logs
warnings.simplefilter("ignore")

# Force transformers/sentence-transformers to skip probing for a TensorFlow backend.
# On machines with TensorFlow installed, that eager import can hang for minutes
# (observed: absl/grpc mutex deadlock) -- we only ever use the PyTorch backend here.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Load environment variables (like OPENAI_API_KEY) BEFORE importing agents
load_dotenv()

from typing import cast
from src.agents.supervisor_loop import Supervisor, RouteDecision

def main():
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n[ERROR] OPENAI_API_KEY is not set!")
        print("Please create a .env file and add your OpenAI API key to run the AI Agents.")
        print("Example: OPENAI_API_KEY=sk-your-key-here\n")
        return

    print("\n" + "="*50)
    print("🚀 FIN-RAG MASTER SUPERVISOR INITIALIZED")
    print("="*50)
    print("Ask a financial question, and the Supervisor will route it")
    print("to the PDF Specialist or the SQL Specialist.")
    print("Type 'exit' to quit.\n")

    supervisor = Supervisor()
    chat_history = []

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            if not user_input.strip():
                continue

            print("\n[Supervisor is thinking...]")
            
            # The Supervisor loop determines the routing logic
            decision = cast(RouteDecision, supervisor.chain.invoke({
                "chat_history": chat_history,
                "input": user_input
            }))
            
            print(f"[Route Decision] -> {decision.next_action}")
            
            if decision.next_action == "CALL_PDF_AGENT":
                print(f"[PDF Agent] Searching Documents for: '{decision.instructions}'...")
                # Note: Because we are testing, this will call the PDF search tool.
                # If Qdrant/Postgres are not running, it will gracefully return an error string.
                result = supervisor.pdf_agent.invoke({"input": decision.instructions})
                final_answer = result["output"]
                
            elif decision.next_action == "CALL_SQL_AGENT":
                print(f"[SQL Agent] Writing SQL for: '{decision.instructions}'...")
                # Calls the SQL Executor tool
                result = supervisor.sql_agent.invoke({"input": decision.instructions})
                final_answer = result["output"]
                
            else: # FINISH
                final_answer = decision.instructions
            
            print(f"\n[Final Answer]:\n{final_answer}\n")
            
            # Append to history
            chat_history.append(("human", user_input))
            chat_history.append(("assistant", final_answer))
            
        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()
