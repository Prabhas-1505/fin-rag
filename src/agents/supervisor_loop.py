import warnings
# Silence all warnings (including Pydantic/LangChain serialization warnings) for clean console logs
warnings.simplefilter("ignore")

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, cast
import os
from dotenv import load_dotenv

load_dotenv()

from src.agents.specialists import create_pdf_specialist, create_sql_specialist
from src.llm.provider import get_chat_llm
import json

class RouteDecision(BaseModel):
    next_action: Literal["CALL_PDF_AGENT", "CALL_SQL_AGENT", "FINISH"] = Field(
        description="The next routing action to take."
    )
    instructions: str = Field(
        description="If calling an agent, specific instructions for that agent. If FINISH, the final answer to the user."
    )

class Supervisor:
    def __init__(self):
        # We use strict structured output to ensure the LLM never hallucinates a bad route
        llm = get_chat_llm()
        self.structured_llm = llm.with_structured_output(RouteDecision)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are the Master Supervisor routing requests to specialist agents. "
             "You have two specialist agents: \n"
             "1. PDF Research Specialist: Answers questions using facts, figures, and quotes found inside "
             "the ingested documents (e.g. reports, papers) via hybrid document search. Route here for ANY "
             "question about content, findings, numbers, or claims made within a document, even if the "
             "document is not financial in nature.\n"
             "2. SQL Analytics Specialist: Answers questions ONLY about the structured company/financial "
             "database (tables: companies, financial_metrics — tickers, revenue, net income, EPS). Route here "
             "only when the question is clearly about that structured database, not about document content.\n\n"
             "If in doubt between the two, prefer the PDF Research Specialist — it is the general-purpose "
             "document question-answering path.\n\n"
             "Analyze the user's request and the current conversation history. Decide if you need to route to an agent "
             "or if you have enough information to FINISH and provide the final answer."
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")
        ])
        
        self.chain = self.prompt | self.structured_llm
        
        self.pdf_agent = create_pdf_specialist()
        self.sql_agent = create_sql_specialist()

    def run(self, user_input: str, max_iterations: int = 5) -> str:
        """
        Deterministic state loop executing the multi-agent routing.
        """
        chat_history = []
        iteration = 0
        
        print(f"\n[Supervisor] Received request: '{user_input}'\n")
        
        while iteration < max_iterations:
            iteration += 1
            print(f"--- Iteration {iteration} ---")
            
            # 1. Ask Supervisor for decision
            decision = cast(RouteDecision, self.chain.invoke({
                "chat_history": chat_history,
                "input": user_input
            }))
            
            print(f"[Supervisor Decision] Action: {decision.next_action}")
            print(f"[Supervisor Decision] Instructions: {decision.instructions}\n")
            
            # 2. Execute routing decision
            if decision.next_action == "FINISH":
                return decision.instructions
                
            elif decision.next_action == "CALL_PDF_AGENT":
                print("[Executing] Calling PDF Specialist...")
                result = self.pdf_agent.invoke({"input": decision.instructions})
                output = result.get("output", "No output generated.")
                chat_history.append(("ai", f"I will call the PDF Specialist with instructions: {decision.instructions}"))
                chat_history.append(("human", f"PDF Specialist result:\n{output}"))
                print(f"[PDF Output] {output}\n")
                
            elif decision.next_action == "CALL_SQL_AGENT":
                print("[Executing] Calling SQL Specialist...")
                result = self.sql_agent.invoke({"input": decision.instructions})
                output = result.get("output", "No output generated.")
                chat_history.append(("ai", f"I will call the SQL Specialist with instructions: {decision.instructions}"))
                chat_history.append(("human", f"SQL Specialist result:\n{output}"))
                print(f"[SQL Output] {output}\n")
                
        return "System halted: Maximum iterations reached."

if __name__ == "__main__":
    print("Supervisor initialized.")
    # Example manual usage:
    # supervisor = Supervisor()
    # final_answer = supervisor.run("What was Apple's Q4 revenue compared to its historical data?")
    # print("FINAL ANSWER:\n", final_answer)
