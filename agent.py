"""Blackboard Multi-Agent Meal Planner using LangChain, LangGraph, and Ollama."""

import os
import sys
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from tools import parse_budget, parse_list, pantry_text, price_of

# -----------------------------------------------------------------------------
# System Prompts
# -----------------------------------------------------------------------------

NUTRITIONIST_PROMPT = """You are a NUTRITIONIST. From the request, write 3 short dietary rules for the meal plan
(one per line, starting with '- '). Return ONLY the 3 lines."""

CHEF_PROMPT = """You are a CHEF. Plan a menu of 3 simple dishes (one per day) that follows the dietary rules.
Use ONLY ingredients from the PANTRY list and at most 4 ingredients per dish. Every ingredient has a price —
the WHOLE menu must stay inside the budget, so prefer cheap items.
Return ONLY this format, with no extra comments, no totals, no notes:
Day 1: <dish name>
- ingredient
- ingredient
Day 2: ...
If there is BUDGET FEEDBACK, the last menu was too expensive: use mostly items that cost $2.5 or less."""

SHOPPER_PROMPT = """You are a SHOPPER. Read the menu and write ONE shopping list of every ingredient:
one simple ingredient per line starting with '- ', no quantities, no duplicates. Return ONLY the list."""


# -----------------------------------------------------------------------------
# State Definition
# -----------------------------------------------------------------------------

class AgentState(TypedDict):
    board: Dict[str, Any]
    next: Optional[str]
    steps: int


# -----------------------------------------------------------------------------
# Blackboard Rules
# -----------------------------------------------------------------------------

RULES: Dict[str, Callable[[Dict[str, Any]], bool]] = {
    "nutritionist": lambda b: not b.get("constraints"),
    "chef": lambda b: bool(b.get("constraints")) and not bool(b.get("menu")),
    "shopper": lambda b: bool(b.get("menu")) and not bool(b.get("ingredients")),
    "auditor": lambda b: bool(b.get("ingredients")) and not bool(b.get("verdict")),
}


# -----------------------------------------------------------------------------
# Node Implementations
# -----------------------------------------------------------------------------

def get_llm():
    """Create the Ollama LLM instance."""
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(model=model_name, base_url=base_url, temperature=0.2)


def create_nodes(llm=None):
    """Factory to create graph node functions, allowing LLM injection for testing."""
    active_llm = llm or get_llm()

    def controller(state: AgentState) -> Dict[str, Any]:
        steps = state.get("steps", 0)
        board = state.get("board", {})

        if steps >= 10:
            return {"next": "DONE", "steps": steps}

        for name, rule in RULES.items():
            if rule(board):
                return {"next": name, "steps": steps + 1}

        return {"next": "DONE", "steps": steps}

    def nutritionist(state: AgentState) -> Dict[str, Any]:
        print("[board] -> nutritionist")
        board = dict(state["board"])
        prompt = ChatPromptTemplate.from_messages([
            ("system", NUTRITIONIST_PROMPT),
            ("human", "Meal Request: {request}\nBudget: ${budget:.2f}"),
        ])
        chain = prompt | active_llm | StrOutputParser()
        constraints = chain.invoke({
            "request": board.get("request", ""),
            "budget": board.get("budget", 30.0),
        })
        board["constraints"] = constraints.strip()
        return {"board": board}

    def chef(state: AgentState) -> Dict[str, Any]:
        print("[board] -> chef")
        board = dict(state["board"])

        human_parts = [
            f"Dietary Rules:\n{board.get('constraints', '')}",
            f"Budget: ${board.get('budget', 30.0):.2f}",
            f"PANTRY:\n{pantry_text()}",
        ]
        if board.get("feedback"):
            human_parts.append(f"BUDGET FEEDBACK:\n{board['feedback']}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", CHEF_PROMPT),
            ("human", "\n\n".join(human_parts)),
        ])
        chain = prompt | active_llm | StrOutputParser()
        menu = chain.invoke({})
        board["menu"] = menu.strip()
        return {"board": board}

    def shopper(state: AgentState) -> Dict[str, Any]:
        print("[board] -> shopper")
        board = dict(state["board"])

        prompt = ChatPromptTemplate.from_messages([
            ("system", SHOPPER_PROMPT),
            ("human", "Menu:\n{menu}"),
        ])
        chain = prompt | active_llm | StrOutputParser()
        raw_list = chain.invoke({"menu": board.get("menu", "")})

        parsed = parse_list(raw_list)
        if not parsed:
            parsed = [
                line.strip().lstrip("-*•0123456789. ")
                for line in raw_list.splitlines()
                if line.strip()
            ]

        # Deduplicate while preserving order
        unique_ingredients: List[str] = []
        seen = set()
        for item in parsed:
            clean = item.strip().lower()
            if clean and clean not in seen:
                seen.add(clean)
                unique_ingredients.append(item.strip())

        total_cost = sum(price_of(item) for item in unique_ingredients)
        board["ingredients"] = unique_ingredients
        board["cost"] = round(total_cost, 2)
        return {"board": board}

    def auditor(state: AgentState) -> Dict[str, Any]:
        print("[board] -> auditor")
        board = dict(state["board"])
        cost = board.get("cost", 0.0) or 0.0
        budget = board.get("budget", 30.0)
        revisions = board.get("revisions", 0)

        if cost > budget and revisions < 2:
            board["feedback"] = (
                f"Total menu cost (${cost:.2f}) exceeds the budget (${budget:.2f}). "
                f"Please choose cheaper pantry items (preferably $2.50 or less)."
            )
            # Erase menu, ingredients, verdict, cost per blackboard specification
            board["menu"] = None
            board["ingredients"] = None
            board["verdict"] = None
            board["cost"] = None
            board["revisions"] = revisions + 1
        else:
            if cost <= budget:
                board["verdict"] = "APPROVED"
            else:
                board["verdict"] = "OVER BUDGET"

        return {"board": board}

    return controller, nutritionist, chef, shopper, auditor


# -----------------------------------------------------------------------------
# Graph Construction
# -----------------------------------------------------------------------------

def build_graph(llm=None):
    """Build and compile the blackboard StateGraph."""
    controller, nutritionist, chef, shopper, auditor = create_nodes(llm=llm)

    builder = StateGraph(AgentState)

    builder.add_node("controller", controller)
    builder.add_node("nutritionist", nutritionist)
    builder.add_node("chef", chef)
    builder.add_node("shopper", shopper)
    builder.add_node("auditor", auditor)

    builder.add_edge(START, "controller")

    def route_controller(state: AgentState) -> str:
        target = state.get("next")
        if not target or target == "DONE":
            return END
        return target

    builder.add_conditional_edges(
        "controller",
        route_controller,
        {
            "nutritionist": "nutritionist",
            "chef": "chef",
            "shopper": "shopper",
            "auditor": "auditor",
            END: END,
        },
    )

    builder.add_edge("nutritionist", "controller")
    builder.add_edge("chef", "controller")
    builder.add_edge("shopper", "controller")
    builder.add_edge("auditor", "controller")

    return builder.compile()


# -----------------------------------------------------------------------------
# CLI Loop
# -----------------------------------------------------------------------------

def run_cli():
    """Interactive CLI loop for the Blackboard Meal Planner."""
    print("=" * 60)
    print(" Blackboard Multi-Agent Meal Planner (LangGraph + Ollama)")
    print("=" * 60)
    print("Stack: Ollama llama3.2 | Specialists communicate via shared board")
    print("Type 'exit' or 'quit' to end.\n")

    graph = build_graph()

    while True:
        try:
            req = input("Meal request: ").strip()
            if not req or req.lower() in ("exit", "quit", "q"):
                print("Exiting. Goodbye!")
                break

            budget = parse_budget(req, default=30.0)

            initial_state: AgentState = {
                "board": {
                    "request": req,
                    "budget": budget,
                    "constraints": None,
                    "menu": None,
                    "ingredients": None,
                    "cost": None,
                    "verdict": None,
                    "feedback": None,
                    "revisions": 0,
                },
                "next": None,
                "steps": 0,
            }

            final_state = graph.invoke(initial_state)
            board = final_state.get("board", {})

            print("\n" + "-" * 40)
            print("--- Final Meal Plan & Shopping List ---")
            print("-" * 40)
            print("\n[Menu]")
            print(board.get("menu") or "No menu generated.")

            print("\n[Shopping List]")
            ingredients = board.get("ingredients") or []
            if ingredients:
                for item in ingredients:
                    p = price_of(item)
                    print(f"- {item} (${p:.2f})")
            else:
                print("No ingredients found.")

            print("\n[Budget & Verdict]")
            cost = board.get("cost", 0.0) or 0.0
            print(f"Total: ${cost:.2f} / Budget: ${board.get('budget', 0.0):.2f}")
            print(f"Verdict: {board.get('verdict', 'UNKNOWN')}")
            revisions = board.get("revisions", 0)
            if revisions > 0:
                print(f"Revisions: {revisions}")
            print("-" * 40 + "\n")

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\n[Error] {e}\n")


if __name__ == "__main__":
    run_cli()
