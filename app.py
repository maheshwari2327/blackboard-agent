"""Streamlit Web UI for the Blackboard Multi-Agent Meal Planner."""

import os
import streamlit as st
from langchain_ollama import ChatOllama

from agent import build_graph, AgentState, RULES
from tools import parse_budget, parse_list, pantry_text, price_of, PRICES

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Blackboard Multi-Agent Meal Planner",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .reportview-container {
        background: #fdfdfd;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2b8a3e;
    }
    .board-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-controller { background-color: #e9ecef; color: #495057; }
    .badge-nutritionist { background-color: #d3f9d8; color: #2b8a3e; }
    .badge-chef { background-color: #fff3bf; color: #f08c00; }
    .badge-shopper { background-color: #d0ebff; color: #1971c2; }
    .badge-auditor { background-color: #ffe3e3; color: #e03131; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar: Configuration & Pantry
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    
    ollama_model = st.text_input("Ollama Model", value=os.getenv("OLLAMA_MODEL", "llama3.2"))
    ollama_url = st.text_input("Ollama Base URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    
    st.divider()
    
    st.subheader("💡 Example Requests")
    examples = [
        "High protein vegetarian meals for $20",
        "Cheap college student meals under $15",
        "Quick family dinners for $25 with chicken",
        "Budget vegan lunches for $18",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["meal_request_input"] = ex

    st.divider()
    
    with st.expander("🏪 Pantry Catalog & Prices", expanded=False):
        st.caption("Pantry items available to the Chef:")
        pantry_data = [{"Ingredient": item, "Price ($)": f"{price:.2f}"} for item, price in sorted(PRICES.items())]
        st.dataframe(pantry_data, use_container_width=True, height=300)

    st.markdown("---")
    st.caption("🤖 **Architecture:** Blackboard Multi-Agent Pattern\n"
               "- Rule-based Controller (No LLM)\n"
               "- 3 LLM Specialists (Nutritionist, Chef, Shopper)\n"
               "- 1 Deterministic Auditor (No LLM)")


# -----------------------------------------------------------------------------
# Main Interface
# -----------------------------------------------------------------------------
st.title("🍳 Blackboard Multi-Agent Meal Planner")
st.markdown(
    "Specialist agents do **not** communicate directly. They read and write strictly to a "
    "**shared Blackboard**, orchestrated by a rule-based **Controller**."
)

col1, col2 = st.columns([3, 1])

default_prompt = st.session_state.get("meal_request_input", "High protein vegetarian meals for $20")

with col1:
    user_request = st.text_area(
        "Meal Request",
        value=default_prompt,
        height=95,
        placeholder="E.g., High protein vegetarian meals for $20",
        help="Specify dietary preferences, ingredients, or budget.",
    )

with col2:
    detected_budget = parse_budget(user_request, default=30.0)
    manual_budget = st.number_input(
        "Budget ($)",
        min_value=5.0,
        max_value=150.0,
        value=float(detected_budget),
        step=1.0,
        help="Budget is auto-detected from your request, or you can override it here.",
    )

run_button = st.button("🚀 Run Blackboard Planner", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Blackboard Execution
# -----------------------------------------------------------------------------
if run_button:
    if not user_request.strip():
        st.warning("Please enter a meal request first.")
    else:
        try:
            llm = ChatOllama(model=ollama_model, base_url=ollama_url, temperature=0.2)
            graph = build_graph(llm=llm)

            initial_state: AgentState = {
                "board": {
                    "request": user_request.strip(),
                    "budget": float(manual_budget),
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

            st.write("---")
            st.subheader("📋 Live Blackboard Activity")

            status_placeholder = st.empty()
            activity_container = st.container()

            final_board = dict(initial_state["board"])
            step_logs = []

            with status_placeholder.status("Coordinating specialists on the blackboard...", expanded=True) as status_box:
                for chunk in graph.stream(initial_state):
                    for node_name, node_data in chunk.items():
                        if node_name == "controller":
                            next_agent = node_data.get("next")
                            current_steps = node_data.get("steps")
                            step_logs.append(f"🎯 **Controller** scheduled: `{next_agent}` (Step {current_steps})")
                            st.markdown(f"<span class='board-badge badge-controller'>Controller</span> Woke up: **{next_agent.upper()}** (Step {current_steps})", unsafe_allow_html=True)

                        elif node_name == "nutritionist":
                            final_board = node_data.get("board", final_board)
                            step_logs.append("🥗 **Nutritionist** posted dietary rules to board.")
                            st.markdown("<span class='board-badge badge-nutritionist'>Nutritionist</span> Wrote dietary constraints to board:", unsafe_allow_html=True)
                            st.code(final_board.get("constraints", ""), language="text")

                        elif node_name == "chef":
                            final_board = node_data.get("board", final_board)
                            step_logs.append("👨‍🍳 **Chef** posted 3-day menu to board.")
                            st.markdown("<span class='board-badge badge-chef'>Chef</span> Drafted 3-day menu on board:", unsafe_allow_html=True)
                            st.code(final_board.get("menu", ""), language="text")

                        elif node_name == "shopper":
                            final_board = node_data.get("board", final_board)
                            step_logs.append(f"🛒 **Shopper** parsed ingredients and calculated cost: ${final_board.get('cost', 0.0):.2f}")
                            st.markdown(f"<span class='board-badge badge-shopper'>Shopper</span> Created shopping list. Total Cost: **${final_board.get('cost', 0.0):.2f}**", unsafe_allow_html=True)

                        elif node_name == "auditor":
                            final_board = node_data.get("board", final_board)
                            verdict = final_board.get("verdict")
                            cost = final_board.get("cost")
                            revisions = final_board.get("revisions", 0)

                            if verdict:
                                st.markdown(f"<span class='board-badge badge-auditor'>Auditor</span> Verdict: **{verdict}**", unsafe_allow_html=True)
                                step_logs.append(f"⚖️ **Auditor** finalized verdict: {verdict}")
                            else:
                                st.markdown(f"<span class='board-badge badge-auditor'>Auditor</span> ⚠️ Cost exceeded budget! Erasing menu and requesting revision #{revisions}...", unsafe_allow_html=True)
                                st.info(f"Feedback to Chef: {final_board.get('feedback')}")
                                step_logs.append(f"⚖️ **Auditor** requested Revision #{revisions}")

                status_box.update(label="Planning Complete!", state="complete", expanded=False)

            # -----------------------------------------------------------------
            # Results Presentation
            # -----------------------------------------------------------------
            st.divider()
            st.subheader("🎯 Final Results")

            final_cost = final_board.get("cost", 0.0) or 0.0
            budget_val = final_board.get("budget", float(manual_budget))
            verdict_val = final_board.get("verdict", "COMPLETED")
            revisions_val = final_board.get("revisions", 0)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Budget", f"${budget_val:.2f}")
            m2.metric("Total Cost", f"${final_cost:.2f}", delta=f"{budget_val - final_cost:+.2f}")
            m3.metric("Revisions", revisions_val)
            m4.metric("Verdict", verdict_val)

            tab_menu, tab_shopping, tab_board, tab_trace = st.tabs([
                "🍽️ 3-Day Menu",
                "🛒 Shopping List",
                "📋 Raw Blackboard State",
                "📜 Step Trace"
            ])

            with tab_menu:
                st.markdown("### 3-Day Menu Plan")
                st.text(final_board.get("menu") or "No menu generated.")
                if final_board.get("constraints"):
                    st.markdown("#### Dietary Constraints Applied:")
                    st.text(final_board.get("constraints"))

            with tab_shopping:
                st.markdown("### Itemized Shopping List")
                ingredients = final_board.get("ingredients") or []
                if ingredients:
                    table_rows = []
                    for ing in ingredients:
                        price = price_of(ing)
                        table_rows.append({"Item": ing, "Estimated Price ($)": f"{price:.2f}"})
                    st.table(table_rows)
                    st.markdown(f"**Total Estimated Cost:** `${final_cost:.2f}` (Budget: `${budget_val:.2f}`)")
                else:
                    st.write("No ingredients found.")

            with tab_board:
                st.markdown("### Current Blackboard Dictionary")
                st.json(final_board)

            with tab_trace:
                st.markdown("### Execution History")
                for log in step_logs:
                    st.write(log)

        except Exception as err:
            st.error(f"Error executing blackboard agent: {err}")
            st.info(
                "Make sure Ollama is running and has the model pulled:\n"
                f"1. `ollama pull {ollama_model}`\n"
                f"2. `ollama run {ollama_model}`"
            )
