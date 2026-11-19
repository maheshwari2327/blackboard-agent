# Blackboard Multi-Agent Meal Planner

A rule-based **Blackboard Architecture** implemented in Python using **LangGraph**, **LangChain**, and **Ollama** (`llama3.2`).

Specialist agents do **not** communicate directly with one another. Instead, they coordinate exclusively by reading and modifying a shared central **board**. A deterministic, rule-based **controller** inspects the board's current state on each turn and wakes whichever specialist can contribute next.

---

## Architecture Overview

```
                          ┌─────────────┐
                          │    START    │
                          └──────┬──────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
            ┌───────►│      controller       │◄────────┐
            │        │   (Rule-Based, No LLM)│         │
            │        └───────────┬───────────┘         │
            │                    │                     │
            │       conditional(rules | DONE)          │
            │                    │                     │
            │     ┌──────────────┼──────────────┬──────┴───────┐
            │     │              │              │              │
            │     ▼              ▼              ▼              ▼
            │ ┌──────────┐ ┌───────────┐  ┌──────────┐   ┌───────────┐
            ├─┤nutrition-│ │   chef    │  │ shopper  │   │  auditor  │
            │ │   ist    │ │   (LLM)   │  │  (LLM)   │   │  (No LLM) │
            │ └──────────┘ └───────────┘  └──────────┘   └───────────┘
            │                                                  │
            │                                                  │
            └──────────────────────────────────────────────────┘
                                 │ (next == "DONE")
                                 ▼
                              ┌─────┐
                              │ END │
                              └─────┘
```

### Specialists & Blackboard Rules

| Specialist | Type | Rule (Trigger Condition) | Action |
|---|---|---|---|
| **controller** | Code / No LLM | `steps >= 10` or first true specialist rule | Evaluates rules in priority order, schedules next specialist or `DONE`. |
| **nutritionist** | LLM | `not constraints` | Reads user request and budget; writes 3 dietary rules to `board["constraints"]`. |
| **chef** | LLM | `constraints and not menu` | Creates a 3-day menu using only priced pantry items. Listens to `board["feedback"]` if revising. |
| **shopper** | LLM + Code | `menu and not ingredients` | Extracts a deduplicated shopping list of ingredients and computes total `cost` from pantry prices. |
| **auditor** | Code / No LLM | `ingredients and not verdict` | If `cost > budget` and `revisions < 2`: erases `menu`/`ingredients`/`cost`/`verdict`, writes `feedback`, and increments `revisions`. Otherwise sets `verdict` to `APPROVED` or `OVER BUDGET`. |

---

## Project Structure

```
blackboard agent/
├── agent.py            # LangGraph StateGraph, prompts, nodes, rules, and CLI loop
├── tools.py            # PRICES catalog, price_of, parse_list, parse_budget, pantry_text
├── test_blackboard.py  # Comprehensive test suite (unit + end-to-end integration with mock LLM)
├── requirements.txt    # langchain, langchain-ollama, langgraph
└── README.md           # Documentation and usage guide
```

---

## System Prompts

### `NUTRITIONIST_PROMPT`
```
You are a NUTRITIONIST. From the request, write 3 short dietary rules for the meal plan
(one per line, starting with '- '). Return ONLY the 3 lines.
```

### `CHEF_PROMPT`
```
You are a CHEF. Plan a menu of 3 simple dishes (one per day) that follows the dietary rules.
Use ONLY ingredients from the PANTRY list and at most 4 ingredients per dish. Every ingredient has a price —
the WHOLE menu must stay inside the budget, so prefer cheap items.
Return ONLY this format, with no extra comments, no totals, no notes:
Day 1: <dish name>
- ingredient
- ingredient
Day 2: ...
If there is BUDGET FEEDBACK, the last menu was too expensive: use mostly items that cost $2.5 or less.
```

### `SHOPPER_PROMPT`
```
You are a SHOPPER. Read the menu and write ONE shopping list of every ingredient:
one simple ingredient per line starting with '- ', no quantities, no duplicates. Return ONLY the list.
```

---

## Installation & Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Ollama is running locally with llama3.2:**
   ```bash
   ollama pull llama3.2
   ollama run llama3.2
   ```

---

## Running the Application

### 1. Web Application (Streamlit)
To launch the interactive web dashboard with live blackboard streaming:
```bash
streamlit run app.py
```
This opens the web interface in your browser at `http://localhost:8501`.

### 2. Interactive CLI
```bash
python agent.py
```

**Example interaction:**
```
============================================================
 Blackboard Multi-Agent Meal Planner (LangGraph + Ollama)
============================================================
Stack: Ollama llama3.2 | Specialists communicate via shared board
Type 'exit' or 'quit' to end.

Meal request: High protein vegetarian meals for $20
[board] -> nutritionist
[board] -> chef
[board] -> shopper
[board] -> auditor

----------------------------------------
--- Final Meal Plan & Shopping List ---
----------------------------------------

[Menu]
Day 1: Tofu Scramble
- tofu
- spinach
- onions
- olive oil
Day 2: Lentil & Vegetable Bowl
- lentils
- carrots
- broccoli
- rice
Day 3: Chickpea Curry
- chickpeas
- tomatoes
- garlic
- onions

[Shopping List]
- tofu ($2.20)
- spinach ($1.80)
- onions ($1.00)
- olive oil ($3.50)
- lentils ($1.20)
- carrots ($1.00)
- broccoli ($1.80)
- rice ($1.20)
- chickpeas ($1.30)
- tomatoes ($1.50)
- garlic ($0.70)

[Budget & Verdict]
Total: $17.20 / Budget: $20.00
Verdict: APPROVED
----------------------------------------
```

### Running Tests
Execute the unit and multi-agent integration test suite:
```bash
python test_blackboard.py
```
