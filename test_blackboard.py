"""Unit and integration tests for Blackboard Multi-Agent Meal Planner."""

import unittest
from typing import Any, List
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from agent import AgentState, RULES, build_graph
from tools import parse_budget, parse_list, pantry_text, price_of, PRICES


class TestBlackboardTools(unittest.TestCase):
    def test_price_of(self):
        self.assertEqual(price_of("eggs"), 2.00)
        self.assertEqual(price_of("canned tuna"), 1.80)
        self.assertEqual(price_of("organic bananas"), 0.80)
        self.assertEqual(price_of("fresh broccoli"), 1.80)
        self.assertEqual(price_of("dragonfruit_unknown"), 3.00)

    def test_parse_list(self):
        text = """Here is the list:
- eggs
- milk
- bread
Day 2:
* rice
- black beans
"""
        items = parse_list(text)
        self.assertEqual(items, ["eggs", "milk", "bread", "rice", "black beans"])

    def test_parse_budget(self):
        self.assertEqual(parse_budget("Healthy food for $20"), 20.0)
        self.assertEqual(parse_budget("Budget $ 25.50 please"), 25.50)
        self.assertEqual(parse_budget("I have 15 dollars"), 15.0)
        self.assertEqual(parse_budget("under 18 bucks"), 18.0)
        self.assertEqual(parse_budget("Quick meals please"), 30.0)

    def test_pantry_text(self):
        text = pantry_text()
        self.assertIn("- eggs ($2.00)", text)
        self.assertIn("- rice ($1.20)", text)


class TestBlackboardAgentFlow(unittest.TestCase):
    def test_approved_first_try(self):
        """Test happy path: nutritionist -> chef -> shopper -> auditor approves."""
        fake_nutritionist = "- High fiber\n- Low sodium\n- Balanced protein"
        fake_chef = """Day 1: Oatmeal Bowl
- oats
- bananas
- peanut butter
Day 2: Scrambled Eggs & Bread
- eggs
- bread
Day 3: Rice & Beans
- rice
- beans"""
        fake_shopper = """- oats
- bananas
- peanut butter
- eggs
- bread
- rice
- beans"""

        fake_llm = FakeListChatModel(responses=[fake_nutritionist, fake_chef, fake_shopper])
        graph = build_graph(llm=fake_llm)

        initial_state: AgentState = {
            "board": {
                "request": "Healthy simple meals for $25",
                "budget": 25.0,
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
        board = final_state["board"]

        self.assertEqual(board["verdict"], "APPROVED")
        self.assertIsNotNone(board["constraints"])
        self.assertIsNotNone(board["menu"])
        self.assertEqual(len(board["ingredients"]), 7)
        self.assertLessEqual(board["cost"], 25.0)
        self.assertEqual(board["revisions"], 0)

    def test_revision_and_approval(self):
        """Test budget exceeded on attempt 1 -> revision -> approved on attempt 2."""
        fake_nutritionist = "- High protein\n- Under budget\n- Whole foods"
        # Round 1: expensive menu with salmon ($6) + ground beef ($5) + chicken breast ($4.5) + olive oil ($3.5) = $19 + ... > $15
        fake_chef_expensive = """Day 1: Salmon Feast
- salmon
- olive oil
Day 2: Steak & Veggies
- ground beef
- broccoli
Day 3: Chicken Bowl
- chicken breast
- rice"""
        fake_shopper_expensive = """- salmon
- olive oil
- ground beef
- broccoli
- chicken breast
- rice"""

        # Round 2: cheap menu after budget feedback
        fake_chef_cheap = """Day 1: Oatmeal
- oats
- bananas
Day 2: Scrambled Eggs
- eggs
- bread
Day 3: Rice & Beans
- rice
- beans"""
        fake_shopper_cheap = """- oats
- bananas
- eggs
- bread
- rice
- beans"""

        fake_llm = FakeListChatModel(responses=[
            fake_nutritionist,
            fake_chef_expensive,
            fake_shopper_expensive,
            fake_chef_cheap,
            fake_shopper_cheap,
        ])

        graph = build_graph(llm=fake_llm)

        initial_state: AgentState = {
            "board": {
                "request": "Protein rich meals for $15",
                "budget": 15.0,
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
        board = final_state["board"]

        self.assertEqual(board["verdict"], "APPROVED")
        self.assertEqual(board["revisions"], 1)
        self.assertLessEqual(board["cost"], 15.0)

    def test_over_budget_after_max_revisions(self):
        """Test cost > budget after 2 revisions -> OVER BUDGET verdict."""
        fake_nutritionist = "- High protein\n- Low carb\n- Keto"
        fake_chef_expensive = """Day 1: Salmon
- salmon
- ground beef
Day 2: Chicken
- chicken breast
- olive oil
Day 3: Beef
- ground beef
- cheese"""
        fake_shopper_expensive = """- salmon
- ground beef
- chicken breast
- olive oil
- cheese"""

        # 3 rounds of expensive (initial + 2 revisions)
        fake_llm = FakeListChatModel(responses=[
            fake_nutritionist,
            fake_chef_expensive, fake_shopper_expensive,  # rev 0 -> cost > $10 -> rev 1
            fake_chef_expensive, fake_shopper_expensive,  # rev 1 -> cost > $10 -> rev 2
            fake_chef_expensive, fake_shopper_expensive,  # rev 2 -> cost > $10 -> OVER BUDGET
        ])

        graph = build_graph(llm=fake_llm)

        initial_state: AgentState = {
            "board": {
                "request": "Keto meals for $10",
                "budget": 10.0,
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
        board = final_state["board"]

        self.assertEqual(board["verdict"], "OVER BUDGET")
        self.assertEqual(board["revisions"], 2)

    def test_controller_max_steps_safeguard(self):
        """Test controller terminates when steps reaches 10."""
        # LLM that keeps returning without satisfying full conditions
        fake_llm = FakeListChatModel(responses=["- rule 1", "- rule 2", "- rule 3"])
        graph = build_graph(llm=fake_llm)

        initial_state: AgentState = {
            "board": {
                "request": "Random request",
                "budget": 30.0,
                "constraints": None,
                "menu": None,
                "ingredients": None,
                "cost": None,
                "verdict": None,
                "feedback": None,
                "revisions": 0,
            },
            "next": None,
            "steps": 10,  # Already at max steps
        }

        final_state = graph.invoke(initial_state)
        self.assertEqual(final_state["next"], "DONE")
        self.assertEqual(final_state["steps"], 10)


if __name__ == "__main__":
    unittest.main()
