"""Grocery pricing, pantry tools, and parsing utilities for the Blackboard Agent."""

import re
from typing import Dict, List

# Catalog of grocery items with their standard prices
PRICES: Dict[str, float] = {
    "oats": 1.50,
    "bananas": 0.80,
    "peanut butter": 2.20,
    "milk": 2.50,
    "eggs": 2.00,
    "bread": 1.50,
    "rice": 1.20,
    "beans": 1.20,
    "lentils": 1.20,
    "chickpeas": 1.30,
    "tofu": 2.20,
    "potatoes": 1.20,
    "onions": 1.00,
    "carrots": 1.00,
    "garlic": 0.70,
    "spinach": 1.80,
    "broccoli": 1.80,
    "cabbage": 1.20,
    "tomatoes": 1.50,
    "tomato sauce": 1.40,
    "pasta": 1.20,
    "canned tuna": 1.80,
    "apples": 2.00,
    "cheese": 3.00,
    "butter": 2.80,
    "olive oil": 3.50,
    "chicken breast": 4.50,
    "ground beef": 5.00,
    "salmon": 6.00,
}


def price_of(item: str) -> float:
    """Find the price of an item using keyword matching. Default is $3.00."""
    item_clean = item.lower().strip()
    if not item_clean:
        return 3.00

    # Direct match
    if item_clean in PRICES:
        return PRICES[item_clean]

    # Substring match (e.g. 'canned black beans' matches 'beans')
    # Prefer longer matching keys for specificity (e.g., 'tomato sauce' over 'tomatoes')
    sorted_keys = sorted(PRICES.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in item_clean or item_clean in key:
            return PRICES[key]

    # Token-level match
    words = re.findall(r"\w+", item_clean)
    for word in words:
        # Check singular/plural variations
        for key in sorted_keys:
            if word == key or word.rstrip("s") == key.rstrip("s"):
                return PRICES[key]

    return 3.00


def parse_list(text: str) -> List[str]:
    """Extract item names from bulleted lines starting with '- '."""
    items: List[str] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Support '-' or '*' bullet points
        if line.startswith("- ") or line.startswith("* "):
            content = line[2:].strip()
            # Remove leading markdown bold/italic if any
            content = content.strip("*_`")
            if content:
                items.append(content)
        elif line.startswith("-") and len(line) > 1 and line[1] != "-":
            content = line[1:].strip().strip("*_`")
            if content:
                items.append(content)
    return items


def parse_budget(text: str, default: float = 30.0) -> float:
    """Parse budget amount from text like '$20', '20 dollars', '20 bucks'."""
    # Pattern 1: $20, $ 20.50
    m = re.search(r"\$\s*(\d+(?:\.\d+)?)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # Pattern 2: 20 dollars, 20 dollar, 20 bucks, 20 usd
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:dollars?|bucks?|usd)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # Pattern 3: budget: 20, budget of 20, under 20, max 20
    m = re.search(r"(?:budget|under|for|max|limit)\s*(?:of|is|:)?\s*\$?(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    return default


def pantry_text() -> str:
    """Return a formatted string of pantry items and prices."""
    lines = [f"- {item} (${price:.2f})" for item, price in sorted(PRICES.items())]
    return "\n".join(lines)
