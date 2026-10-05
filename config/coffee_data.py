"""
coffee_data.py
--------------
Static metadata about each coffee type used to turn a raw ML prediction
label into a rich, personalized result for the UI.
"""

COFFEE_INFO = {
    "Espresso": {
        "emoji": "☕",
        "tagline": "Bold • Rich • Bitter",
        "description": (
            "Bold, concentrated and intense. A shot of pure coffee character "
            "with no dilution and no distractions."
        ),
        "best_for": "People who enjoy strong, direct coffee with an intense kick.",
        "personality_name": "The Bold Traditionalist",
        "personality_description": (
            "You don't need bells and whistles -- you want coffee that means "
            "business. Straightforward, intense, and always to the point."
        ),
    },
    "Americano": {
        "emoji": "☕",
        "tagline": "Smooth • Strong • Simple",
        "description": (
            "Espresso diluted with hot water for a stronger black coffee with "
            "a smoother, more relaxed intensity than a straight shot."
        ),
        "best_for": "People who enjoy a strong black coffee without the harsh edge.",
        "personality_name": "The Pure Minimalist",
        "personality_description": (
            "You like things simple and honest. No sugar, no fuss -- just good "
            "coffee, done right."
        ),
    },
    "Cappuccino": {
        "emoji": "☕",
        "tagline": "Balanced • Foamy • Classic",
        "description": (
            "Equal parts espresso, steamed milk, and foam -- a balanced, "
            "coffee-forward classic with a light, airy texture."
        ),
        "best_for": "People who enjoy a creamy but still coffee-forward experience.",
        "personality_name": "The Balanced Classic",
        "personality_description": (
            "You like a bit of everything -- structure and softness, tradition "
            "and comfort. Balance is your thing."
        ),
    },
    "Latte": {
        "emoji": "☕",
        "tagline": "Smooth • Creamy • Balanced",
        "description": (
            "Smooth espresso with plenty of steamed milk. Gentle, comforting, "
            "and easy to drink any time of day."
        ),
        "best_for": "People who prefer a mellow, creamy, and approachable coffee.",
        "personality_name": "The Smooth Explorer",
        "personality_description": (
            "You enjoy coffee that balances flavor and comfort. Easygoing, "
            "adaptable, and always up for something new."
        ),
    },
    "Mocha": {
        "emoji": "☕",
        "tagline": "Sweet • Chocolatey • Indulgent",
        "description": (
            "Coffee combined with chocolate and milk for a rich, dessert-like "
            "experience that satisfies a sweet tooth."
        ),
        "best_for": "People who enjoy sweet, indulgent, dessert-like flavors.",
        "personality_name": "The Sweet Indulgent",
        "personality_description": (
            "You believe coffee should be a treat, not just a habit. Rich, "
            "sweet, and a little bit indulgent -- just like you like it."
        ),
    },
    "Cold Brew": {
        "emoji": "🧊",
        "tagline": "Smooth • Refreshing • Cool",
        "description": (
            "Steeped slowly and served cold -- smooth, naturally less bitter, "
            "and refreshingly easy to drink."
        ),
        "best_for": "People who prefer chilled, refreshing coffee with a smooth edge.",
        "personality_name": "The Chill Strategist",
        "personality_description": (
            "You plan ahead and like things done right, not rushed. Cool, "
            "composed, and always a step ahead."
        ),
    },
    "Filter Coffee": {
        "emoji": "☕",
        "tagline": "Classic • Traditional • Honest",
        "description": (
            "A classic brewed coffee experience -- traditional, unfussy, and "
            "true to coffee's roots."
        ),
        "best_for": "People who enjoy traditional, no-frills coffee flavors.",
        "personality_name": "The Timeless Classic",
        "personality_description": (
            "You appreciate things that have stood the test of time. "
            "Dependable, grounded, and unafraid of tradition."
        ),
    },
}


def get_coffee_info(label: str) -> dict:
    """Return metadata for a coffee label, with a safe fallback."""
    return COFFEE_INFO.get(
        label,
        {
            "emoji": "☕",
            "tagline": "Distinct • Balanced",
            "description": "A coffee style that matches your unique preferences.",
            "best_for": "People with a distinct taste in coffee.",
            "personality_name": "The Coffee Individualist",
            "personality_description": "Your taste doesn't fit into just one box.",
        },
    )
