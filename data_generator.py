"""
data_generator.py
-----------------
Generates a realistic, diverse review dataset (600 samples, 200 per class).
Avoids template repetition through combinatorial variation across:
  - sentence structures
  - product domains (electronics, food, clothing, service, software)
  - intensity modifiers
  - contextual phrases

This ensures the model is learning generalizable sentiment geometry,
not memorizing surface-level patterns from tiny repeated templates.

Run:
    python data_generator.py
Output:
    data/reviews.csv  (600 rows, columns: review, label)
"""

import csv
import random
import os
from itertools import product

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# VOCABULARY BANKS
# ─────────────────────────────────────────────────────────────────────────────

DOMAINS = {
    'electronics': [
        "laptop", "phone", "headphones", "camera", "tablet", "keyboard",
        "monitor", "speaker", "smartwatch", "charger", "router", "earbuds"
    ],
    'food': [
        "restaurant", "meal", "food", "dish", "coffee", "pizza", "burger",
        "dessert", "sushi", "pasta", "salad", "sandwich"
    ],
    'clothing': [
        "jacket", "shirt", "shoes", "dress", "jeans", "coat", "sneakers",
        "hoodie", "suit", "bag", "hat", "boots"
    ],
    'service': [
        "customer service", "support team", "delivery", "shipping",
        "staff", "experience", "service", "team", "response", "help desk"
    ],
    'software': [
        "app", "software", "tool", "platform", "interface", "subscription",
        "update", "feature", "plugin", "dashboard", "API", "product"
    ]
}

# Flatten all items
ALL_ITEMS = [item for items in DOMAINS.values() for item in items]

POSITIVE_TEMPLATES = [
    "Absolutely love this {item}. {detail}. Would buy again.",
    "The {item} exceeded every expectation I had. {detail}.",
    "This is hands down the best {item} I have ever owned. {detail}.",
    "Incredibly impressed with this {item}. {detail}. Five stars easily.",
    "Outstanding {item}. {detail}. Highly recommend to everyone.",
    "My {item} arrived quickly and works flawlessly. {detail}.",
    "Cannot believe how good this {item} is for the price. {detail}.",
    "Genuinely amazed by the quality of this {item}. {detail}.",
    "So happy with my purchase. The {item} is {adj_pos} and {adj_pos2}.",
    "Best decision I made this year was buying this {item}. {detail}.",
    "Wow, this {item} blew me away. {detail}. Will definitely recommend.",
    "Solid build quality and {adj_pos} performance. This {item} delivers.",
    "I was skeptical at first but this {item} truly delivered. {detail}.",
    "The {item} is even better than advertised. {detail}.",
    "Top tier {item}. {detail}. You will not be disappointed.",
    "Worth every penny. This {item} is {adj_pos} and {adj_pos2}.",
    "I bought this {item} as a gift and the recipient loved it. {detail}.",
    "Exceptional {item}. {detail}. Ordering another one for sure.",
    "Just got my {item} and I am beyond satisfied. {detail}.",
    "Premium feel, premium results. This {item} is {adj_pos} in every way.",
]

NEGATIVE_TEMPLATES = [
    "Terrible {item}. {detail}. Complete waste of money.",
    "This {item} broke within {timeframe}. {detail}. Never buying again.",
    "Worst {item} I have ever purchased. {detail}. Avoid at all costs.",
    "Extremely disappointed with this {item}. {detail}.",
    "Do not buy this {item}. {detail}. Absolute garbage.",
    "The {item} stopped working after {timeframe}. {detail}. Unacceptable.",
    "False advertising. This {item} is nothing like described. {detail}.",
    "Returned this {item} immediately. {detail}. Total scam.",
    "I cannot believe this {item} is sold legally. {detail}. Disgraceful.",
    "The {item} is {adj_neg} and {adj_neg2}. {detail}. Very upset.",
    "Horrible experience. The {item} failed to do what it promised. {detail}.",
    "Cheap build quality and {adj_neg} results. This {item} is a disaster.",
    "Paid premium price for a {adj_neg} {item}. {detail}. Outrageous.",
    "This {item} damaged my property. {detail}. Completely unacceptable.",
    "Ordered the {item} for a special occasion. It arrived broken. {detail}.",
    "Customer service was unhelpful when my {item} malfunctioned. {detail}.",
    "The {item} is {adj_neg} and falls apart immediately. {detail}.",
    "Save your money. This {item} is {adj_neg} junk. {detail}.",
    "Regret buying this {item} every single day. {detail}.",
    "Zero stars if I could. The {item} is {adj_neg} and {adj_neg2}. {detail}.",
]

NEUTRAL_TEMPLATES = [
    "The {item} is okay. {detail}. Nothing special but gets the job done.",
    "Average {item}. {detail}. Does what it is supposed to, nothing more.",
    "Not bad, not great. The {item} is a middle-of-the-road product. {detail}.",
    "The {item} meets basic expectations. {detail}. Would not rave about it.",
    "Decent enough {item} for the price. {detail}. Fine if you are on a budget.",
    "The {item} works as described. {detail}. Nothing to write home about.",
    "I have mixed feelings about this {item}. {detail}. Some good, some bad.",
    "Mediocre {item}. {detail}. There are probably better options out there.",
    "The {item} is functional but unremarkable. {detail}.",
    "Acceptable quality for the price. The {item} is {adj_neu}. {detail}.",
    "The {item} does the job. {detail}. Would consider other options next time.",
    "Received my {item} on time. {detail}. Functionality is about average.",
    "Used the {item} for {timeframe} now. {detail}. It is holding up okay.",
    "Nothing wrong with the {item} per se, but {detail}. Just underwhelming.",
    "The {item} is passable. {detail}. Not something I would gift someone.",
    "For casual use the {item} is fine. {detail}. Power users may want more.",
    "The {item} has its pros and cons. {detail}. Ends up being about average.",
    "Somewhat satisfied with the {item}. {detail}. Room for improvement.",
    "The {item} works but feels a bit {adj_neu}. {detail}.",
    "Neutral opinion on this {item}. {detail}. Not excited but not upset either.",
]

# Detail phrases by sentiment
POSITIVE_DETAILS = [
    "Build quality is phenomenal",
    "Performance is lightning fast",
    "The design is sleek and elegant",
    "Setup was incredibly easy",
    "Battery life is outstanding",
    "Delivery was faster than expected",
    "Packaging was superb",
    "Customer support was extremely responsive",
    "Works seamlessly right out of the box",
    "Exceeded the listed specifications",
    "Every feature works as advertised",
    "The material feels premium and durable",
    "Colors are vibrant and accurate",
    "Sound quality is crystal clear",
    "Performance improvement was immediately noticeable",
    "Fits perfectly and feels comfortable",
    "The interface is intuitive and smooth",
    "Highly polished experience from start to finish",
    "No issues whatsoever after extended use",
    "Absolutely worth the premium price tag",
]

NEGATIVE_DETAILS = [
    "Build quality is shockingly poor",
    "Performance is unbearably slow",
    "Design looks nothing like the photos",
    "Setup took hours and still did not work",
    "Battery drains completely in under an hour",
    "Delivery was two weeks late",
    "Arrived in damaged packaging",
    "Customer support was completely useless",
    "Required five reboots just to get started",
    "Specs listed are completely fabricated",
    "Half the advertised features do not function",
    "Materials feel like cheap plastic",
    "Colors are dull and inaccurate",
    "Sound is distorted and full of static",
    "Performance degraded rapidly within days",
    "Fits terribly and is extremely uncomfortable",
    "Interface crashes constantly",
    "Riddled with bugs and glitches throughout",
    "Failed completely after less than a week",
    "Absolutely not worth any amount of money",
]

NEUTRAL_DETAILS = [
    "Build quality is about average",
    "Performance is acceptable for the price point",
    "Design is generic but inoffensive",
    "Setup took a fair amount of time",
    "Battery life is neither great nor terrible",
    "Delivery arrived within the estimated window",
    "Packaging was standard",
    "Customer support responded after a few days",
    "Took some time to get working properly",
    "Meets the stated specifications, nothing more",
    "Features work but feel unpolished",
    "Materials feel standard for the price",
    "Colors are acceptable but a bit washed out",
    "Sound is okay for background listening",
    "Performance is consistent but not impressive",
    "Sizing runs slightly small but wearable",
    "Interface is functional but dated",
    "Experience is fine for basic use cases",
    "Performance has been consistent so far",
    "Represents fair value but not great value",
]

POSITIVE_ADJ = [
    "excellent", "fantastic", "premium", "outstanding", "superb",
    "remarkable", "impressive", "exceptional", "brilliant", "flawless"
]
POSITIVE_ADJ2 = [
    "reliable", "durable", "well-crafted", "intuitive", "responsive",
    "efficient", "polished", "powerful", "versatile", "sleek"
]
NEGATIVE_ADJ = [
    "terrible", "awful", "dreadful", "appalling", "abysmal",
    "pathetic", "shoddy", "defective", "worthless", "broken"
]
NEGATIVE_ADJ2 = [
    "unreliable", "flimsy", "poorly-made", "confusing", "unresponsive",
    "inefficient", "unfinished", "weak", "useless", "overpriced"
]
NEUTRAL_ADJ = [
    "average", "ordinary", "standard", "basic", "acceptable",
    "mediocre", "middling", "so-so", "unremarkable", "passable"
]
TIMEFRAMES = [
    "one day", "two days", "a week", "three days", "four days",
    "a few uses", "the first use", "one month", "two weeks"
]


def generate_review(template: str, sentiment: str) -> str:
    item = random.choice(ALL_ITEMS)
    if sentiment == 'positive':
        detail  = random.choice(POSITIVE_DETAILS)
        adj_pos = random.choice(POSITIVE_ADJ)
        adj_pos2 = random.choice(POSITIVE_ADJ2)
        filled = template.format(
            item=item, detail=detail, adj_pos=adj_pos,
            adj_pos2=adj_pos2, timeframe=random.choice(TIMEFRAMES)
        )
    elif sentiment == 'negative':
        detail   = random.choice(NEGATIVE_DETAILS)
        adj_neg  = random.choice(NEGATIVE_ADJ)
        adj_neg2 = random.choice(NEGATIVE_ADJ2)
        filled = template.format(
            item=item, detail=detail, adj_neg=adj_neg,
            adj_neg2=adj_neg2, timeframe=random.choice(TIMEFRAMES)
        )
    else:
        detail  = random.choice(NEUTRAL_DETAILS)
        adj_neu = random.choice(NEUTRAL_ADJ)
        filled = template.format(
            item=item, detail=detail, adj_neu=adj_neu,
            timeframe=random.choice(TIMEFRAMES)
        )
    return filled


def generate_dataset(n_per_class: int = 200, output_path: str = "data/reviews.csv"):
    """
    Generate n_per_class samples for each of: positive, negative, neutral.
    Uses cycling + randomization to maximize lexical diversity.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []

    for sentiment, templates in [
        ('positive', POSITIVE_TEMPLATES),
        ('negative', NEGATIVE_TEMPLATES),
        ('neutral',  NEUTRAL_TEMPLATES),
    ]:
        generated = set()
        attempts  = 0
        while len(generated) < n_per_class and attempts < n_per_class * 10:
            tmpl   = random.choice(templates)
            review = generate_review(tmpl, sentiment)
            if review not in generated:
                generated.add(review)
                rows.append({'review': review, 'label': sentiment})
            attempts += 1

    random.shuffle(rows)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['review', 'label'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Generated {len(rows)} reviews → {output_path}")
    counts = {}
    for r in rows:
        counts[r['label']] = counts.get(r['label'], 0) + 1
    print(f"   Distribution: {counts}")


if __name__ == "__main__":
    generate_dataset(n_per_class=200, output_path="data/reviews.csv")
