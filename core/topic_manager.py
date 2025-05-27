import random

topics = {
    "Politics": [
        "Should elections be fully digitalized?",
        "Is democracy in danger?",
        "How do fake news influence politics?",
        "Should politicians be held accountable for false promises?",
        "How effective is direct democracy?"
    ],
    "Social Issues": [
        "Are gender quotas necessary?",
        "Should men and women earn the same salary?",
        "Is cancel culture a threat to free speech?",
        "How does social media shape modern society?"
    ],
    "Religion": [
        "Is religion still relevant today?",
        "Should religions have political influence?",
        "Is secularism the best approach for modern societies?",
        "Does the church still have too much power?"
    ],
    "Economy": [
        "Should the rich be taxed more heavily?",
        "Has capitalism failed?",
        "Are cryptocurrencies the future or just a bubble?",
        "Should cash be abolished?"
    ],
    "Technology": [
        "Is artificial intelligence a threat to jobs?",
        "Should artificial intelligence be regulated?",
        "Will machines eventually replace humans completely?",
        "Is social media a danger to democracy?"
    ],
    "Climate Change": [
        "Is climate change primarily caused by human activity?",
        "Should meat consumption be restricted?",
        "Are electric cars truly environmentally friendly?",
        "Is the Fridays for Future movement effective?"
    ],
    "Human Rights": [
        "Is racism still a systemic problem?",
        "Should former colonial powers pay reparations?",
        "Is feminism still necessary today?",
        "Has the LGBTQ+ movement made enough progress?"
    ],
    "Education": [
        "Should education be free for everyone?",
        "Are universities still necessary in the digital age?",
        "Is the school system outdated?",
        "Are grades a fair assessment method?"
    ]
}

default_category = "Politics"

def get_topic(category=None):
    """Wählt ein Thema aus einer bestimmten Kategorie oder zufällig aus allen Themen."""
    if category and category in topics:
        return random.choice(topics[category])
    else:
        all_topics = [topic for sublist in topics.values() for topic in sublist]
        return random.choice(all_topics)