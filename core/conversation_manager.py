import sys
import os
import uuid  # For generating a unique ID

# Add the project path to the Python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.db import save_message
from core.chatbot2 import chatbot_conversation
import datetime
import json
import glob
from analysis import sentiment_analysis
from analysis import escalation_detector  # Added for advanced analysis


def save_conversation(conversation, category=None, host_mode="neutral", api="openai"):
    # Generate a unique conversation_id
    conversation_id = str(uuid.uuid4())
    print(f"DEBUG: Saving conversation with ID: {conversation_id}, Category: {category}, Mode: {host_mode}, API: {api}")

    for message in conversation:
        # Save host messages, but without analysis values
        if message["speaker"] == "Host":
            print(f"DEBUG: Saving Host message without analysis: {message}")
            save_message(
                conversation_id,
                message["speaker"],
                message["message"],
                message["timestamp"],
                None,  # No sentiment analysis
                None,  # No escalation analysis
                None,  # TextBlob Polarity
                None,  # TextBlob Subjectivity
                None,  # VADER Compound
                None,  # VADER Negative
                None,  # VADER Neutral
                None,  # VADER Positive
                None,  # Transformer Label
                None,  # Transformer Score
                None,  # Toxicity Is Toxic
                None,  # Toxicity Score
                None,  # DistilBERT Label
                None,  # DistilBERT Score
                api,   # Add API
                category,
                host_mode,
                None   # escalation_reason for host messages
            )
        else:
            # Save bot messages with analysis values
            print(f"DEBUG: Saving Bot message: {message}")
            extended = sentiment_analysis.analyze_sentiments(message["message"])
            save_message(
                conversation_id,
                message["speaker"],
                message["message"],
                message["timestamp"],
                extended['sentiment']['overall'],
                extended['escalation'],
                extended['textblob']['polarity'],
                extended['textblob']['subjectivity'],
                extended['vader']['compound'],
                extended['vader']['neg'],
                extended['vader']['neu'],
                extended['vader']['pos'],
                extended['transformer']['label'],
                extended['transformer']['score'],
                extended['toxicity']['is_toxic'],
                extended['toxicity']['score'],
                extended['distilbert']['label'],
                extended['distilbert']['score'],
                api,
                category,
                host_mode,
                str(extended.get('escalation_reason', ''))
            )
    print("💾 Conversation saved to SQLite database")

# Remove time-pressure logic from the conversation
def generate_multiple_conversations(num_conversations=10, category=None, host_mode="neutral", api="openai"):
    """
    Generates multiple conversations and saves them.
    :param num_conversations: Number of conversations to generate
    :param category: Category of the topic
    :param host_mode: Mode of the host ('neutral', 'provocative', 'extreme')
    :param api: The API to use ('openai' or 'deepseek')
    """
    for i in range(num_conversations):
        print(f"🔄 Generating conversation {i + 1}/{num_conversations} in mode '{host_mode}' using API '{api}'...")
        conversation = chatbot_conversation(category, host_mode=host_mode, api=api)
        if not conversation:
            print("ERROR: chatbot_conversation returned None!")
            continue
        save_conversation(conversation, category, host_mode, api=api)  # Pass API
    print(f"✅ Successfully generated {num_conversations} conversations in mode '{host_mode}' using API '{api}'.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and save chatbot conversations.")
    parser.add_argument("--num_conversations", type=int, default=2, help="Number of conversations to generate")
    parser.add_argument("--category", type=str, default="Technology", help="Category of the conversation")
    parser.add_argument("--host_mode", type=str, default="neutral", choices=["neutral", "provocative", "extreme"], help="Host mode")
    parser.add_argument(
        "--api",
        choices=["openai", "deepseek", "ollama"],
        default="openai",
        help="API to use for generating conversations (not needed for visualization)."
    )

    args = parser.parse_args()

    generate_multiple_conversations(
        num_conversations=args.num_conversations,
        category=args.category,
        host_mode=args.host_mode,
        api=args.api  # Pass API selection
    )