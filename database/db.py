import os
import sqlite3

def get_db_connection():
    # Get the absolute path to the database file
    db_path = os.path.join(
        os.path.dirname(__file__), "chatbot_conversations.db"
    )
    print(f"DEBUG: Using database at {db_path}")  # Debug output for database path
    conn = sqlite3.connect(db_path)

    # Create the table if it doesn't exist or update the schema
    conn.execute('''CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    speaker TEXT,
                    message TEXT,
                    timestamp TEXT,
                    sentiment_score REAL,
                    escalation TEXT
                )''')

    # Ensure the column 'conversation_id' exists
    try:
        conn.execute("ALTER TABLE conversations ADD COLUMN conversation_id TEXT")
    except sqlite3.OperationalError:
        # Column already exists
        pass

    # Add additional columns if they are not already present
    extra_columns = [
        ("textblob_polarity", "REAL"),
        ("textblob_subjectivity", "REAL"),
        ("vader_compound", "REAL"),
        ("vader_neg", "REAL"),
        ("vader_neu", "REAL"),
        ("vader_pos", "REAL"),
        ("transformer_label", "TEXT"),
        ("transformer_score", "REAL"),
        ("toxicity_is_toxic", "TEXT"),
        ("toxicity_score", "REAL"),
        ("category", "TEXT"),
        ("host_mode", "TEXT"),
        ("escalation_reason", "TEXT"),
        ("distilbert_label", "TEXT"),
        ("distilbert_score", "REAL"),
        ("api", "TEXT"),
    ]
    for column, col_type in extra_columns:
        try:
            conn.execute(f"ALTER TABLE conversations ADD COLUMN {column} {col_type}")
        except sqlite3.OperationalError:
            # Column already exists
            pass

    return conn

# Function to save a message with extended fields
def save_message(conversation_id, speaker, message, timestamp, sentiment_score, escalation,
                 textblob_polarity, textblob_subjectivity, vader_compound, vader_neg, vader_neu, vader_pos,
                 transformer_label, transformer_score, toxicity_is_toxic, toxicity_score,
                 distilbert_label, distilbert_score, api, category, host_mode, escalation_reason):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO conversations (
            speaker, message, timestamp, sentiment_score, escalation,
            textblob_polarity, textblob_subjectivity, vader_compound, vader_neg, vader_neu, vader_pos,
            transformer_label, transformer_score, conversation_id, api,
            toxicity_is_toxic, toxicity_score, category, host_mode, escalation_reason,
            distilbert_label, distilbert_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            speaker, message, timestamp, sentiment_score, escalation,
            textblob_polarity, textblob_subjectivity, vader_compound, vader_neg, vader_neu, vader_pos,
            transformer_label, transformer_score, conversation_id, api,
            toxicity_is_toxic, toxicity_score, category, host_mode, str(escalation_reason),
            distilbert_label, distilbert_score
        )
    )
    conn.commit()
    conn.close()

# Function to retrieve all conversations
def get_all_conversations():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM conversations")
    conversations = cursor.fetchall()
    conn.close()
    return conversations

def get_aggregated_conversations():
    """
    Aggregates data from the conversations table, grouped by conversation_id.
    Returns:
      - conversation_id
      - Number of messages
      - Average sentiment score
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
    SELECT 
        conversation_id,
        COUNT(*) AS message_count,
        AVG(sentiment_score) AS avg_sentiment
    FROM conversations
    GROUP BY conversation_id
    """
    cursor.execute(query)
    aggregated_data = cursor.fetchall()
    conn.close()
    return aggregated_data




