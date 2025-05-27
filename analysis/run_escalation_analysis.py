import sqlite3
from escalation_detector import analyze_sentiments

# Connection to the database
conn = sqlite3.connect("/Users/isabellamoura/Library/Mobile Documents/com~apple~CloudDocs/7. Semester/BA/ChatbotPatterns1/database/chatbot_conversations.db")
cursor = conn.cursor()

# Add missing columns
columns_to_add = [
    ("textblob_polarity", "REAL"),
    ("textblob_subjectivity", "REAL"),
    ("vader_compound", "REAL"),
    ("vader_neg", "REAL"),
    ("vader_neu", "REAL"),
    ("vader_pos", "REAL"),
    ("transformer_label", "TEXT"),
    ("transformer_score", "REAL")
]

for column, col_type in columns_to_add:
    try:
        cursor.execute(f"ALTER TABLE conversations ADD COLUMN {column} {col_type}")
    except sqlite3.OperationalError:
        # Column already exists
        pass

# Retrieve messages that have not yet been analyzed
cursor.execute("""
SELECT id, message FROM conversations
WHERE textblob_polarity IS NULL OR vader_compound IS NULL OR transformer_label IS NULL
""")
rows = cursor.fetchall()

print("Analyzing and saving messages:\n")

for row in rows:
    msg_id, message = row
    result = analyze_sentiments(message)

    # Save results in the `conversations` table
    cursor.execute("""
    UPDATE conversations
    SET textblob_polarity = ?, textblob_subjectivity = ?,
        vader_compound = ?, vader_neg = ?, vader_neu = ?, vader_pos = ?,
        transformer_label = ?, transformer_score = ?, escalation = ?
    WHERE id = ?
    """, (
        result['textblob']['polarity'], result['textblob']['subjectivity'],
        result['vader']['compound'], result['vader']['neg'], result['vader']['neu'], result['vader']['pos'],
        result['transformer']['label'], result['transformer']['score'],
        result['escalation'], msg_id
    ))

    # Output in the terminal
    print(f"ID {msg_id}: {message[:60]}...")
    print(f"→ Escalation: {result['escalation']}")
    print(f"→ TextBlob Polarity: {result['textblob']['polarity']}, VADER Compound: {result['vader']['compound']}")
    print("-" * 60)

# Save changes and close the connection
conn.commit()
conn.close()

print("Analysis completed and results saved.")
