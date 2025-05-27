import sys
import os
import openai  # Importiere das OpenAI-Modul
from dotenv import load_dotenv

# Lade die Umgebungsvariablen aus der .env-Datei
load_dotenv()

# Setze den OpenAI-API-Schlüssel
openai.api_key = os.getenv("OPENAI_API_KEY")

# Add the project path to the Python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sqlite3
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from analysis.sentiment_analysis import analyze_sentiments  # Importiere die Analysefunktion
from core.conversation_manager import generate_multiple_conversations  # Import the function for conversation generation

# Connect to the database
db_path = "/Users/isabellamoura/Library/Mobile Documents/com~apple~CloudDocs/7. Semester/BA/ChatbotPatterns1/database/chatbot_conversations.db"

# Entfernen Sie das Caching oder verwenden Sie st.cache_data
def load_data():
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM conversations"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Konvertiere die 'timestamp'-Spalte mit einem spezifischen Format
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%dT%H:%M:%S", errors='coerce')
    except Exception as e:
        st.error(f"Error parsing timestamps: {e}")
    
    return df

# Dashboard title
st.title("Chatbot Conversations Dashboard")

# Input fields for conversation generation
st.sidebar.header("Generate Conversations")
topic = st.sidebar.selectbox("Select a Topic", ["Technology", "Education", "Climate Change", "Human Rights", "Economy", "Social Issues", "Politics", "Religion"])
mode = st.sidebar.selectbox("Select a Mode", ["neutral", "provocative", "extreme"])
num_conversations = st.sidebar.number_input("Number of Conversations", min_value=1, max_value=50, value=5, step=1)

# Auswahl der API
api_choice = st.sidebar.selectbox("Select API", ["OpenAI", "DeepSeek"], index=0)  # Standard ist OpenAI

# Auswahl des Konversationstyps
conversation_type = st.sidebar.selectbox("Select Conversation Type", ["Normal", "Pro-Con"], index=0)  # Standard ist Normal

# Button to trigger conversation generation
if st.sidebar.button("Generate Conversations"):
    st.sidebar.write("Generating conversations...")
    
    # API-Auswahl
    api = api_choice.lower()  # Konvertiere die Auswahl in Kleinbuchstaben
    
    # Konversationstyp auswählen
    if conversation_type == "Pro-Con":
        from core.chatbot2 import generate_pro_con_conversation
        generate_pro_con_conversation(category=topic, api=api, mode=mode)
    else:
        generate_multiple_conversations(num_conversations=num_conversations, category=topic, host_mode=mode, api=api)
    
    st.sidebar.success(f"{num_conversations} {conversation_type.lower()} conversation(s) successfully generated using {api.capitalize()}!")
    
    # Debug: Überprüfen Sie die neuen Daten
    conn = sqlite3.connect(db_path)
    new_data = pd.read_sql_query("SELECT * FROM conversations ORDER BY timestamp DESC LIMIT 5", conn)
    conn.close()
    st.sidebar.write("Neue Daten:", new_data)
    
    # Set a state to reload the data
    st.session_state["data_updated"] = True

# Load data
if "data_updated" in st.session_state and st.session_state["data_updated"]:
    df = load_data()
    st.session_state["data_updated"] = False  # Reset the state
else:
    df = load_data()

# Escalation level distribution
st.header("Escalation Level Distribution")
escalation_counts = df['escalation'].value_counts()
st.bar_chart(escalation_counts)

# Escalation levels over time
st.header("Escalation Levels Over Time")
escalation_over_time = df.groupby(df['timestamp'].dt.date)['escalation'].value_counts().unstack()
st.line_chart(escalation_over_time)

# Boxplot for sentiment values
st.header("Distribution of Sentiment Values")
fig, ax = plt.subplots()
sns.boxplot(data=df[['textblob_polarity', 'vader_compound', 'transformer_score']], ax=ax)
st.pyplot(fig)

# Display data
st.header("Display Data")
st.dataframe(df)

# **Neue Funktionalität: Nachricht analysieren**
st.header("Analyze a Custom Message")
user_message = st.text_area("Enter a message to analyze:", "")

if st.button("Analyze Message"):
    if user_message.strip():
        # Nachricht analysieren
        analysis_result = analyze_sentiments(user_message)
        
        # Ergebnisse anzeigen
        st.subheader("Analysis Results")
        st.write(f"**Message:** {user_message}")
        st.write(f"**Sentiment VADER Compound:** {analysis_result['sentiment']['overall']}")
        st.write(f"**TextBlob Subjectivity:** {analysis_result['textblob']['subjectivity']}")
        st.write(f"**Transformer Sentiment (RoBERTa):** {analysis_result['transformer']['label']} ({analysis_result['transformer']['score']:.2f})")
        st.write(f"**Transformer Sentiment (DistilBERT):** {analysis_result['distilbert']['label']} ({analysis_result['distilbert']['score']:.2f})")
        st.write(f"**Toxicity Detected:** {analysis_result['toxicity']['is_toxic']}")
        st.write(f"**Toxicity Score:** {analysis_result['toxicity']['score']:.2f}")
        st.write(f"**Escalation Level:** {analysis_result['escalation']}")
        st.write(f"**Escalation Reasons:** {', '.join(analysis_result['escalation_reason'])}")
    else:
        st.warning("Please enter a message to analyze.")

# **Neue Funktionalität: Chatbot-Interaktion**
st.header("Chatbot Interaction")

# Initialisiere den Chatverlauf in der Session
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Eingabefeld für die Benutzernachricht
user_input = st.text_input("Enter your message:", "")

if st.button("Send"):
    if user_input.strip():
        # Nachricht des Benutzers analysieren
        user_analysis = analyze_sentiments(user_input)
        st.session_state["chat_history"].append({
            "speaker": "User",
            "message": user_input,
            "analysis": user_analysis
        })

        # Bot-Antwort generieren
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Wähle das gewünschte Modell
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=150,
                temperature=0.7
            )
            bot_message = response['choices'][0]['message']['content'].strip()

            # Bot-Nachricht analysieren
            bot_analysis = analyze_sentiments(bot_message)
            st.session_state["chat_history"].append({
                "speaker": "Bot",
                "message": bot_message,
                "analysis": bot_analysis
            })
        except Exception as e:
            st.error(f"Error generating bot response: {e}")
    else:
        st.warning("Please enter a message to send.")

# Chatverlauf anzeigen
st.subheader("Chat History")
for chat in st.session_state["chat_history"]:
    st.markdown(f"**{chat['speaker']}:** {chat['message']}")
    st.write(f"- **Sentiment (VADER Compound):** {chat['analysis']['sentiment']['overall']}")
    st.write(f"- **TextBlob Polarity:** {chat['analysis']['textblob']['polarity']}")
    st.write(f"- **TextBlob Subjectivity:** {chat['analysis']['textblob']['subjectivity']}")
    st.write(f"- **Transformer Sentiment:** {chat['analysis']['transformer']['label']} ({chat['analysis']['transformer']['score']:.2f})")
    st.write(f"- **Toxicity Detected:** {chat['analysis']['toxicity']['is_toxic']}")
    st.write(f"- **Toxicity Score:** {chat['analysis']['toxicity']['score']:.2f}")
    st.write(f"- **Escalation Level:** {chat['analysis']['escalation']}")
    st.markdown("---")