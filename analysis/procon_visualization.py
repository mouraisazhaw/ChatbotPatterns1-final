import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.topic_manager import topics

# Connect to database
db_path = os.path.join(
    os.path.dirname(__file__), "../database/chatbot_conversations.db"
)
conn = sqlite3.connect(db_path)

# Create directory for figures
figures_dir = os.path.join(os.path.dirname(__file__), "figures", "procon")
os.makedirs(figures_dir, exist_ok=True)

# Load data
df = pd.read_sql_query("SELECT * FROM conversations", conn)
conn.close()

# Preprocess
df['category'] = df['category'].str.strip().str.title()
topics_title = [t.title() for t in topics]

# Filter only Pro-Con conversations
df_procon = df[df["host_mode"].str.lower().str.startswith("pro-con")]

# Erzeuge die "turn"-Spalte: Jede Pro-Bot/Con-Bot-Paarung bekommt eine Turn-Nummer innerhalb einer Konversation
df_procon = df_procon.sort_values(["conversation_id", "timestamp"])
df_procon["turn"] = (
    df_procon[df_procon["speaker"].isin(["Pro-Bot", "Con-Bot"])]
    .groupby("conversation_id")
    .cumcount()
)

df_procon = df_procon[df_procon["speaker"].isin(["Pro-Bot", "Con-Bot"])]

# 1. Escalation-Level Vergleich Pro-Bot vs. Con-Bot
bot_escalation_counts = (
    df_procon.groupby(["speaker", "escalation"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["none", "light", "strong"], fill_value=0)
)
bot_escalation_counts.plot(
    kind="bar",
    stacked=True,
    color=["#4daf4a", "#ffb300", "#d62728"],
    figsize=(8, 6)
)
plt.title("Pro-Con: Escalation Levels – Pro-Bot vs. Con-Bot")
plt.xlabel("Bot")
plt.ylabel("Count")
plt.legend(title="Escalation Level")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "escalation_comparison_procon_bots.png"))
plt.show()

# 2. Sentiment-Vergleich Pro-Bot vs. Con-Bot (VADER)
plt.figure(figsize=(8, 5))
sns.boxplot(
    x=df_procon['speaker'],
    y=df_procon['vader_compound'],
    palette=["#4daf4a", "#d62728"]
)
plt.title("Pro-Con: Sentiment (VADER) – Pro-Bot vs. Con-Bot")
plt.xlabel("Bot")
plt.ylabel("VADER Compound Sentiment")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "sentiment_comparison_procon_bots.png"))
plt.show()

# 2b. Sentiment-Vergleich Pro-Bot vs. Con-Bot (TextBlob)
if "textblob_polarity" in df_procon.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x=df_procon["speaker"],
        y=df_procon["textblob_polarity"],
        palette=["#4daf4a", "#d62728"]
    )
    plt.title("Pro-Con: Sentiment (TextBlob) – Pro-Bot vs. Con-Bot")
    plt.xlabel("Bot")
    plt.ylabel("TextBlob Polarity")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sentiment_comparison_textblob_procon_bots.png"))
    plt.show()
else:
    print("Spalte 'textblob_polarity' nicht gefunden – TextBlob-Boxplot wird übersprungen.")

# 3. Escalations nach Topic & Mode (gestapelte Balken)
for mode in ["pro-con-neutral", "pro-con-provocative", "pro-con-extreme"]:
    df_mode = df_procon[df_procon["host_mode"] == mode]
    n_messages_topic = df_mode.groupby("category").size().reindex(topics_title, fill_value=0)
    escalation_counts_topic = (
        df_mode.groupby(["category", "escalation"])
               .size()
               .unstack(fill_value=0)
               .reindex(topics_title, fill_value=0)
    )
    escalation_per_1000_topic = escalation_counts_topic.div(n_messages_topic, axis=0).fillna(0) * 1000

    plt.figure(figsize=(12, 7))
    bottom = np.zeros(len(topics_title))
    colors = ["#4daf4a", "#ffb300", "#d62728"]
    labels = ["none", "light", "strong"]

    for idx, escalation in enumerate(labels):
        plt.bar(
            topics_title,
            escalation_per_1000_topic[escalation],
            bottom=bottom,
            color=colors[idx],
            edgecolor="black",
            width=0.7,
            label=escalation.capitalize()
        )
        bottom += escalation_per_1000_topic[escalation].values

    plt.title(f"Pro-Con: Escalations per 1000 Messages by Topic ({mode.replace('pro-con-', '').title()})")
    plt.xlabel("Topic")
    plt.ylabel("Escalations per 1000 Messages")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Escalation Level")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"escalations_per_1000_{mode}.png"))
    plt.show()

# 4. Heatmap: Strong Escalations nach Topic & Mode
modes = ["pro-con-neutral", "pro-con-provocative", "pro-con-extreme"]
df_heat = df_procon[df_procon["host_mode"].isin(modes)]
heatmap_data = (
    df_heat[df_heat["escalation"] == "strong"]
    .groupby(["category", "host_mode"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=topics_title, columns=modes, fill_value=0)
)
n_messages = (
    df_heat.groupby(["category", "host_mode"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=topics_title, columns=modes, fill_value=0)
)
heatmap_per_1000 = heatmap_data.div(n_messages).fillna(0) * 1000

plt.figure(figsize=(11, 7))
sns.heatmap(
    heatmap_per_1000,
    annot=True,
    fmt=".1f",
    cmap="Reds",
    linewidths=0.5,
    cbar_kws={"label": "Strong Escalations per 1000 Messages"}
)
plt.title("Pro-Con: Strong Escalations per 1000 Messages by Topic & Mode")
plt.xlabel("Host Mode")
plt.ylabel("Topic")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "heatmap_strong_escalations_procon.png"))
plt.show()

# 5. Sentiment-Verlauf über Turns (Mittelwert pro Turn, beide Bots)
df_procon_turns = df_procon[df_procon["speaker"].isin(["Pro-Bot", "Con-Bot"])]
turn_sentiment = (
    df_procon_turns.groupby(["turn", "speaker"])["vader_compound"]
    .mean()
    .unstack()
)
plt.figure(figsize=(10, 6))
# Verbinde die Punkte mit Linien für beide Bots
plt.plot(
    turn_sentiment.index,
    turn_sentiment["Pro-Bot"],
    marker="o",
    linestyle="-",
    color="orange",
    label="Pro-Bot"
)
plt.plot(
    turn_sentiment.index,
    turn_sentiment["Con-Bot"],
    marker="o",
    linestyle="-",
    color="blue",
    label="Con-Bot"
)
plt.title("Pro-Con: Sentiment-Verlauf über Turns (VADER)")
plt.xlabel("Turn")
plt.ylabel("Durchschnittlicher VADER Sentiment")
plt.legend(title="Bot")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "sentiment_trend_per_turn_procon.png"))
plt.show()

# 5b. Sentiment-Verlauf über Turns (Mittelwert pro Turn, TextBlob)
if "textblob_polarity" in df_procon.columns:
    turn_sentiment_textblob = (
        df_procon_turns.groupby(["turn", "speaker"])["textblob_polarity"]
        .mean()
        .unstack()
    )
    plt.figure(figsize=(10, 6))
    for speaker in turn_sentiment_textblob.columns:
        plt.plot(
            turn_sentiment_textblob.index,
            turn_sentiment_textblob[speaker],
            marker="o",
            linestyle="-",
            label=speaker
        )
    plt.title("Pro-Con: Sentiment-Verlauf über Turns (TextBlob)")
    plt.xlabel("Turn")
    plt.ylabel("Durchschnittlicher TextBlob Polarity")
    plt.legend(title="Bot")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sentiment_trend_per_turn_textblob.png"))
    plt.show()
else:
    print("Spalte 'textblob_polarity' nicht gefunden – TextBlob-Turn-Darstellung wird übersprungen.")

# 5c. Sentiment-Verlauf über Turns (Mittelwert pro Turn, Transformer Score)
if "transformer_score" in df_procon.columns:
    turn_sentiment_transformer = (
        df_procon_turns.groupby(["turn", "speaker"])["transformer_score"]
        .mean()
        .unstack()
    )
    plt.figure(figsize=(10, 6))
    for speaker in turn_sentiment_transformer.columns:
        plt.plot(
            turn_sentiment_transformer.index,
            turn_sentiment_transformer[speaker],
            marker="o",
            linestyle="-",
            label=speaker
        )
    plt.title("Pro-Con: Sentiment-Verlauf über Turns (Transformer Score)")
    plt.xlabel("Turn")
    plt.ylabel("Durchschnittlicher Transformer Score")
    plt.legend(title="Bot")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sentiment_trend_per_turn_transformer.png"))
    plt.show()
else:
    print("Spalte 'transformer_score' nicht gefunden – Transformer-Turn-Darstellung wird übersprungen.")

# 5d. Sentiment-Verlauf über Turns (Häufigkeit der Transformer Labels)
if "transformer_label" in df_procon.columns:
    transformer_label_counts = (
        df_procon_turns.groupby(["turn", "speaker", "transformer_label"])
        .size()
        .unstack(fill_value=0)
    )
    transformer_label_counts = transformer_label_counts.div(
        transformer_label_counts.sum(axis=1), axis=0
    ).fillna(0)  # Normalisiere auf relative Häufigkeiten

    plt.figure(figsize=(12, 7))
    transformer_label_counts.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="viridis")
    plt.title("Pro-Con: Transformer Label-Verteilung über Turns")
    plt.xlabel("Turn")
    plt.ylabel("Relative Häufigkeit")
    plt.legend(title="Transformer Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "transformer_label_distribution_per_turn.png"))
    plt.show()
else:
    print("Spalte 'transformer_label' nicht gefunden – Transformer-Label-Darstellung wird übersprungen.")

# 5e. Sentiment-Verlauf über Turns (Häufigkeit der DistilBERT Labels)
if "distilbert_label" in df_procon.columns:
    distilbert_label_counts = (
        df_procon_turns.groupby(["turn", "speaker", "distilbert_label"])
        .size()
        .unstack(fill_value=0)
    )
    distilbert_label_counts = distilbert_label_counts.div(
        distilbert_label_counts.sum(axis=1), axis=0
    ).fillna(0)  # Normalisiere auf relative Häufigkeiten

    plt.figure(figsize=(12, 7))
    distilbert_label_counts.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="coolwarm")
    plt.title("Pro-Con: DistilBERT Label-Verteilung über Turns")
    plt.xlabel("Turn")
    plt.ylabel("Relative Häufigkeit")
    plt.legend(title="DistilBERT Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "distilbert_label_distribution_per_turn.png"))
    plt.show()
else:
    print("Spalte 'distilbert_label' nicht gefunden – DistilBERT-Label-Darstellung wird übersprungen.")

# 6. Wortwolken für Pro-Bot und Con-Bot
for bot in ["Pro-Bot", "Con-Bot"]:
    text = " ".join(df_procon[df_procon["speaker"] == bot]["message"].dropna().astype(str))
    wordcloud = WordCloud(
        width=800, height=400, background_color="white",
        stopwords=STOPWORDS, colormap="viridis"
    ).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Wordcloud: {bot}")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"wordcloud_{bot.lower().replace('-', '_')}.png"))
    plt.show()



# --- Escalation Levels over Turns (Pro-Bot und Con-Bot) ---

# Filter only relevant speakers (Pro-Bot and Con-Bot)
df_procon_turns = df_procon[df_procon["speaker"].isin(["Pro-Bot", "Con-Bot"])]

# Count escalation levels per turn
escalation_counts = (
    df_procon_turns.groupby(["turn", "escalation"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["none", "light", "strong"], fill_value=0)
)

# Normalize to calculate relative frequencies
escalation_percentages = escalation_counts.div(escalation_counts.sum(axis=1), axis=0).fillna(0)

# Plot escalation levels over turns
plt.figure(figsize=(10, 6))
escalation_percentages.plot(
    kind="line",
    marker="o",
    ax=plt.gca(),
    color=["#4daf4a", "#ffb300", "#d62728"],  # Colors for none, light, strong
    linewidth=2
)
plt.title("Escalation Levels over Turns (Pro-Bot und Con-Bot)", pad=18, weight="bold")
plt.xlabel("Turn")
plt.ylabel("Relative Frequency")
# Konvertiere den Maximalwert des Index in einen Ganzzahltyp
plt.xticks(ticks=range(0, int(escalation_percentages.index.max()) + 1, 5))  # Adjust tick frequency
plt.legend(title="Escalation Level", loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "escalation_levels_over_turns_procon.png"))
plt.show()

# --- Escalation Levels over Turns (Categorical Y-Axis) ---

# Map escalation levels to numeric values for plotting
escalation_mapping = {"none": 0, "light": 1, "strong": 2}
df_procon_turns["escalation_numeric"] = df_procon_turns["escalation"].map(escalation_mapping)

# Plot escalation levels over turns (Categorical Y-Axis)
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df_procon_turns,
    x="turn",
    y="escalation_numeric",
    hue="speaker",  # Different lines for each bot
    palette=["#1f77b4", "#ff7f0e"],  # Colors for Pro-Bot and Con-Bot
    marker="o",
    linewidth=2,
    alpha=0.8  # Reduce transparency for better visibility
)

# Customize y-axis to show categorical labels
plt.yticks(
    ticks=[0, 1, 2],
    labels=["None", "Light", "Strong"],
    fontsize=12
)
plt.xticks(fontsize=12)
plt.title("Escalation Levels over Turns (Pro-Bot und Con-Bot)", pad=18, weight="bold", fontsize=14)
plt.xlabel("Turn", fontsize=12)
plt.ylabel("Escalation Level", fontsize=12)
plt.legend(title="Speaker", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "escalation_levels_over_turns_categorical_procon_adjusted.png"))
plt.show()

# --- Escalation Levels over Turns (Pro und Con Bots, mit Schattierungen) ---

# Filter only relevant speakers (Pro-Bot and Con-Bot)
df_ai_turns_pro_con = df_procon[df_procon["speaker"].isin(["Pro-Bot", "Con-Bot"])]

# Map escalation levels to numeric values for plotting
escalation_mapping = {"none": 0, "light": 1, "strong": 2}
df_ai_turns_pro_con["escalation_numeric"] = df_ai_turns_pro_con["escalation"].map(escalation_mapping)

# Calculate mean and standard deviation for shaded areas
mean_escalation = df_ai_turns_pro_con.groupby(["turn", "speaker"])["escalation_numeric"].mean().unstack()
std_escalation = df_ai_turns_pro_con.groupby(["turn", "speaker"])["escalation_numeric"].std().unstack()

# Plot escalation levels over turns with shaded areas
plt.figure(figsize=(12, 6))

# Plot shaded areas (standard deviation)
for speaker, color in zip(mean_escalation.columns, ["#1f77b4", "#ff7f0e"]):  # Colors for Pro and Con Bots
    plt.fill_between(
        mean_escalation.index,
        mean_escalation[speaker] - std_escalation[speaker],
        mean_escalation[speaker] + std_escalation[speaker],
        color=color,
        alpha=0.2,  # Transparency for shaded areas
        label=f"{speaker} (±Std)"
    )

# Plot lines
sns.lineplot(
    data=df_ai_turns_pro_con,
    x="turn",
    y="escalation_numeric",
    hue="speaker",  # Different lines for Pro and Con Bots
    palette=["#1f77b4", "#ff7f0e"],  # Colors for Pro and Con Bots
    marker="o",
    linewidth=2
)

# Customize y-axis to show categorical labels
plt.yticks(
    ticks=[0, 1, 2],
    labels=["None", "Light", "Strong"]
)
plt.title("Escalation Levels over Turns (Pro and Con Bots)", pad=18, weight="bold")
plt.xlabel("Turn")
plt.ylabel("Escalation Level")
plt.legend(title="Speaker", loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "escalation_levels_over_turns_pro_con_shaded.png"))
plt.show()

# --- Combined Escalation Levels for Bot 1, Bot 2, Pro-Bot, and Con-Bot ---

# Filter OpenAI data
df_ai = df[df["api"] == "openai"]
df_ai = df_ai[df_ai["category"].isin(topics_title)]

# Optional: Debugging-Ausgabe
print("df_ai shape:", df_ai.shape)


# Combine data for all bots
df_combined = pd.concat([
    df_ai[["speaker", "escalation"]],  # Assuming df_ai contains Bot 1 and Bot 2 data
    df_procon[["speaker", "escalation"]]  # Pro-Bot and Con-Bot data
])

# Count escalation levels per bot
combined_escalation_counts = (
    df_combined.groupby(["speaker", "escalation"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["none", "light", "strong"], fill_value=0)
)

# Normalize counts to percentages
combined_escalation_percentages = combined_escalation_counts.div(combined_escalation_counts.sum(axis=1), axis=0) * 100

# Stacked bar chart for all speaker roles (percentage)
combined_escalation_percentages.plot(
    kind="bar",
    stacked=True,
    color=["#4daf4a", "#ffb300", "#d62728"],  # Colors for none, light, strong
    figsize=(10, 6)
)
plt.title("Comparison of Escalation Levels by Speaker Role", pad=18, weight="bold")
plt.xlabel("Speaker Role", fontsize=12)
plt.ylabel("Percentage (%)", fontsize=12)
plt.legend(title="Escalation Level", fontsize=10)
plt.xticks(rotation=30, ha="right", fontsize=10)
plt.tight_layout()

# Save the combined visualization
plt.savefig(os.path.join(figures_dir, "escalation_comparison_by_speaker_role_percentage.png"))
plt.savefig(os.path.join(figures_dir, "escalation_comparison_by_speaker_role_percentage.pdf"))
plt.show()

