import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pandas.plotting import parallel_coordinates
import ast

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.topic_manager import topics

# Argumente prüfen
if len(sys.argv) > 1 and sys.argv[1] == "labels_only":
    run_labels_only = True
else:
    run_labels_only = False

# Globale Einstellungen für einheitliches Design
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16
})

# Einheitliche Farben für Eskalationslevel
escalation_colors = ["#4daf4a", "#ffb300", "#d62728"]  # Grün, Orange, Rot
escalation_labels = ["None", "Light", "Strong"]

# Connect to database
db_path = os.path.join(
    os.path.dirname(__file__), "../database/chatbot_conversations.db"
)
conn = sqlite3.connect(db_path)

# Create directory for figures
figures_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(figures_dir, exist_ok=True)

# Load data
df = pd.read_sql_query("SELECT * FROM conversations", conn)
conn.close()

# Preprocess
df['category'] = df['category'].str.strip().str.title()
topics_title = [t.title() for t in topics]
df_ai = df[df["api"] == "openai"]
df_ai = df_ai[df_ai["category"].isin(topics_title)]

# Filtere Pro-Con-Konversationen heraus (nur normale Konversationen)
df_ai = df_ai[~df_ai["host_mode"].str.lower().str.startswith("pro-con")]

# Filtere Host-Nachrichten heraus
if "speaker" in df_ai.columns:
    df_ai = df_ai[df_ai["speaker"].str.lower() != "host"]
elif "bot" in df_ai.columns:
    df_ai = df_ai[df_ai["bot"].str.lower() != "host"]

# Filtere nur relevante Nachrichten (z. B. von Bot 1 und Bot 2)
if "speaker" in df_ai.columns:
    df_ai = df_ai[df_ai["speaker"].isin(["Bot 1", "Bot 2"])]
elif "bot" in df_ai.columns:
    df_ai = df_ai[df_ai["bot"].isin(["Bot 1", "Bot 2"])]
    
# --- Normalized Escalations per 1000 Messages by Topic (OpenAI only) ---

# Number of messages per topic
n_messages = df_ai.groupby("category").size().reindex(topics_title, fill_value=0)

# Count each escalation type per topic
escalation_counts = (
    df_ai.groupby(["category", "escalation"])
         .size()
         .unstack(fill_value=0)
         .reindex(topics_title, fill_value=0)
)

# Normalize per 1000 messages
escalation_per_1000 = escalation_counts.div(n_messages, axis=0).fillna(0) * 1000

# Plot stacked bar chart (normalized)
plt.figure(figsize=(12, 7))
bar_width = 0.7
bottom = np.zeros(len(topics_title))

for idx, escalation in enumerate(escalation_labels):
    plt.bar(
        topics_title,
        escalation_per_1000[escalation.lower()],
        bottom=bottom,
        color=escalation_colors[idx],
        width=bar_width,
        label=escalation
    )
    bottom += escalation_per_1000[escalation.lower()].values

plt.title("Normalized Escalations per 1000 Messages by Topic", pad=18, weight="bold")
plt.xlabel("Topic")
plt.ylabel("Escalations per 1000 Messages")
plt.ylim(0, escalation_per_1000.sum(axis=1).max() * 1.15)
plt.xticks(rotation=30, ha="right")
plt.legend(title="Escalation Level", loc="upper right", bbox_to_anchor=(1.1, 1))
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "escalations_per_topic.pdf"))
plt.savefig(os.path.join(figures_dir, "escalations_per_topic.png"))
plt.show()

# --- Escalation distribution by host_mode (Neutral, Provocative, Extreme) ---

# 1. Filter for OpenAI and relevant host_modes
modes = ["neutral", "provocative", "extreme"]
df_modes = df_ai[df_ai["host_mode"].str.lower().isin(modes)]

# 2. Count messages per mode
n_messages_mode = df_modes.groupby("host_mode").size().reindex(modes, fill_value=0)

# 3. Count escalations per mode
escalation_counts_mode = (
    df_modes.groupby(["host_mode", "escalation"])
            .size()
            .unstack(fill_value=0)
            .reindex(modes, fill_value=0)
)

# 4. Normalize per 1000 messages
escalation_per_1000_mode = escalation_counts_mode.div(n_messages_mode, axis=0).fillna(0) * 1000

# 5. Visualization (Stacked Bar Chart)
plt.figure(figsize=(10, 7))
bar_width = 0.7
bottom = np.zeros(len(modes))

for idx, escalation in enumerate(escalation_labels):
    plt.bar(
        [m.title() for m in modes],
        escalation_per_1000_mode[escalation.lower()],
        bottom=bottom,
        color=escalation_colors[idx],
        width=bar_width,
        label=escalation
    )
    bottom += escalation_per_1000_mode[escalation.lower()].values

plt.title("Normalized Escalations per 1000 Messages by Mode", pad=18, weight="bold")
plt.xlabel("Mode")
plt.ylabel("Escalations per 1000 Messages")
plt.ylim(0, escalation_per_1000_mode.sum(axis=1).max() * 1.15)
plt.xticks(rotation=0, ha="center")
plt.legend(title="Escalation Level", loc="upper right", bbox_to_anchor=(1.1, 1))
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "escalations_per_mode.pdf"))
plt.savefig(os.path.join(figures_dir, "escalations_per_mode.png"))
plt.show()

# --- Escalations per topic for each mode (Neutral, Provocative, Extreme) ---

for mode in ["neutral", "provocative", "extreme"]:
    df_mode = df_ai[df_ai["host_mode"].str.lower() == mode]
    n_messages_topic = df_mode.groupby("category").size().reindex(topics_title, fill_value=0)
    escalation_counts_topic = (
        df_mode.groupby(["category", "escalation"])
               .size()
               .unstack(fill_value=0)
               .reindex(topics_title, fill_value=0)
    )
    escalation_per_1000_topic = escalation_counts_topic.div(n_messages_topic, axis=0).fillna(0) * 1000

    plt.figure(figsize=(12, 7))
    bar_width = 0.7
    bottom = np.zeros(len(topics_title))

    for idx, escalation in enumerate(escalation_labels):
        plt.bar(
            topics_title,
            escalation_per_1000_topic[escalation.lower()],
            bottom=bottom,
            color=escalation_colors[idx],
            width=bar_width,
            label=escalation
        )
        bottom += escalation_per_1000_topic[escalation.lower()].values

    plt.title(f"Escalations per 1000 Messages by Topic ({mode.title()})", pad=18, weight="bold")
    plt.xlabel("Topic")
    plt.ylabel("Escalations per 1000 Messages")
    plt.ylim(0, escalation_per_1000_topic.sum(axis=1).max() * 1.15)
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Escalation Level")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"escalations_per_topic_{mode}.pdf"))
    plt.savefig(os.path.join(figures_dir, f"escalations_per_topic_{mode}.png"))
    plt.show()

# --- Heatmap of strong escalation frequency by topic & mode ---

# Only relevant modes and topics
modes = ["neutral", "provocative", "extreme"]
df_heat = df_ai[df_ai["host_mode"].str.lower().isin(modes)]

# Escalation frequency (e.g., share of "strong" per 1000 messages) per topic & mode
heatmap_data = (
    df_heat[df_heat["escalation"] == "strong"]
    .groupby(["category", "host_mode"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=topics_title, columns=modes, fill_value=0)
)

# Normalize per 1000 messages per topic & mode
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
plt.title("Heatmap: Strong Escalations per 1000 Messages by Topic & Mode", pad=18, weight="bold")
plt.xlabel("Host Mode")
plt.ylabel("Topic")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "heatmap_strong_escalations.pdf"))
plt.savefig(os.path.join(figures_dir, "heatmap_strong_escalations.png"))
plt.show()

# --- Heatmap: Combined Escalation Levels by Topic & Mode ---

# Prepare data for heatmap
heatmap_data_combined = (
    df_ai.groupby(["category", "host_mode", "escalation"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["none", "light", "strong"], fill_value=0)
    .groupby(["category", "host_mode"])
    .sum()
    .unstack(fill_value=0)
)

# Normalize per 1000 messages
n_messages_combined = (
    df_ai.groupby(["category", "host_mode"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=topics_title, columns=modes, fill_value=0)
)
heatmap_combined_per_1000 = heatmap_data_combined.div(n_messages_combined).fillna(0) * 1000

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    heatmap_combined_per_1000,
    annot=True,
    fmt=".1f",
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={"label": "Escalations per 1000 Messages"}
)
plt.title("Heatmap: Combined Escalation Levels by Topic & Mode", pad=18, weight="bold")
plt.xlabel("Host Mode")
plt.ylabel("Topic")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "heatmap_combined_escalations.pdf"))
plt.savefig(os.path.join(figures_dir, "heatmap_combined_escalations.png"))
plt.show()

# --- Tabular Overview: Escalation Levels by Topic & Mode ---

# Prepare data for table
table_data = (
    df_ai.groupby(["category", "host_mode", "escalation"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["none", "light", "strong"], fill_value=0)
    .groupby(["category", "host_mode"])
    .sum()
    .unstack(fill_value=0)
)

# Normalize per 1000 messages
table_combined_per_1000 = table_data.div(n_messages_combined).fillna(0) * 1000

# Save table as CSV for supplementary material
table_combined_per_1000.to_csv(os.path.join(figures_dir, "escalation_table.csv"))

# Optional: Print table in console
print("Tabular Overview: Escalation Levels per 1000 Messages")
print(table_combined_per_1000)

# --- Separate Tables for Neutral, Provocative, and Extreme Modes ---

# Modes to process
modes = ["neutral", "provocative", "extreme"]

# Loop through each mode and create a table
for mode in modes:
    # Filter data for the current mode
    df_mode = df_ai[df_ai["host_mode"].str.lower() == mode]
    
    # Count escalation levels per topic
    table_data_mode = (
        df_mode.groupby(["category", "escalation"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["none", "light", "strong"], fill_value=0)
        .reindex(index=topics_title, fill_value=0)
    )
    
    # Normalize per 1000 messages
    n_messages_mode = df_mode.groupby("category").size().reindex(topics_title, fill_value=0)
    table_data_mode_per_1000 = table_data_mode.div(n_messages_mode, axis=0).fillna(0) * 1000
    
    # Save table as CSV
    table_filename = os.path.join(figures_dir, f"escalation_table_{mode}.csv")
    table_data_mode_per_1000.to_csv(table_filename)
    
    # Print table in console (optional)
    print(f"Tabular Overview for Mode: {mode.title()}")
    print(table_data_mode_per_1000)

# --- Sentiment Score Transformation (directional/continuous) ---
def transform_score(row, label_col, score_col):
    label = str(row[label_col]).strip().lower()
    score = row[score_col]
    if pd.isna(label) or pd.isna(score):
        return np.nan
    if label == "positive":
        return score
    elif label == "negative":
        return -score
    else:
        return 0  # or np.nan, if neutral or unknown

df_ai["transformer_sentiment"] = df_ai.apply(
    lambda row: transform_score(row, "transformer_label", "transformer_score"), axis=1
)
df_ai["distilbert_sentiment"] = df_ai.apply(
    lambda row: transform_score(row, "distilbert_label", "distilbert_score"), axis=1
)

# --- Distribution of sentiment values (histogram & density plot) ---

sentiment_cols = ["textblob_polarity", "vader_compound", "transformer_sentiment", "distilbert_sentiment"]

plt.figure(figsize=(12, 7))
for col in sentiment_cols:
    sns.kdeplot(df_ai[col].dropna(), label=col.replace('_', ' ').title(), fill=True, alpha=0.3)
plt.title("Distribution of Sentiment Values")
plt.xlabel("Sentiment Value")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "sentiment_distribution_kde.pdf"))
plt.savefig(os.path.join(figures_dir, "sentiment_distribution_kde.png"))
plt.show()

# Optional: Individual histograms
for col in sentiment_cols:
    plt.figure(figsize=(7, 4))
    plt.hist(df_ai[col].dropna(), bins=30, color="#4daf4a", alpha=0.7)
    plt.title(f"Histogram: {col.replace('_', ' ').title()}")
    plt.xlabel("Sentiment Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"histogram_{col}.pdf"))
    plt.savefig(os.path.join(figures_dir, f"histogram_{col}.png"))
    plt.show()

# --- Boxplot: Sentiment by escalation level ---
sentiment_cols = ["textblob_polarity", "vader_compound"]
for col in sentiment_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x=df_ai['escalation'],
        y=df_ai[col],
        order=["none", "light", "strong"],
        palette=["#4daf4a", "#ffb300", "#d62728"]
    )
    plt.title(f"{col.replace('_', ' ').title()}: Distribution by Escalation Level")
    plt.xlabel("Escalation Level")
    plt.ylabel(col.replace('_', ' ').title())
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"boxplot_{col}_by_escalation_filtered.png"))
    plt.show()

# Boxplot: Message length by escalation level
df_ai["msg_length"] = df_ai["message"].str.len()
plt.figure(figsize=(8, 5))
sns.boxplot(
    x=df_ai['escalation'],
    y=df_ai["msg_length"],
    order=["none", "light", "strong"],
    palette=escalation_colors
)
plt.title("Message Length by Escalation Level", pad=18, weight="bold")
plt.xlabel("Escalation Level")
plt.ylabel("Message Length (characters)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "boxplot_msg_length_by_escalation.pdf"))
plt.savefig(os.path.join(figures_dir, "boxplot_msg_length_by_escalation.png"))
plt.show()

# --- Distribution of Transformer labels by escalation level ---

label_counts = (
    df_ai.groupby(["escalation", "transformer_label"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=["none", "light", "strong"], fill_value=0)
)

label_counts.plot(
    kind="bar",
    stacked=True,
    color=["#4daf4a", "#d62728", "#ffb300"],  # green, red, orange (adjustable)
    figsize=(8, 6)
)
plt.title("Distribution of Transformer Labels by Escalation Level")
plt.xlabel("Escalation Level")
plt.ylabel("Count")
plt.legend(title="Transformer Label")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "transformer_labels_by_escalation.pdf"))
plt.savefig(os.path.join(figures_dir, "transformer_labels_by_escalation.png"))
plt.show()

# --- Distribution of DistilBERT labels by escalation level ---

distilbert_label_counts = (
    df_ai.groupby(["escalation", "distilbert_label"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=["none", "light", "strong"], fill_value=0)
)

distilbert_label_counts.plot(
    kind="bar",
    stacked=True,
    color=["#4daf4a", "#d62728", "#ffb300"],  # green, red, orange (adjustable)
    figsize=(8, 6)
)
plt.title("Distribution of DistilBERT Labels by Escalation Level")
plt.xlabel("Escalation Level")
plt.ylabel("Count")
plt.legend(title="DistilBERT Label")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "distilbert_labels_by_escalation.pdf"))
plt.savefig(os.path.join(figures_dir, "distilbert_labels_by_escalation.png"))
plt.show()

# --- Label agreement: Transformer vs. DistilBERT by escalation level ---

# New column: True if both labels are equal (case-insensitive)
df_ai["label_agreement"] = (
    df_ai["transformer_label"].str.lower() == df_ai["distilbert_label"].str.lower()
)

# Group by escalation level and agreement
agreement_counts = (
    df_ai.groupby(["escalation", "label_agreement"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=["none", "light", "strong"], fill_value=0)
)

# Plot
agreement_counts.rename(columns={True: "Equal", False: "Different"}, inplace=True)
agreement_counts.plot(
    kind="bar",
    color=["#4daf4a", "#d62728"],
    figsize=(8, 6)
)
plt.title("Label Agreement: Transformer vs. DistilBERT by Escalation Level")
plt.xlabel("Escalation Level")
plt.ylabel("Count")
plt.legend(title="Label Agreement")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "label_agreement_by_escalation.pdf"))
plt.savefig(os.path.join(figures_dir, "label_agreement_by_escalation.png"))
plt.show()

# --- Mann-Whitney-U-Test: Compare sentiment distributions (none vs. strong) ---

print("Mann-Whitney-U-Test: Compare sentiment distributions (none vs. strong)\n")
sentiment_cols = ["textblob_polarity", "vader_compound", "transformer_sentiment", "distilbert_sentiment"]

for col in sentiment_cols:
    group_none = df_ai[df_ai["escalation"] == "none"][col].dropna()
    group_strong = df_ai[df_ai["escalation"] == "strong"][col].dropna()
    if len(group_none) > 0 and len(group_strong) > 0:
        stat, p = mannwhitneyu(group_none, group_strong, alternative="two-sided")
        print(f"{col}: U={stat:.0f}, p={p:.4f}")
    else:
        print(f"{col}: Not enough data for test.")



# --- Comparison of escalation levels between Bot 1 and Bot 2 ---

# Assume the bot name column is "bot" or "speaker"
bot_col = "speaker" if "speaker" in df_ai.columns else "bot"

# Count escalation levels per bot
bot_escalation_counts = (
    df_ai.groupby([bot_col, "escalation"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["none", "light", "strong"], fill_value=0)
)

# Stacked bar chart
bot_escalation_counts.plot(
    kind="bar",
    stacked=True,
    color=["#4daf4a", "#ffb300", "#d62728"],
    figsize=(8, 6)
)
plt.title("Comparison of Escalation Levels between Bot 1 and Bot 2")
plt.xlabel("Bot")
plt.ylabel("Count")
plt.legend(title="Escalation Level")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "escalation_comparison_bots.pdf"))
plt.savefig(os.path.join(figures_dir, "escalation_comparison_bots.png"))
plt.show()

# --- K-Means clustering of sentiment values ---

# Only sentiment values
sentiment_cols = ["textblob_polarity", "vader_compound", "transformer_sentiment", "distilbert_sentiment"]
X = df_ai[sentiment_cols].dropna()

# Clustering (e.g., 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# 2D reduction for plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="Set1", alpha=0.6)
plt.title("K-Means Clustering of Sentiment Values (PCA 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "kmeans_clustering_sentiment.pdf"))
plt.savefig(os.path.join(figures_dir, "kmeans_clustering_sentiment.png"))
plt.show()

# --- t-SNE clustering of sentiment values ---

# Only sentiment values
sentiment_cols = ["textblob_polarity", "vader_compound", "transformer_sentiment", "distilbert_sentiment"]
X = df_ai[sentiment_cols].dropna()

# Sample to a maximum of 1000 rows for t-SNE
if len(X) > 1000:
    X_sample = X.sample(1000, random_state=42)
else:
    X_sample = X

# t-SNE reduction to 2D
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_sample)

clusters_sample = kmeans.predict(X_sample)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters_sample, cmap="Set1", alpha=0.6)
plt.title("t-SNE Clustering of Sentiment Values (Sampled)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "tsne_clustering_sentiment_colored.pdf"))
plt.savefig(os.path.join(figures_dir, "tsne_clustering_sentiment_colored.png"))
plt.show()

# --- Frequency of Escalation Reasons (OpenAI only) ---

# Flatten all escalation reasons into a single list
all_reasons = []
for reasons in df_ai["escalation_reason"].dropna():
    # If stored as string representation of list, convert to list
    if isinstance(reasons, str):
        try:
            reasons_list = ast.literal_eval(reasons)
        except Exception:
            reasons_list = [reasons]
    else:
        reasons_list = reasons
    # Skip if reasons_list is None
    if reasons_list is None:
        continue
    # If it's a single string, wrap in list
    if isinstance(reasons_list, str):
        reasons_list = [reasons_list]
    all_reasons.extend(reasons_list)

# Count occurrences
from collections import Counter
reason_counts = Counter(all_reasons)

# Bar plot
plt.figure(figsize=(10, 6))
sns.barplot(
    x=list(reason_counts.keys()),
    y=list(reason_counts.values()),
    palette="viridis"
)
plt.title("Frequency of Escalation Reasons")
plt.xlabel("Escalation Reason")
plt.ylabel("Count")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "escalation_reason_frequency.png"))
plt.show()

# Gruppiere die Gründe nach Modellfamilie
model_groups = {
    "VADER": lambda r: "vader" in r.lower(),
    "DistilBERT": lambda r: "distilbert" in r.lower(),
    "Transformer": lambda r: "transformer" in r.lower(),
    "TextBlob": lambda r: "textblob" in r.lower(),
    "Other": lambda r: True  # fallback
}

group_counts = {"VADER": 0, "DistilBERT": 0, "Transformer": 0, "TextBlob": 0, "Other": 0}
for reason in all_reasons:
    for group, check in model_groups.items():
        if check(reason):
            group_counts[group] += 1
            break

# Bar plot für Modellgruppen
plt.figure(figsize=(7, 5))
sns.barplot(
    x=list(group_counts.keys()),
    y=list(group_counts.values()),
    hue=list(group_counts.keys()),  # Setze `hue` auf die gleichen Werte wie `x`
    dodge=False,  # Deaktiviere das Gruppieren
    palette="viridis"
)
plt.title("Frequency of Escalation Reasons by Model Group")
plt.xlabel("Model Group")
plt.ylabel("Count")
plt.legend(title="Model Group")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "escalation_reason_modelgroup_frequency.png"))
plt.show()

# --- Visualizations for Escalation Reasons ---

# Flatten all escalation reasons into a single list
all_reasons = []
for reasons in df_ai["escalation_reason"].dropna():
    # If stored as string representation of list, convert to list
    if isinstance(reasons, str):
        try:
            reasons_list = ast.literal_eval(reasons)
        except Exception:
            reasons_list = [reasons]
    else:
        reasons_list = reasons
    # Skip if reasons_list is None
    if reasons_list is None:
        continue
    # If it's a single string, wrap in list
    if isinstance(reasons_list, str):
        reasons_list = [reasons_list]
    all_reasons.extend(reasons_list)

# Create a DataFrame for escalation reasons
df_reasons = pd.DataFrame({"escalation_reason": all_reasons})

# Wiederhole die Eskalationsstufen nur für die Länge von `all_reasons`
df_reasons["escalation"] = (
    df_ai.loc[df_ai["escalation_reason"].dropna().index.repeat(
        df_ai["escalation_reason"].dropna().apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 1)
    ), "escalation"].values[:len(all_reasons)]
)

# 1. Barplot: Frequency of Escalation Reasons
plt.figure(figsize=(12, 6))
sns.countplot(
    data=df_reasons,
    x="escalation_reason",
    hue="escalation",
    order=df_reasons["escalation_reason"].value_counts().index,  # Sort by frequency
    palette={"none": "#4daf4a", "light": "#ffb300", "strong": "#d62728"}  # Ergänze "none"
)
plt.title("Frequency of Escalation Reasons by Escalation Level")
plt.xlabel("Escalation Reason")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Escalation Level")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "escalation_reason_frequency.png"))
plt.show()

# 2. Stacked Barplot: Proportion of Models Contributing to Escalation Levels
# Group by escalation level and reason
reason_counts = df_reasons.groupby(["escalation", "escalation_reason"]).size().unstack(fill_value=0)

# Normalize to calculate proportions
reason_percentages = reason_counts.div(reason_counts.sum(axis=1), axis=0)

# Reorder the escalation levels to "none", "light", "strong"
reason_percentages = reason_percentages.reindex(["none", "light", "strong"])

# Define custom colors for reasons (no red tones)
custom_colors = {
    "vader_negative": "#a6cee3",  # Helleres Blau für "negative"
    "vader_strong_negative": "#1f78b4",  # Dunkleres Blau für "strong_negative"
    "distilbert_negative": "#b2df8a",  # Helleres Grün für "negative"
    "distilbert_strong_negative": "#33a02c",  # Dunkleres Grün für "strong_negative"
    "roberta_negative": "#fdbf6f",  # Helleres Gelb-Orange für "negative"
    "roberta_strong_negative": "#ff7f00",  # Dunkleres Gelb-Orange für "strong_negative"
    # Füge weitere Farben für andere Gründe hinzu, falls nötig
}

# Plot stacked bar chart
plt.figure(figsize=(10, 6))
reason_percentages.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    color=[custom_colors.get(reason, "#cccccc") for reason in reason_percentages.columns],  # Verwende die benutzerdefinierten Farben
)
plt.title("Proportion of Models Contributing to Escalation Levels")
plt.xlabel("Escalation Level")
plt.ylabel("Proportion")
plt.xticks(rotation=0, ha="center")  # Set labels horizontal
plt.legend(title="Escalation Reason", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "escalation_reason_stacked_barplot_roberta.png"))
plt.show()

# --- Plot escalation table for each mode ---

# Modes to process
modes = ["neutral", "provocative", "extreme"]

for mode in modes:
    # Load the CSV file for the current mode
    file_path = f"/Users/isabellamoura/Library/Mobile Documents/com~apple~CloudDocs/7. Semester/BA/ChatbotPatterns1/analysis/figures/escalation_table_{mode}.csv"
    df = pd.read_csv(file_path)

    # Round values to integers (no decimal places)
    df.iloc[:, 1:] = df.iloc[:, 1:].round(0).astype(int)

    # Print the table in the terminal
    print(f"\nTabular Overview for Mode: {mode.title()}")
    print(df.to_string(index=False))

    # Plot the table as an image
    plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
    plt.axis('off')  # Turn off the axes
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=["#f2f2f2"] * len(df.columns),  # Light gray for column headers
        cellColours=[["#ffffff"] * len(df.columns) for _ in range(len(df))]  # White for cells
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Save the table as an image
    output_path = f"/Users/isabellamoura/Library/Mobile Documents/com~apple~CloudDocs/7. Semester/BA/ChatbotPatterns1/analysis/figures/escalation_table_{mode}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

# --- Sentiment Trends over Turns (Mean per Turn, TextBlob, VADER, Transformer, DistilBERT) ---

# Filter only relevant speakers (e.g., Bot 1 and Bot 2)
df_ai_turns = df_ai[df_ai["speaker"].isin(["Bot 1", "Bot 2"])]

# Create the "turn" column
df_ai_turns = df_ai_turns.sort_values(["conversation_id", "timestamp"])
df_ai_turns["turn"] = (
    df_ai_turns.groupby("conversation_id").cumcount()
)

# Visualization for each sentiment score
sentiment_scores = {
    "TextBlob Polarity": "textblob_polarity",
    "VADER Compound": "vader_compound",
    "Transformer Score": "transformer_score",
    "DistilBERT Score": "distilbert_score"
}

for title, column in sentiment_scores.items():
    if column in df_ai_turns.columns:
        turn_sentiment = (
            df_ai_turns.groupby(["turn", "speaker"])[column]
            .mean()
            .unstack()
        )
        plt.figure(figsize=(10, 6))
        for speaker in turn_sentiment.columns:
            plt.plot(
                turn_sentiment.index,
                turn_sentiment[speaker],
                marker="o",
                linestyle="-",
                label=speaker
            )
        plt.title(f"Sentiment Trend over Turns ({title})")
        plt.xlabel("Turn")
        plt.ylabel(f"Average {title}")
        plt.legend(title="Bot")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"sentiment_trend_per_turn_{column}.png"))
        plt.show()
    else:
        print(f"Column '{column}' not found – Skipping visualization for {title}.")

# --- Transformer Label Distribution over Turns ---
if run_labels_only and "transformer_label" in df_ai_turns.columns:
    transformer_label_counts = (
        df_ai_turns.groupby(["turn", "speaker", "transformer_label"])
        .size()
        .unstack(fill_value=0)
    )
    transformer_label_counts = transformer_label_counts.div(
        transformer_label_counts.sum(axis=1), axis=0
    ).fillna(0)  # Normalize to relative frequencies

    plt.figure(figsize=(12, 7))
    transformer_label_counts.plot(
        kind="bar",
        stacked=True,
        ax=plt.gca(),
        colormap="viridis"  # Einheitliche Farbpalette
    )
    plt.title("Transformer Label Distribution over Turns")
    plt.xlabel("Turn")
    plt.ylabel("Relative Frequency")
    plt.xticks(ticks=range(0, 61, 10), labels=range(0, 61, 10))  # X-Achse auf 0-60 begrenzen
    plt.xlim(-0.5, 60.5)  # Sicherstellen, dass die Achse bis 60 reicht
    plt.legend(title="Transformer Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "transformer_label_distribution_per_turn.png"))
    plt.show()

# --- DistilBERT Label Distribution over Turns ---
if run_labels_only and "distilbert_label" in df_ai_turns.columns:
    distilbert_label_counts = (
        df_ai_turns.groupby(["turn", "speaker", "distilbert_label"])
        .size()
        .unstack(fill_value=0)
    )
    distilbert_label_counts = distilbert_label_counts.div(
        distilbert_label_counts.sum(axis=1), axis=0
    ).fillna(0)  # Normalize to relative frequencies

    plt.figure(figsize=(12, 7))
    distilbert_label_counts.plot(
        kind="bar",
        stacked=True,
        ax=plt.gca(),
        colormap="viridis"  # Einheitliche Farbpalette
    )
    plt.title("DistilBERT Label Distribution over Turns")
    plt.xlabel("Turn")
    plt.ylabel("Relative Frequency")
    plt.xticks(ticks=range(0, 61, 10), labels=range(0, 61, 10))  # X-Achse auf 0-60 begrenzen
    plt.xlim(-0.5, 60.5)  # Sicherstellen, dass die Achse bis 60 reicht
    plt.legend(title="DistilBERT Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "distilbert_label_distribution_per_turn.png"))
    plt.show()

# --- RoBERTa Label Distribution over Turns ---
if run_labels_only and "transformer_label" in df_ai_turns.columns:
    roberta_label_counts = (
        df_ai_turns.groupby(["turn", "speaker", "transformer_label"])
        .size()
        .unstack(fill_value=0)
    )
    roberta_label_counts = roberta_label_counts.div(
        roberta_label_counts.sum(axis=1), axis=0
    ).fillna(0)  # Normalize to relative frequencies

    plt.figure(figsize=(12, 7))
    roberta_label_counts.plot(
        kind="bar",
        stacked=True,
        ax=plt.gca(),
        colormap="viridis"  # Einheitliche Farbpalette
    )
    plt.title("RoBERTa Label Distribution over Turns")
    plt.xlabel("Turn")
    plt.ylabel("Relative Frequency")
    plt.xticks(ticks=range(0, 61, 10), labels=range(0, 61, 10))  # X-Achse auf 0-60 begrenzen
    plt.xlim(-0.5, 60.5)  # Sicherstellen, dass die Achse bis 60 reicht
    plt.legend(title="RoBERTa Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "roberta_label_distribution_per_turn.png"))
    plt.show()
else:
    print("Column 'transformer_label' not found – Skipping RoBERTa Label visualization.")

# --- Escalation Levels over Turns ---

# Filter only relevant speakers (e.g., Bot 1 and Bot 2)
df_ai_turns = df_ai[df_ai["speaker"].isin(["Bot 1", "Bot 2"])]

# Create the "turn" column
df_ai_turns = df_ai_turns.sort_values(["conversation_id", "timestamp"])
df_ai_turns["turn"] = (
    df_ai_turns.groupby("conversation_id").cumcount()
)

# Count escalation levels per turn
escalation_counts = (
    df_ai_turns.groupby(["turn", "escalation"])
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
plt.title("Escalation Levels over Turns", pad=18, weight="bold")
plt.xlabel("Turn")
plt.ylabel("Relative Frequency")
plt.xticks(ticks=range(0, escalation_percentages.index.max() + 1, 5))  # Adjust tick frequency
plt.legend(title="Escalation Level", loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "escalation_levels_over_turns.png"))
plt.show()

# --- Escalation Levels over Turns (Categorical Y-Axis) ---

# Filter only relevant speakers (e.g., Bot 1 and Bot 2)
df_ai_turns = df_ai[df_ai["speaker"].isin(["Bot 1", "Bot 2"])]

# Create the "turn" column
df_ai_turns = df_ai_turns.sort_values(["conversation_id", "timestamp"])
df_ai_turns["turn"] = (
    df_ai_turns.groupby("conversation_id").cumcount()
)

# Map escalation levels to numeric values for plotting
escalation_mapping = {"none": 0, "light": 1, "strong": 2}
df_ai_turns["escalation_numeric"] = df_ai_turns["escalation"].map(escalation_mapping)

# Plot escalation levels over turns
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df_ai_turns,
    x="turn",
    y="escalation_numeric",
    hue="speaker",  # Different lines for each bot
    palette=["#1f77b4", "#ff7f0e"],  # Colors for Bot 1 and Bot 2
    marker="o",
    linewidth=2
)

# Customize y-axis to show categorical labels
plt.yticks(
    ticks=[0, 1, 2],
    labels=["None", "Light", "Strong"]
)
plt.title("Escalation Levels over Turns", pad=18, weight="bold")
plt.xlabel("Turn")
plt.ylabel("Escalation Level")
plt.legend(title="Speaker", loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "escalation_levels_over_turns_categorical.png"))
plt.show()

# --- Escalation Levels over Turns for Pro and Con Bots ---

# Filter only Pro and Con Bots
df_ai_turns_pro_con = df_ai[df_ai["speaker"].str.contains("Pro|Con", case=False, na=False)]

# Create the "turn" column
df_ai_turns_pro_con = df_ai_turns_pro_con.sort_values(["conversation_id", "timestamp"])
df_ai_turns_pro_con["turn"] = (
    df_ai_turns_pro_con.groupby("conversation_id").cumcount()
)

# Map escalation levels to numeric values for plotting
escalation_mapping = {"none": 0, "light": 1, "strong": 2}
df_ai_turns_pro_con["escalation_numeric"] = df_ai_turns_pro_con["escalation"].map(escalation_mapping)

# Plot escalation levels over turns for Pro and Con Bots
plt.figure(figsize=(12, 6))
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
plt.savefig(os.path.join(figures_dir, "escalation_levels_over_turns_pro_con.png"))
plt.show()

