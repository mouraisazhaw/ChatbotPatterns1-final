import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3

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

# Lade die Daten (Pfad anpassen, falls nötig)
data_path = os.path.join(
    os.path.dirname(__file__), "../database/chatbot_conversations.db"
)
conn = sqlite3.connect(data_path)
df = pd.read_sql_query("SELECT * FROM conversations", conn)
conn.close()

# Datenvorbereitung
df['host_mode'] = df['host_mode'].str.lower().str.strip()
df['escalation_level'] = df['escalation'].str.lower().str.strip()

# Filtere NaN-Werte bei `host_mode` und `escalation_level`
df_filtered = df.dropna(subset=['host_mode', 'escalation_level'])

# Filtere Host-Nachrichten vollständig heraus
if "speaker" in df_filtered.columns:
    df_filtered = df_filtered[~df_filtered["speaker"].str.lower().str.contains("host")]
elif "bot" in df_filtered.columns:
    df_filtered = df_filtered[~df_filtered["bot"].str.lower().str.contains("host")]

# Filtere nur relevante Nachrichten (z. B. von Bot 1 und Bot 2)
if "speaker" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["speaker"].isin(["Bot 1", "Bot 2"])]
elif "bot" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["bot"].isin(["Bot 1", "Bot 2"])]

# Sicherstellen, dass die Turns als numerische Werte vorliegen
if "turn" in df_filtered.columns:
    df_filtered["turn"] = pd.to_numeric(df_filtered["turn"], errors="coerce")

# Sicherstellen, dass die Spalte 'turn' existiert
if "turn" not in df_filtered.columns:
    if "message_id" in df_filtered.columns:
        # Ableitung von 'turn' aus 'message_id', falls vorhanden
        df_filtered["turn"] = df_filtered["message_id"].rank(method="dense").astype(int)
    else:
        # Initialisierung, falls keine Ableitung möglich ist
        df_filtered["turn"] = 0

# Sortieren der Daten nach Conversation ID und Turn
df_filtered = df_filtered.sort_values(by=["conversation_id", "turn"])

# Gruppieren nach Conversation ID und Eskalationslevel
escalation_by_turn = df_filtered.groupby(["conversation_id", "turn"])["escalation_level"].apply(
    lambda x: x.mode()[0] if not x.empty else None
).reset_index()

# Eskalationslevel in numerische Werte umwandeln (für die Visualisierung)
escalation_mapping = {"none": 0, "light": 1, "strong": 2}
escalation_by_turn["escalation_numeric"] = escalation_by_turn["escalation_level"].map(escalation_mapping)

# Visualisierung: Eskalationslevel über Turns hinweg
plt.figure(figsize=(12, 6))
sns.lineplot(
    x="turn",
    y="escalation_numeric",
    hue="conversation_id",
    data=escalation_by_turn,
    palette="tab10",
    legend=False  # Optional: Legende deaktivieren, wenn viele Gespräche vorhanden sind
)
plt.title("Escalation Levels Over Turns", pad=18, weight="bold")
plt.xlabel("Turn")
plt.ylabel("Escalation Level (0 = None, 1 = Light, 2 = Strong)")
plt.yticks([0, 1, 2], ["None", "Light", "Strong"])  # Beschriftung der Y-Achse
plt.tight_layout()

# Speicherort für die Grafik
figures_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(figures_dir, exist_ok=True)
output_path_turns = os.path.join(figures_dir, "escalation_levels_over_turns.png")
plt.savefig(output_path_turns, dpi=300)
plt.show()

# Gruppieren nach Eskalationsstufe und Host-Modus
count_by_mode = df_filtered.groupby(['host_mode', 'escalation_level']).size().reset_index(name='count')

# Farbpalette für Eskalationsstufen definieren
escalation_palette = {
    "none": ["#a1d99b", "#31a354", "#006d2c"],  # Grün-Töne für neutral, provocative, extreme
    "light": ["#fed976", "#feb24c", "#fd8d3c"],  # Gelb-Töne für neutral, provocative, extreme
    "strong": ["#fc9272", "#fb6a4a", "#de2d26"]  # Rot-Töne für neutral, provocative, extreme
}

# Funktion, um die Farben basierend auf Eskalationsstufe und Host-Modus zuzuweisen
def get_bar_colors(data, escalation_palette):
    colors = []
    host_mode_order = ["neutral", "provocative", "extreme"]  # Reihenfolge der Host-Modi
    for _, row in data.iterrows():
        escalation_level = row["escalation_level"]
        host_mode = row["host_mode"]
        color_index = host_mode_order.index(host_mode)  # Index des Host-Modus
        colors.append(escalation_palette[escalation_level][color_index])
    return colors

# Farben für die Balken berechnen
bar_colors = get_bar_colors(count_by_mode, escalation_palette)

# Visualisierung 1.1: Gruppiertes Balkendiagramm mit abgestuften Farben
plt.figure(figsize=(10, 6))
sns.barplot(
    x='escalation_level',
    y='count',
    hue='host_mode',
    data=count_by_mode,
    order=["none", "light", "strong"],  # Reihenfolge der Eskalationsstufen
    hue_order=["neutral", "provocative", "extreme"],  # Reihenfolge der Host-Modi
    palette=bar_colors  # Farben basierend auf Eskalationsstufe und Host-Modus
)
plt.title('Message Count by Escalation Level and Host Mode', pad=18, weight="bold")
plt.xlabel('Escalation Level')
plt.ylabel('Number of Messages')
plt.legend(title='Host Mode', loc='upper right', bbox_to_anchor=(1.1, 1))
plt.tight_layout()

# Speicherort für die Grafik
output_path = os.path.join(figures_dir, "message_count_by_escalation_and_host_mode.png")
plt.savefig(output_path, dpi=300)
plt.show()

# 1.2 Boxplot: Verteilung der Nachrichtenlängen nach Eskalationsstufe
# Nachrichtenlänge berechnen
df_filtered['message_length'] = df_filtered['message'].str.len()

# Visualisierung: Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='escalation_level',
    y='message_length',
    data=df_filtered,
    order=["none", "light", "strong"],
    palette=escalation_colors
)
plt.title('Message Length by Escalation Level', pad=18, weight="bold")
plt.xlabel('Escalation Level')
plt.ylabel('Message Length (characters)')
plt.tight_layout()

# Speicherort für die Grafik
output_path_boxplot = os.path.join(figures_dir, "message_length_by_escalation_level.png")
plt.savefig(output_path_boxplot, dpi=300)
plt.show()

# 1.3 Balkendiagramm: Durchschnittliche Nachrichtenlänge pro Eskalationsstufe
# Durchschnittliche Nachrichtenlänge berechnen
avg_message_length = df_filtered.groupby('escalation_level')['message_length'].mean().reset_index()

# Visualisierung: Balkendiagramm
plt.figure(figsize=(10, 6))
sns.barplot(
    x='escalation_level',
    y='message_length',
    data=avg_message_length,
    order=["none", "light", "strong"],
    palette=escalation_colors
)
plt.title('Average Message Length by Escalation Level', pad=18, weight="bold")
plt.xlabel('Escalation Level')
plt.ylabel('Average Message Length (characters)')
plt.tight_layout()

# Speicherort für die Grafik
output_path_avg_length = os.path.join(figures_dir, "avg_message_length_by_escalation_level.png")
plt.savefig(output_path_avg_length, dpi=300)
plt.show()

# 1.4 Gestapeltes Balkendiagramm: Prozentuale Eskalationen pro Host-Modus
# Berechnung der prozentualen Verteilung der Eskalationsstufen pro Host-Modus
percentages = (
    count_by_mode
    .pivot(index="host_mode", columns="escalation_level", values="count")
    .fillna(0)  # Fehlende Werte mit 0 auffüllen
)

# Prozentuale Verteilung berechnen
percentages = percentages.div(percentages.sum(axis=1), axis=0) * 100

# DataFrame für Visualisierung vorbereiten
percentages = percentages.reset_index().melt(id_vars="host_mode", var_name="escalation_level", value_name="percentage")

# Visualisierung: Gestapeltes Balkendiagramm
plt.figure(figsize=(10, 6))
sns.barplot(
    x="host_mode",
    y="percentage",
    hue="escalation_level",
    data=percentages,
    hue_order=["none", "light", "strong"],  # Reihenfolge der Eskalationsstufen
    palette=["#4daf4a", "#ffb300", "#d62728"]  # Grün, Gelb, Rot
)
plt.title('Percentage of Escalations by Host Mode', pad=18, weight="bold")
plt.xlabel('Host Mode')
plt.ylabel('Percentage (%)')
plt.legend(title='Escalation Level', loc='upper right', bbox_to_anchor=(1.1, 1))
plt.tight_layout()

# Speicherort für die Grafik
output_path_percentages = os.path.join(figures_dir, "percentage_escalations_by_host_mode.png")
plt.savefig(output_path_percentages, dpi=300)
plt.show()

# 1.4 Tabellarische Darstellung: Prozentuale Eskalationen pro Host-Modus
# Berechnung der prozentualen Verteilung der Eskalationsstufen pro Host-Modus
percentages_table = (
    count_by_mode
    .pivot(index="host_mode", columns="escalation_level", values="count")
    .fillna(0)  # Fehlende Werte mit 0 auffüllen
)

# Prozentuale Verteilung berechnen
percentages_table = percentages_table.div(percentages_table.sum(axis=1), axis=0) * 100

# Reihenfolge der Eskalationsstufen und Host-Modi festlegen
percentages_table = percentages_table[["none", "light", "strong"]]  # Eskalationsstufen in der gewünschten Reihenfolge
percentages_table = percentages_table.reindex(["neutral", "provocative", "extreme"])  # Host-Modi in der gewünschten Reihenfolge

# Tabellarische Darstellung ausgeben
print("\nPercentage of Escalations by Host Mode (Tabular Format):")
print(percentages_table.round(2))  # Werte auf 2 Dezimalstellen runden

# Optional: Tabelle als CSV speichern
output_path_table = os.path.join(figures_dir, "percentage_escalations_by_host_mode.csv")
percentages_table.to_csv(output_path_table)
