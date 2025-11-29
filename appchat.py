import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import jarque_bera
from flask import Flask, render_template, request
import os

# === LISTA TITOLI E NOMI ===
TITOLI_FTSEMIB = [
    "UCG.MI", "ISP.MI", "ENEL.MI", "RACE.MI", "G.MI",
    "ENI.MI", "PRY.MI", "PST.MI", "LDO.MI", "BMPS.MI"
]

NOMI_COMPLETI = {
    "UCG.MI": "UniCredit",
    "ISP.MI": "Intesa Sanpaolo",
    "ENEL.MI": "Enel",
    "RACE.MI": "Ferrari",
    "G.MI": "Generali",
    "ENI.MI": "Eni",
    "PRY.MI": "Prysmian",
    "PST.MI": "Poste Italiane",
    "LDO.MI": "Leonardo",
    "BMPS.MI": "Banca MPS"
}

# === FUNZIONI ANALISI ===
def scarica_dati_ftse(titoli, data_inizio="2019-01-01"):
    prezzi = pd.DataFrame()
    for ticker in titoli:
        df = yf.Ticker(ticker).history(start=data_inizio)
        if not df.empty:
            prezzi[ticker] = df["Close"]
    return prezzi

def calcola_rendimenti(prezzi):
    rendimenti = prezzi.pct_change().dropna()
    rendimenti_log = np.log(prezzi / prezzi.shift(1)).dropna()
    return rendimenti, rendimenti_log

def statistiche_rendimenti(rendimenti):
    stats = pd.DataFrame()
    stats["Min"] = rendimenti.min()
    stats["Max"] = rendimenti.max()
    stats["Media"] = rendimenti.mean()
    stats["StdDev"] = rendimenti.std()

    moda_vals = []
    for col in rendimenti.columns:
        m = rendimenti[col].mode()
        moda_vals.append(m.iloc[0] if len(m) else np.nan)
    stats["Moda"] = moda_vals

    jb = {}
    for col in rendimenti.columns:
        _, p = jarque_bera(rendimenti[col])
        jb[col] = p
    stats["Jarque-Bera p-value"] = pd.Series(jb)

    return stats

# === INIZIO APP ===
app = Flask(__name__)

prezzi = scarica_dati_ftse(TITOLI_FTSEMIB)
rendimenti, rendimenti_log = calcola_rendimenti(prezzi)
stats = statistiche_rendimenti(rendimenti)
correlazioni = rendimenti.corr()

grafici_dir = "static/grafici"
os.makedirs(grafici_dir, exist_ok=True)

print("Generazione grafici...")

# GENERA GRAFICI UNA VOLTA
for ticker in TITOLI_FTSEMIB:
    safe = ticker.replace(".", "_")
    nome = NOMI_COMPLETI[ticker]

    data = rendimenti[ticker].dropna()
    mu, sigma = data.mean(), data.std()

    # Prezzi
    plt.figure(figsize=(10,4))
    plt.plot(prezzi.index, prezzi[ticker])
    plt.title(f"Andamento prezzi {nome}")
    plt.tight_layout()
    plt.savefig(f"{grafici_dir}/{safe}_prezzi.png")
    plt.close()

    # Istogramma
    plt.figure(figsize=(6,4))
    plt.hist(data, bins=50, density=True, alpha=0.6)
    x_vals = np.linspace(data.min(), data.max(), 300)
    normal_curve = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_vals - mu)/sigma)**2)
    plt.plot(x_vals, normal_curve, linewidth=2)
    plt.title(f"Istogramma {nome}")
    plt.tight_layout()
    plt.savefig(f"{grafici_dir}/{safe}_istogramma.png")
    plt.close()

    # Boxplot
    plt.figure(figsize=(4,6))
    sns.boxplot(y=data)
    plt.title(f"Boxplot {nome}")
    plt.tight_layout()
    plt.savefig(f"{grafici_dir}/{safe}_boxplot.png")
    plt.close()

# KDE
plt.figure(figsize=(14,6))
for t in rendimenti_log.columns:
    sns.kdeplot(rendimenti_log[t].dropna(), label=NOMI_COMPLETI[t])
plt.legend()
plt.tight_layout()
plt.savefig(f"{grafici_dir}/kde_rendimenti.png")
plt.close()

# Rendimenti cumulati
plt.figure(figsize=(12,6))
rend_cum = (1 + rendimenti).cumprod()
for t in rend_cum.columns:
    plt.plot(rend_cum.index, rend_cum[t], label=NOMI_COMPLETI[t])
plt.legend()
plt.tight_layout()
plt.savefig(f"{grafici_dir}/rendimenti_cumulati.png")
plt.close()

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(correlazioni, annot=True, cmap="coolwarm", fmt=".2f")
plt.tight_layout()
plt.savefig(f"{grafici_dir}/heatmap_correlazioni.png")
plt.close()

# === ROUTE ===
@app.route("/")
def index():
    selected = request.args.get("titolo", TITOLI_FTSEMIB[0])
    safe = selected.replace(".", "_")

    return render_template(
        "index.html",
        titoli=[(t, NOMI_COMPLETI[t]) for t in TITOLI_FTSEMIB],
        NOMI_COMPLETI=NOMI_COMPLETI,
        selected=selected,
        prezzo=f"/static/grafici/{safe}_prezzi.png",
        istogramma=f"/static/grafici/{safe}_istogramma.png",
        boxplot=f"/static/grafici/{safe}_boxplot.png",
        stats=stats.round(4).to_html(classes="table table-striped", border=0)
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
