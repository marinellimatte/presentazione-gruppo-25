import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import jarque_bera
from flask import Flask, render_template
import os

# === LISTA TITOLI E NOMI COMPLETI ===
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
        t = yf.Ticker(ticker)
        df = t.history(start=data_inizio)
        if not df.empty:
            prezzi[ticker] = df["Close"]
    return prezzi

def calcola_rendimenti(prezzi):
    rendimenti = prezzi.pct_change().dropna()
    rendimenti_log = np.log(prezzi / prezzi.shift(1)).dropna()
    return rendimenti, rendimenti_log

def statistiche_rendimenti(rendimenti):
    moda_vals = []
    for col in rendimenti.columns:
        m = rendimenti[col].mode()
        moda_vals.append(m.iloc[0] if len(m) > 0 else np.nan)

    stats = pd.DataFrame({
        "Min": rendimenti.min(),
        "Max": rendimenti.max(),
        "Moda": moda_vals,
        "Media": rendimenti.mean(),
        "StdDev": rendimenti.std()
    }, index=rendimenti.columns)

    jb_results = {}
    for col in rendimenti.columns:
        stat, p = jarque_bera(rendimenti[col])
        jb_results[col] = p

    stats["Jarque-Bera p-value"] = pd.Series(jb_results)
    return stats

# === FLASK APP ===
app = Flask(__name__, static_folder="static")

prezzi = scarica_dati_ftse(TITOLI_FTSEMIB)
rendimenti, rendimenti_log = calcola_rendimenti(prezzi)
stats = statistiche_rendimenti(rendimenti)
correlazioni = rendimenti.corr()

grafici_dir = "static/grafici"
os.makedirs(grafici_dir, exist_ok=True)

# Grafici singoli
for ticker in TITOLI_FTSEMIB:
    ticker_safe = ticker.replace(".", "_")
    nome = NOMI_COMPLETI[ticker]
    data = rendimenti[ticker].dropna()
    mu, sigma = data.mean(), data.std()

    # Prezzi
    plt.figure(figsize=(10,4))
    plt.plot(prezzi.index, prezzi[ticker])
    plt.title(f"Andamento prezzi {nome}")
    plt.tight_layout()
    plt.savefig(f"{grafici_dir}/{ticker_safe}_prezzi.png")
    plt.close()

    # Istogramma
    plt.figure(figsize=(6,4))
    plt.hist(data, bins=50, density=True, alpha=0.6)
    x_vals = np.linspace(data.min(), data.max(), 300)
    normal_curve = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mu) / sigma) ** 2)
    plt.plot(x_vals, normal_curve, linewidth=2)
    plt.title(f"Istogramma rendimenti {nome}")
    plt.tight_layout()
    plt.savefig(f"{grafici_dir}/{ticker_safe}_istogramma.png")
    plt.close()

    # Boxplot
    plt.figure(figsize=(4,6))
    sns.boxplot(y=data)
    plt.title(f"Boxplot {nome}")
    plt.tight_layout()
    plt.savefig(f"{grafici_dir}/{ticker_safe}_boxplot.png")
    plt.close()

# Grafico KDE
plt.figure(figsize=(14,6))
for t in rendimenti_log.columns:
    sns.kdeplot(rendimenti_log[t].dropna(), label=NOMI_COMPLETI[t])
plt.title("Curve di Densità (KDE) dei Rendimenti Logaritmici")
plt.xlabel("Rendimento Logaritmico")
plt.ylabel("Densità")
plt.legend()
plt.tight_layout()
plt.savefig(f"{grafici_dir}/kde_rendimenti.png")
plt.close()

# Grafico rendimenti cumulati
plt.figure(figsize=(12,6))
rend_cum = (1 + rendimenti).cumprod()
for t in rend_cum.columns:
    plt.plot(rend_cum.index, rend_cum[t], label=NOMI_COMPLETI[t])
plt.title("Rendimenti cumulati (2019–oggi)")
plt.xlabel("Anno")
plt.ylabel("Fattore di crescita")
plt.legend()
plt.tight_layout()
plt.savefig(f"{grafici_dir}/rendimenti_cumulati.png")
plt.close()

# Heatmap correlazioni
plt.figure(figsize=(8,6))
sns.heatmap(correlazioni, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap correlazioni tra titoli")
plt.tight_layout()
plt.savefig(f"{grafici_dir}/heatmap_correlazioni.png")
plt.close()





@app.route("/")
def index():
    return render_template("index.html",
        titoli=[(t, NOMI_COMPLETI[t]) for t in TITOLI_FTSEMIB],
        stats=stats.round(4).to_html(classes="table table-striped", border=0)
    )

if __name__ == "__main__":
    app.run
