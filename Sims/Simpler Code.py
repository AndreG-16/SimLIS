import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Parameter --------
days = 7
vehicles_per_day = 80                 # Anzahl zu ladender Fahrzeuge/Tag
p_kw = 11.0                           # "stündlicher Stromverbrauch pro Fahrzeug" = Ladeleistung kW
dur_h = 2                             # Ladedauer in vollen Stunden
max_parallel = 9                   # z.B. 10 oder None für unbegrenzt
rng = np.random.default_rng(123)

# Stunden-Verteilung (24 Werte, Summe = 1). Beispiel: mehr abends.
hour_probs = np.array([
    0.01,0.01,0.01,0.01, 0.01,0.02,0.03,0.04,
    0.05,0.06,0.07,0.08, 0.09,0.09,0.09,0.08,
    0.08,0.06,0.05,0.04, 0.03,0.02,0.01,0.01
], dtype=float)
hour_probs = hour_probs / hour_probs.sum()

# -------- Simulation --------
H = 24 * days
load_kw = np.zeros(H, dtype=float)     # Lastgang in kW (Energie/h = kWh)
active_sessions = [[] for _ in range(H)]

idx = 0
for d in range(days):
    # Stunden-Arrival-Liste für den Tag
    hours = rng.choice(24, size=vehicles_per_day, p=hour_probs)
    for h in hours:
        start = d*24 + h
        end = min(start + dur_h, H)
        # optional: Kappung paralleler Ladevorgänge
        if max_parallel is not None:
            # Trage Session nur ein, wenn in allen Stunden Kapazität bleibt
            feasible = True
            for t in range(start, end):
                if len(active_sessions[t]) >= max_parallel:
                    feasible = False
                    break
            if not feasible:
                continue
        # Session eintragen
        for t in range(start, end):
            active_sessions[t].append(1)

# Leistung pro Stunde = (#parallele Sessions) * p_kw
for t in range(H):
    load_kw[t] = len(active_sessions[t]) * p_kw

# -------- Ausgabe --------
ts = pd.date_range("2025-01-01", periods=H, freq="H")
series = pd.Series(load_kw, index=ts, name="Standortlast (kW)")
series.plot(title="Lastgang (kWh pro Stunde) – Woche")
plt.ylabel("kWh pro Stunde (≈ kW)")
plt.xlabel("Zeit")
plt.tight_layout()
plt.show()
