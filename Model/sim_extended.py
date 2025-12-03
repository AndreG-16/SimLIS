"""
Anpassung des Lastprofils:
- Verlängerung des Lastprofils auf einen gewünschten Zeithorizont, indem das Basisprofil (7 Tage) periodisch wiederholt wird
"""

from __future__ import annotations

from datetime import datetime, timedelta, date
from typing import List, Tuple, Dict, Iterable, Optional

import math
import numpy as np


def extend_profile(
    timestamps: List[datetime],
    load_kw: np.ndarray,
    target_days: int,
    noise_std: float = 0.0,
    random_seed: Optional[int] = None,
) -> Tuple[List[datetime], np.ndarray]:
    """
    Erweitert ein periodisches Lastprofil auf einen gewünschten Zeithorizont,
    indem das Basisprofil wiederholt wird.

    Parameters
    ----------
    timestamps : list[datetime]
        Zeitstempel des Basisprofils (gleichmäßiger Zeitschritt!).
    load_kw : np.ndarray
        Lastwerte in kW, gleiche Länge wie timestamps.
    target_days : int
        Gewünschte Gesamtdauer des erweiterten Profils in Tagen.
    noise_std : float, optional
        Standardabweichung eines optionalen Normalrauschens, das auf die
        Lastwerte addiert wird (für leichte Variation). Default: 0.0 (kein Rauschen).
    random_seed : int, optional
        Zufalls-Seed für reproduzierbares Rauschen.

    Returns
    -------
    new_timestamps : list[datetime]
        Erweiterter Zeitindex mit target_days Dauer.
    new_load_kw : np.ndarray
        Erweiterter Lastvektor (gleicher Zeitschritt wie im Original).

    Raises
    ------
    ValueError
        Falls weniger als 2 Zeitstempel übergeben werden oder der Zeitschritt
        nicht konstant ist.
    """
    if len(timestamps) < 2:
        raise ValueError("extend_profile benötigt mindestens 2 Zeitstempel.")

    if len(timestamps) != len(load_kw):
        raise ValueError("timestamps und load_kw müssen die gleiche Länge haben.")

    # Zeitschritt bestimmen
    dt = timestamps[1] - timestamps[0]
    dt_minutes = dt.total_seconds() / 60.0
    if dt_minutes <= 0:
        raise ValueError("Zeitschritt muss positiv sein.")

    # Prüfen, ob der Zeitschritt konstant ist (grob)
    for i in range(1, len(timestamps) - 1):
        if timestamps[i + 1] - timestamps[i] != dt:
            raise ValueError("Zeitstempel sind nicht äquidistant. extend_profile setzt konstante Zeitschritte voraus.")

    # Wie viele Schritte pro Tag?
    steps_per_day = int(round(24 * 60 / dt_minutes))
    if steps_per_day <= 0:
        raise ValueError("Berechnung von steps_per_day fehlgeschlagen.")

    # Wie viele Tage deckt das Basisprofil ab?
    n_steps = len(timestamps)
    original_days = n_steps / steps_per_day

    # Wie oft müssen wir wiederholen, um target_days zu erreichen?
    repeats = int(math.ceil(target_days / original_days))

    # Basisprofil wiederholen
    extended_load = np.tile(load_kw, repeats)

    # Optional: Rauschen hinzufügen
    if noise_std > 0.0:
        if random_seed is not None:
            np.random.seed(random_seed)
        noise = np.random.normal(loc=0.0, scale=noise_std, size=extended_load.shape)
        extended_load = extended_load + noise

    # Neuen Zeitindex erzeugen (durchgehende Fortführung ab erstem Timestamp)
    new_timestamps = [timestamps[0] + i * dt for i in range(len(extended_load))]

    # Auf gewünschte Zielanzahl Schritte zuschneiden
    steps_target = int(target_days * steps_per_day)
    new_timestamps = new_timestamps[:steps_target]
    extended_load = extended_load[:steps_target]

    return new_timestamps, extended_load


def power_to_energy_per_step(
    load_kw: np.ndarray,
    dt_minutes: float,
) -> np.ndarray:
    """
    Rechnet eine Leistungszeitreihe [kW] in Schritt-Energien [kWh] um.

    Parameters
    ----------
    load_kw : np.ndarray
        Leistung in kW pro Zeitschritt.
    dt_minutes : float
        Dauer eines Zeitschritts in Minuten.

    Returns
    -------
    np.ndarray
        Energie in kWh pro Zeitschritt.
    """
    dt_hours = dt_minutes / 60.0
    return load_kw * dt_hours


def aggregate_daily_energy_kwh(
    timestamps: List[datetime],
    load_kw: np.ndarray,
) -> Dict[date, float]:
    """
    Aggregiert ein Lastprofil zu Tagesenergien [kWh].

    Parameters
    ----------
    timestamps : list[datetime]
        Zeitstempel (äquidistant).
    load_kw : np.ndarray
        Leistung in kW, gleiche Länge wie timestamps.

    Returns
    -------
    dict[date, float]
        Mapping von Kalendertag -> Gesamtenergie in kWh.
    """
    if len(timestamps) < 2:
        return {}

    if len(timestamps) != len(load_kw):
        raise ValueError("timestamps und load_kw müssen die gleiche Länge haben.")

    dt = timestamps[1] - timestamps[0]
    dt_minutes = dt.total_seconds() / 60.0
    if dt_minutes <= 0:
        raise ValueError("Zeitschritt muss positiv sein.")

    energy_per_step = power_to_energy_per_step(load_kw, dt_minutes)

    daily_energy: Dict[date, float] = {}

    for ts, e in zip(timestamps, energy_per_step):
        d = ts.date()
        daily_energy[d] = daily_energy.get(d, 0.0) + float(e)

    return daily_energy


def aggregate_max_power_per_day(
    timestamps: List[datetime],
    load_kw: np.ndarray,
) -> Dict[date, float]:
    """
    Aggregiert ein Lastprofil zu täglicher maximaler Leistung [kW].

    Parameters
    ----------
    timestamps : list[datetime]
        Zeitstempel (äquidistant).
    load_kw : np.ndarray
        Leistung in kW, gleiche Länge wie timestamps.

    Returns
    -------
    dict[date, float]
        Mapping von Kalendertag -> maximale Leistung in kW.
    """
    if len(timestamps) != len(load_kw):
        raise ValueError("timestamps und load_kw müssen die gleiche Länge haben.")

    daily_max: Dict[date, float] = {}

    for ts, p in zip(timestamps, load_kw):
        d = ts.date()
        if d not in daily_max:
            daily_max[d] = float(p)
        else:
            daily_max[d] = max(daily_max[d], float(p))

    return daily_max
