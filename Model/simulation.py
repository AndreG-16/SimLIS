from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any, Callable
from pathlib import Path

# =============================================================================
# Modulüberblick
# =============================================================================
# Dieses Modul simuliert den Lastgang eines Ladepark-Standorts über einen konfigurierten Zeithorizont.
# Es trennt bewusst:
#   (1) Grundmodell / Session-Modellierung (Flotte, Ankünfte, Standzeiten, Energiebedarf),
#   (2) Lademanagement (immediate, market, generation inkl. Fallbacks).
#
# Ziel ist:
#   - Zeitprofil der ev-Ladeleistung (kW)
#   - Session-Details für KPI-Analysen
#   - optional Debug-Zeitreihen für Notebook-Auswertungen


# =============================================================================
# 0) Hilfsfunktionen: Zeit / Einheiten
# =============================================================================
# Konventionen:
# - Intern wird in kWh pro Zeitschritt gerechnet.
# - Umrechnung kW -> kWh/step erfolgt mit step_hours = time_resolution_min / 60
# - CSV-Eingänge können Dezimal-Komma nutzen (z.B. "2,16") und verschiedene Trenner haben.


def read_scenario_from_yaml(scenario_path: str) -> dict:
    """
    Liest eine YAML-Szenario-Datei ein, validiert Pflichtfelder und gibt das Szenario als dict zurück.
    """
    with open(scenario_path, "r", encoding="utf-8") as file_handle:
        scenario = yaml.safe_load(file_handle)

    if not isinstance(scenario, dict):
        raise ValueError("YAML konnte nicht als dict gelesen werden.")

    required_top_level_keys = [
        "time_resolution_min",
        "simulation_horizon_days",
        "start_datetime",
        "site",
        "vehicles",
    ]
    for key in required_top_level_keys:
        if key not in scenario:
            raise ValueError(f"Pflichtfeld fehlt in YAML: '{key}'")

    required_site_keys = [
        "number_chargers",
        "rated_power_kw",
        "grid_limit_p_avb_kw",
    ]
    for key in required_site_keys:
        if key not in scenario["site"]:
            raise ValueError(f"Pflichtfeld fehlt in YAML.site: '{key}'")

    return scenario


def infer_timezone_from_holidays(scenario: dict) -> str | None:
    """
    Leitet eine Zeitzone aus der Angabe `scenario["holidays"]["country"]` ab.

    Hinweis: Das ist nur für Länder mit genau einer relevanten Zeitzone eindeutig.
    Für Deutschland ("DE") wird "Europe/Berlin" zurückgegeben.
    """
    holidays_cfg = scenario.get("holidays") or {}
    country = str(holidays_cfg.get("country", "")).strip().upper()

    # Minimales Mapping (bei Bedarf um weitere Länder erweitern)
    if country == "DE":
        return "Europe/Berlin"

    # Fallback: timezone-naives Verhalten beibehalten
    return None


def _step_hours(time_resolution_min: int) -> float:
    """
    Rechnet eine Zeitauflösung in Minuten in Stunden pro Zeitschritt um.
    """
    return float(time_resolution_min) / 60.0


def _ensure_datetime_index(
    dataframe: pd.DataFrame,
    datetime_column_name: str,
    timezone: Any = None,
) -> pd.DataFrame:
    """
    Erstellt aus einer Zeitstempel-Spalte einen sauberen, sortierten DatetimeIndex.

    Es werden mehrere explizite Datums-/Zeitformate unterstützt (ohne „Raten“), u.a. ISO-Formate
    sowie gängige deutsche Formate. Ungültige Zeitstempel werden verworfen.

    Optional kann eine Zeitzone übergeben werden: Dann werden die Zeitstempel in diese Zeitzone
    lokalisiert und DST-Fälle (Sommer-/Winterzeit) werden robust behandelt:
    - Bei doppelten Zeiten (Winterzeit-Umstellung) wird nach Möglichkeit automatisch aufgelöst.
    - Bei nicht existierenden Zeiten (Sommerzeit-Umstellung) wird nach vorne verschoben.
    Falls das nicht möglich ist, werden problematische Zeitstempel auf NaT gesetzt und entfernt.
    """
    dataframe = dataframe.copy()

    if datetime_column_name not in dataframe.columns:
        raise ValueError(f"Datetime column '{datetime_column_name}' not found in dataframe columns.")

    datetime_text = dataframe[datetime_column_name].astype(str).str.strip()

    parsed = pd.to_datetime(datetime_text, errors="coerce", format="%Y-%m-%d %H:%M:%S")

    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(datetime_text[mask], errors="coerce", format="%Y-%m-%d %H:%M")

    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(datetime_text[mask], errors="coerce", format="%Y-%m-%dT%H:%M:%S")

    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(datetime_text[mask], errors="coerce", format="%d.%m.%Y %H:%M")

    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(datetime_text[mask], errors="coerce", format="%d.%m.%y %H:%M")

    dataframe[datetime_column_name] = parsed
    dataframe = dataframe.dropna(subset=[datetime_column_name])

    if len(dataframe) == 0:
        sample_values = datetime_text.head(5).to_list()
        raise ValueError(
            f"All timestamps could not be parsed from column '{datetime_column_name}'. "
            f"Sample values: {sample_values}"
        )

    dataframe = dataframe.sort_values(datetime_column_name).set_index(datetime_column_name)

    if not dataframe.index.is_unique:
        dataframe = dataframe.groupby(level=0).mean(numeric_only=True)

    # Optional: Zeitzone setzen (DST-/Sommerzeit-Winterzeit-robust)
    if timezone is not None and getattr(dataframe.index, "timezone", None) is None:
        try:
            dataframe.index = dataframe.index.tz_localize(
                timezone,
                ambiguous="infer",            # Winterzeit: doppelte Stunde -> versuchen zu inferieren
                nonexistent="shift_forward",  # Sommerzeit: fehlende Stunde -> nach vorne schieben
            )
        except Exception:
            # Konservativer Fallback: problematische Zeitstempel auf NaT setzen und entfernen
            dataframe.index = dataframe.index.tz_localize(
                timezone,
                ambiguous="NaT",
                nonexistent="NaT",
            )
            dataframe = dataframe[~dataframe.index.isna()]

    return dataframe


def _convert_energy_series_to_kwh_per_step(values: np.ndarray, unit: str, time_resolution_min: int) -> np.ndarray:
    """
    Konvertiert eine Zeitreihe in kWh pro Zeitschritt (kWh/step) anhand der angegebenen Einheit.
    """
    cleaned_unit = (unit or "").strip()
    step_hours = _step_hours(time_resolution_min)

    if cleaned_unit == "kWh":
        return values.astype(float)
    if cleaned_unit == "MWh":
        return values.astype(float) * 1000.0
    if cleaned_unit == "kW":
        return values.astype(float) * step_hours
    if cleaned_unit == "MW":
        return values.astype(float) * 1000.0 * step_hours

    raise ValueError(f"Unbekannte Energieeinheit: '{cleaned_unit}'")


def _convert_price_series_to_eur_per_kwh(values: np.ndarray, unit: str) -> np.ndarray:
    """
    Konvertiert eine Preis-Zeitreihe in €/kWh anhand der angegebenen Einheit.
    """
    cleaned_unit = (unit or "").strip()

    if cleaned_unit == "€/kWh":
        return values.astype(float)
    if cleaned_unit == "€/MWh":
        return values.astype(float) / 1000.0

    raise ValueError(f"Unbekannte Preiseinheit: '{cleaned_unit}'")


def _read_table_flex(csv_path: str, prefer_decimal_comma: bool = True) -> pd.DataFrame:
    """
    Liest CSV-Dateien robust ein (Tab/Semikolon/Komma; optional Dezimalkomma)
    und unterstützt Tausendertrennzeichen (z.B. SMARD: "3.545,50").
    """
    decimal_character = "," if prefer_decimal_comma else "."
    thousands_character = "." if prefer_decimal_comma else ","

    parsing_attempts = [
        dict(sep="\t", decimal=decimal_character, thousands=thousands_character),
        dict(sep=";",  decimal=decimal_character, thousands=thousands_character),
        dict(sep=",",  decimal=decimal_character, thousands=thousands_character),
    ]

    last_exception: Optional[Exception] = None
    for attempt in parsing_attempts:
        try:
            dataframe = pd.read_csv(csv_path, **attempt)
            if dataframe.shape[1] >= 2:
                return dataframe
        except Exception as exc:
            last_exception = exc

    raise ValueError(f"CSV konnte nicht robust gelesen werden: {csv_path} ({last_exception})")


def _reindex_to_simulation_timestamps(
    series: pd.Series,
    timestamps: pd.DatetimeIndex,
    method: str = "nearest",
) -> pd.Series:
    """
    Bringt eine Zeitreihe auf das Zeitraster der Simulation.

    Dabei werden zwei typische Probleme abgefangen:
    1) Unsortierte Zeitstempel → werden zuerst sortiert.
    2) Doppelte Zeitstempel (kommt z.B. bei Exporten/Marktdaten vor) → werden vor dem Reindex
       zusammengeführt, weil `reindex()` sonst fehlschlagen kann.

    Das Ergebnis enthält genau die `timestamps` der Simulation. Fehlende Werte werden gemäß
    `method` (Standard: "nearest") aus den vorhandenen Punkten abgeleitet.
    """
    # Reihenfolge sicherstellen (für groupby/reindex hilfreich und deterministisch)
    series = series.sort_index()

    # Doppelte Zeitstempel auflösen, damit `reindex()` sauber funktioniert.
    # Standard: Mittelwert pro Zeitstempel (passt meist für PV-/Preiszeitreihen, wenn Duplikate identisch/ähnlich sind).
    if not series.index.is_unique:
        series = series.groupby(level=0).mean()

        # Alternative (falls du lieber "letzter gewinnt" willst):
        # series = series[~series.index.duplicated(keep="last")]

    # Auf die Simulationszeitstempel bringen (z.B. 15-min Raster).
    # method="nearest" nimmt den zeitlich nächstgelegenen Messpunkt.
    return series.reindex(timestamps, method=method)


# =============================================================================
# 1) Daten-Reader: Gebäudeprofil / Marktpreise / Ladekurven / PV-Generation
# =============================================================================

@dataclass                          # erzeugt automatisch __init__, __repr__ und __eq__ für eine Klasse, die primär Daten hält.
class VehicleChargingCurve:
    """
    Container für eine fahrzeugspezifische Ladekennlinie.

    Die Kennlinie beschreibt die maximal aufnehmbare Ladeleistung in Abhängigkeit vom
    Ladezustand (SoC). Sie wird z.B. genutzt, um pro Zeitschritt die zulässige
    Ladeleistung zu begrenzen (SoC-abhängiges Fahrzeuglimit).

    - `state_of_charge_fraction` enthält SoC-Werte als Bruchteil von 0.0 bis 1.0
      (typischerweise aufsteigend sortiert).
    - `power_kw` enthält die dazugehörige Ladeleistung in kW (gleiche Länge wie SoC-Array).
    """
    vehicle_name: str
    manufacturer: str
    model: str
    vehicle_class: str
    battery_capacity_kwh: float
    state_of_charge_fraction: np.ndarray
    power_kw: np.ndarray


def read_local_load_profile_from_csv(
    csv_path: str,
    value_column_one_based: int,
    value_unit: str,
    annual_scaling_value: float,
    time_resolution_min: int,
    timestamps: pd.DatetimeIndex,
) -> pd.Series:
    """
    Liest ein Gebäude-/Grundlastprofil aus einer CSV und gibt es als kWh pro Simulationszeitschritt zurück.

    Skalierungslogik (Option A):
    - 'annual_scaling_value' wird als Jahresenergie in kWh/Jahr interpretiert.
    - Das gilt unabhängig davon, ob die CSV-Werte als
        * "kWh" (Energie pro CSV-Zeitschritt) oder
        * "kW"  (mittlere Leistung pro CSV-Zeitschritt)
      vorliegen.

    Das Profil wird so skaliert, dass die daraus resultierende Energie der betrachteten CSV-Zeitspanne
    zur vorgegebenen Jahresenergie passt. Deckt die CSV nicht genau ein Jahr ab, wird anteilig skaliert:
        Zielenergie für CSV-Zeitraum = Jahresenergie * (CSV-Stunden / 8760)

    Am Ende wird das Profil auf kWh pro SIMULATIONS-Zeitschritt umgerechnet und auf `timestamps`
    reindiziert (Zeitpunkte werden auf das Simulationsraster gemappt).
    """
    dataframe = _read_table_flex(csv_path, prefer_decimal_comma=True)

    datetime_column_name = "DateTime" if "DateTime" in dataframe.columns else str(dataframe.columns[0])

    # 1) Wertspalte bestimmen, bevor der Index gesetzt wird (Original-Spalten der CSV)
    original_columns = list(dataframe.columns)
    value_column_zero_based = int(value_column_one_based) - 1
    if value_column_zero_based < 0 or value_column_zero_based >= len(original_columns):
        raise ValueError(
            f"base_load_value_col={value_column_one_based} ist außerhalb der CSV-Spaltenanzahl "
            f"({len(original_columns)})."
        )
    value_column_name = original_columns[value_column_zero_based]

    # 2) DatetimeIndex setzen (Zeitzone von den Simulations-Timestamps übernehmen, falls vorhanden)
    timezone = getattr(timestamps, "timezone", None)
    dataframe = _ensure_datetime_index(dataframe, datetime_column_name, timezone=timezone)

    # 3) Werte lesen (negatives abschneiden)
    raw_values = pd.to_numeric(dataframe[value_column_name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    raw_values = np.maximum(raw_values, 0.0)

    cleaned_unit = (value_unit or "").strip()

    # --- Zeitschrittweite des Profils in Stunden bestimmen (aus den CSV-Timestamps) ---
    sim_step_hours = _step_hours(time_resolution_min)

    profile_step_hours = sim_step_hours  # Fallback: gleiche Auflösung wie Simulation annehmen
    if len(dataframe.index) >= 2:
        diffs = pd.Series(dataframe.index).diff().dropna()
        if len(diffs) > 0:
            median_seconds = float(diffs.dt.total_seconds().median())
            if np.isfinite(median_seconds) and median_seconds > 0:
                profile_step_hours = median_seconds / 3600.0

    # Gesamtstunden, die die CSV ungefähr abdeckt
    csv_hours = float(len(raw_values)) * float(profile_step_hours)
    if csv_hours <= 0:
        raise ValueError("Gebäudeprofil: CSV-Zeitraum ist leer oder Schrittweite ungültig.")

    # Zielenergie für genau den Zeitraum, den die CSV abdeckt (kWh)
    annual_kwh = float(annual_scaling_value)
    if annual_kwh <= 0:
        raise ValueError("annual_scaling_value (kWh/Jahr) muss > 0 sein.")
    target_kwh_for_csv_period = annual_kwh * (csv_hours / 8760.0)

    # --- CSV-Werte zunächst in kWh pro CSV-Zeitschritt umrechnen (noch unskaliert) ---
    if cleaned_unit == "kWh":
        shape_kwh_per_csv_step = raw_values.astype(float)

    elif cleaned_unit == "kW":
        # kW * Stunden_pro_CSV_Schritt -> kWh pro CSV-Schritt
        shape_kwh_per_csv_step = raw_values.astype(float) * float(profile_step_hours)

    else:
        raise ValueError("value_unit muss 'kWh' oder 'kW' sein.")

    shape_sum_kwh = float(np.sum(shape_kwh_per_csv_step))
    if shape_sum_kwh <= 0.0:
        raise ValueError("Gebäudeprofil: Summe der (kWh-)Shape-Werte ist 0.")

    # --- Skaliert so, dass die Energie im CSV-Zeitraum der Zielenergie entspricht ---
    scaling_factor = float(target_kwh_for_csv_period) / float(shape_sum_kwh)
    scaled_kwh_per_csv_step = shape_kwh_per_csv_step * scaling_factor

    # --- Umrechnung auf kWh pro SIMULATIONS-Zeitschritt ---
    # erst in Leistung (kW) zurückrechnen, dann auf Simulationsschritt-Energie (kWh/step)
    scaled_power_kw = scaled_kwh_per_csv_step / max(float(profile_step_hours), 1e-12)
    scaled_kwh_per_sim_step = scaled_power_kw * float(sim_step_hours)

    series = pd.Series(scaled_kwh_per_sim_step, index=dataframe.index, name="base_load_kwh_per_step")
    series = _reindex_to_simulation_timestamps(series, timestamps, method="nearest")
    return series


def read_market_profile_from_csv(
    csv_path: str,
    value_column_one_based: int,
    value_unit: str,
    timestamps: pd.DatetimeIndex,
) -> pd.Series:
    """
    Liest ein Marktpreisprofil aus CSV ein, normalisiert auf €/kWh und reindiziert es auf Simulations-Zeitstempel.
    value_column_one_based bezieht sich auf die Original-CSV-Spalten (inkl. Zeitspalte).
    """
    dataframe = _read_table_flex(csv_path, prefer_decimal_comma=True)

    datetime_column_name = "Datum von" if "Datum von" in dataframe.columns else str(dataframe.columns[0])

    # 1) Spaltenname anhand originaler CSV-Spalten bestimmen 
    original_columns = list(dataframe.columns)
    value_column_zero_based = int(value_column_one_based) - 1
    if value_column_zero_based < 0 or value_column_zero_based >= len(original_columns):
        raise ValueError(
            f"market_strategy_value_col={value_column_one_based} ist außerhalb der CSV-Spaltenanzahl "
            f"({len(original_columns)})."
        )
    value_column_name = original_columns[value_column_zero_based]

    # 2) DatetimeIndex setzen
    timezone = getattr(timestamps, "timezone", None)
    dataframe = _ensure_datetime_index(dataframe, datetime_column_name, timezone=timezone)

    # 3) Werte lesen + Einheit konvertieren
    raw_values = pd.to_numeric(dataframe[value_column_name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    prices_eur_per_kwh = _convert_price_series_to_eur_per_kwh(raw_values, value_unit)

    series = pd.Series(prices_eur_per_kwh, index=dataframe.index, name="market_price_eur_per_kwh")
    series = _reindex_to_simulation_timestamps(series, timestamps, method="ffill")
    return series


def read_vehicle_load_profiles_from_csv(vehicle_curve_csv: str) -> Dict[str, VehicleChargingCurve]:
    """
    Liest Fahrzeug-Ladekurven im Multi-Header-Format ein und baut daraus ein Dict {vehicle_name: VehicleChargingCurve}.
    """
    raw_dataframe = pd.read_csv(vehicle_curve_csv, sep=None, engine="python", header=None, decimal=",")

    if raw_dataframe.shape[0] < 6 or raw_dataframe.shape[1] < 2:
        raise ValueError("Ladekurven-CSV scheint nicht das erwartete Format zu haben (zu wenige Zeilen/Spalten).")

    manufacturer_row = raw_dataframe.iloc[0, :].tolist()
    model_row = raw_dataframe.iloc[1, :].tolist()
    vehicle_class_row = raw_dataframe.iloc[2, :].tolist()
    battery_capacity_row = raw_dataframe.iloc[3, :].tolist()

    state_of_charge_percent = pd.to_numeric(raw_dataframe.iloc[4:, 0], errors="coerce").to_numpy(dtype=float)
    if np.all(np.isnan(state_of_charge_percent)):
        raise ValueError("Ladekurven-CSV: SOC-Spalte konnte nicht als Zahl gelesen werden.")
    state_of_charge_percent = np.nan_to_num(state_of_charge_percent, nan=0.0)
    state_of_charge_fraction = np.clip(state_of_charge_percent / 100.0, 0.0, 1.0)

    curves_by_vehicle_name: Dict[str, VehicleChargingCurve] = {}

    for column_index in range(1, raw_dataframe.shape[1]):
        manufacturer = str(manufacturer_row[column_index]).strip()
        model = str(model_row[column_index]).strip()
        vehicle_class = str(vehicle_class_row[column_index]).strip()

        battery_capacity_value = pd.to_numeric(pd.Series([battery_capacity_row[column_index]]), errors="coerce").iloc[0]
        if pd.isna(battery_capacity_value):
            continue
        battery_capacity_kwh = float(battery_capacity_value)

        power_values_kw = (
            pd.to_numeric(raw_dataframe.iloc[4:, column_index], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        power_values_kw = np.maximum(power_values_kw, 0.0)

        if len(power_values_kw) != len(state_of_charge_fraction):
            raise ValueError("Ladekurven-CSV: SOC und Leistung haben unterschiedliche Länge.")
        if len(state_of_charge_fraction) < 2:
            continue

        vehicle_name = f"{manufacturer} {model}".strip()
        if vehicle_name == "" or vehicle_name.lower() == "nan nan":
            vehicle_name = f"Vehicle_{column_index}"

        curves_by_vehicle_name[vehicle_name] = VehicleChargingCurve(
            vehicle_name=vehicle_name,
            manufacturer=manufacturer,
            model=model,
            vehicle_class=vehicle_class,
            battery_capacity_kwh=battery_capacity_kwh,
            state_of_charge_fraction=state_of_charge_fraction.copy(),
            power_kw=power_values_kw.copy(),
        )

    if len(curves_by_vehicle_name) == 0:
        raise ValueError("Ladekurven-CSV: Es konnten keine Fahrzeugkurven extrahiert werden.")

    return curves_by_vehicle_name


def build_simulation_timestamps(scenario: dict) -> pd.DatetimeIndex:
    """
    Baut die Simulations-Zeitstempel anhand von Startdatum, Horizont (Tage) und Zeitschritt (min).
    Optional werden die Zeitstempel in einer Zeitzone erzeugt (tz-aware), wenn diese im Szenario ableitbar ist.
    """
    time_resolution_min = int(scenario["time_resolution_min"])
    simulation_horizon_days = int(scenario["simulation_horizon_days"])

    timezone = scenario.get("timezone", None)  # falls du sie vorher im Szenario setzt/ableitest

    start = pd.Timestamp(str(scenario["start_datetime"]))

    if timezone is not None:
        if start.tzinfo is None:
            start = start.tz_localize(timezone)
        else:
            start = start.tz_convert(timezone)

    number_steps_total = int(simulation_horizon_days) * int(24 * 60 / time_resolution_min)

    timestamps = pd.date_range(
        start=start,
        periods=number_steps_total,
        freq=f"{time_resolution_min}min",
    )
    return pd.DatetimeIndex(timestamps)


def _infer_datetime_column_name_for_signal(dataframe: pd.DataFrame) -> str:
    """
    Versucht, einen passenden Zeitstempel-Spaltennamen in einer CSV zu erkennen.
    """
    for candidate in ["Datum von", "DateTime", "Datetime", "timestamp", "Timestamp", "time", "Time", "date", "Date"]:
        if candidate in dataframe.columns:
            return candidate
    return str(dataframe.columns[0])


def read_generation_profile_from_csv(
    csv_path: str,
    value_column_one_based: int,
    value_unit: str,
    pv_profile_reference_kwp: float,
    pv_system_size_kwp: float,
    time_resolution_min: int,
    timestamps: pd.DatetimeIndex,
) -> pd.Series:
    """
    Liest ein pv-Erzeugungsprofil aus CSV ein, konvertiert es nach kWh/step und skaliert es auf pv_system_size_kwp.
    value_column_one_based bezieht sich auf die Original-CSV-Spalten (inkl. Zeitspalte).
    """
    dataframe = _read_table_flex(csv_path, prefer_decimal_comma=True)

    datetime_column_name = _infer_datetime_column_name_for_signal(dataframe)

    # 1) Spaltenname anhand originaler CSV-Spalten bestimmen
    original_columns = list(dataframe.columns)
    value_column_zero_based = int(value_column_one_based) - 1

    if value_column_zero_based < 0 or value_column_zero_based >= len(original_columns):
        raise ValueError(
            f"generation_strategy_value_col={value_column_one_based} ist außerhalb der CSV-Spaltenanzahl "
            f"({len(original_columns)})."
        )

    value_column_name = original_columns[value_column_zero_based]

    # 2) Jetzt DatetimeIndex setzen
    timezone = getattr(timestamps, "timezone", None)
    dataframe = _ensure_datetime_index(dataframe, datetime_column_name, timezone=timezone)

    # 3) Werte aus der gewählten Spalte ziehen (Spalte existiert weiterhin)
    raw_values = pd.to_numeric(dataframe[value_column_name], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    generation_kwh_per_step_reference = _convert_energy_series_to_kwh_per_step(
        values=raw_values,
        unit=value_unit,
        time_resolution_min=time_resolution_min,
    )

    pv_profile_reference_kwp = float(pv_profile_reference_kwp)
    pv_system_size_kwp = float(pv_system_size_kwp)

    if pv_profile_reference_kwp <= 0.0:
        raise ValueError("pv_profile_reference_kwp muss > 0 sein.")
    if pv_system_size_kwp < 0.0:
        raise ValueError("pv_system_size_kwp muss >= 0 sein.")

    scaling_factor = pv_system_size_kwp / pv_profile_reference_kwp
    pv_generation_kwh_per_step = generation_kwh_per_step_reference * scaling_factor

    series = pd.Series(pv_generation_kwh_per_step, index=dataframe.index, name="pv_generation_kwh_per_step")
    series = _reindex_to_simulation_timestamps(series, timestamps, method="nearest")
    return series


# =============================================================================
# 2) Sampling: Sessions (Ankunft / Standzeit / SoC / Fleet-Mix)
# =============================================================================

@dataclass
class SampledSession:
    """
    Repräsentiert eine zufällig erzeugte (gesampelte) Ladesession innerhalb der Simulation.

    Eine Session beschreibt das Verhalten eines Fahrzeugs am Standort:
    - wann es ankommt und wieder abfährt (Zeitstempel),
    - welche Zeitschritt-Indizes diese Zeiten im Simulationsraster haben,
    - wie lange es parkt,
    - mit welchem Ladezustand (SoC) es ankommt,
    - welches Fahrzeugmodell und welche Fahrzeugklasse betroffen sind,
    - sowie den Tagtyp (z.B. Werktag/Samstag/Sonn- bzw. Feiertag), der für die Sampling-Logik genutzt wird.
    """
    session_id: str
    arrival_time: datetime
    departure_time: datetime
    arrival_step: int
    departure_step: int
    duration_minutes: float
    state_of_charge_at_arrival: float
    vehicle_name: str
    vehicle_class: str
    day_type: str


def _sample_uniform_from_range(value_or_range, random_generator: np.random.Generator) -> float:
    """
    Zieht entweder einen konstanten Wert oder einen Uniform-Sample aus einem [min, max]-Intervall.
    """
    if isinstance(value_or_range, list) and len(value_or_range) == 2:
        return float(random_generator.uniform(float(value_or_range[0]), float(value_or_range[1])))
    return float(value_or_range)


def _sample_from_distribution_component(component: dict, random_generator: np.random.Generator) -> float:
    """
    Zieht einen Sample-Wert aus einer einzelnen Verteilungs-Komponente (normal/beta/lognormal).
    """
    distribution_name = str(component.get("distribution", "")).strip().lower()

    if distribution_name == "normal":
        mean_value = _sample_uniform_from_range(component.get("mu"), random_generator)  # YAML nutzt "mu"
        standard_deviation = _sample_uniform_from_range(component.get("sigma"), random_generator)  # YAML nutzt "sigma"
        standard_deviation = max(float(standard_deviation), 1e-9)
        return float(random_generator.normal(loc=float(mean_value), scale=float(standard_deviation)))

    if distribution_name == "beta":
        alpha_value = _sample_uniform_from_range(component.get("alpha"), random_generator)
        beta_value = _sample_uniform_from_range(component.get("beta"), random_generator)
        alpha_value = max(float(alpha_value), 1e-9)
        beta_value = max(float(beta_value), 1e-9)
        return float(random_generator.beta(a=alpha_value, b=beta_value))

    if distribution_name == "lognormal":
        mean_value = _sample_uniform_from_range(component.get("mu"), random_generator)  # YAML nutzt "mu"
        standard_deviation = _sample_uniform_from_range(component.get("sigma"), random_generator)  # YAML nutzt "sigma"
        standard_deviation = max(float(standard_deviation), 1e-9)
        return float(random_generator.lognormal(mean=float(mean_value), sigma=float(standard_deviation)))

    raise ValueError(f"Unbekannte distribution in Component: '{distribution_name}'")


def sample_from_distribution_specification(distribution_specification: dict, random_generator: np.random.Generator) -> float:
    """
    Zieht einen Sample-Wert aus einer Mixture-Spezifikation oder direkt aus einer einzelnen Komponente.
    Erwartet:
      { type: mixture, components: [ {distribution:..., weight:...}, ... ] }
    oder direkt eine einzelne Komponente (normal/beta/lognormal).
    """
    specification_type = str(distribution_specification.get("type", "")).strip().lower()
    if specification_type == "mixture":
        components = distribution_specification.get("components", [])
        if not isinstance(components, list) or len(components) == 0:
            raise ValueError("mixture benötigt eine nicht-leere Liste 'components'")

        component_weights = np.array([float(component.get("weight", 1.0)) for component in components], dtype=float)
        component_weights = np.maximum(component_weights, 0.0)
        weight_sum = float(np.sum(component_weights))
        if weight_sum <= 0.0:
            raise ValueError("mixture: Gewichtssumme ist 0")
        component_weights = component_weights / weight_sum

        chosen_component_index = int(random_generator.choice(len(components), p=component_weights))
        chosen_component = components[chosen_component_index]
        return _sample_from_distribution_component(chosen_component, random_generator)

    return _sample_from_distribution_component(distribution_specification, random_generator)


def _get_day_type(simulation_day_start: datetime, holiday_dates: List[datetime]) -> str:
    """
    Klassifiziert einen Tag als working_day / saturday / sunday_holiday (inkl. expliziter Feiertage).
    """
    date_only = simulation_day_start.date()
    if any(date_only == holiday_date.date() for holiday_date in holiday_dates):
        return "sunday_holiday"

    weekday_index = int(simulation_day_start.weekday())  # 0=Mo .. 6=So
    if weekday_index == 5:
        return "saturday"
    if weekday_index == 6:
        return "sunday_holiday"
    return "working_day"


def _sample_vehicle_by_class(
    vehicle_curves_by_name: Dict[str, VehicleChargingCurve],
    fleet_mix: dict,
    random_generator: np.random.Generator,
) -> Tuple[str, str]:
    """
    Wählt ein Fahrzeug anhand des Fleet-Mix:
      1) Klasse nach fleet_mix Gewichten
      2) Fahrzeug gleichverteilt innerhalb der Klasse
    """
    available_classes = sorted({curve.vehicle_class for curve in vehicle_curves_by_name.values()})
    if len(available_classes) == 0:
        raise ValueError("Keine Fahrzeugklassen in vehicle_curves_by_name gefunden.")

    class_names_from_yaml = [class_name for class_name in fleet_mix.keys() if class_name in available_classes]
    if len(class_names_from_yaml) == 0:
        # Wenn YAML-Klassen nicht matchen: alle verfügbaren Klassen gleichgewichtet verwenden.
        class_names = available_classes
        class_weights = np.ones(len(class_names), dtype=float)
    else:
        class_names = class_names_from_yaml
        class_weights = np.array([float(fleet_mix[class_name]) for class_name in class_names], dtype=float)

    class_weights = np.maximum(class_weights, 0.0)
    if float(np.sum(class_weights)) <= 0.0:
        class_weights = np.ones(len(class_names), dtype=float)
    class_weights = class_weights / float(np.sum(class_weights))

    chosen_class = str(random_generator.choice(class_names, p=class_weights))

    vehicle_names_in_class = [
        curve.vehicle_name
        for curve in vehicle_curves_by_name.values()
        if str(curve.vehicle_class).strip() == chosen_class
    ]
    if len(vehicle_names_in_class) == 0:
        # Fallback: irgendein Fahrzeug (sollte nur in inkonsistenten Datenfällen passieren).
        vehicle_names_in_class = list(vehicle_curves_by_name.keys())

    chosen_vehicle_name = str(random_generator.choice(vehicle_names_in_class))
    return chosen_vehicle_name, chosen_class


def sample_sessions_for_simulation_day(
    scenario: dict,
    simulation_day_start: datetime,
    timestamps: pd.DatetimeIndex,
    holiday_dates: List[datetime],
    vehicle_curves_by_name: Dict[str, VehicleChargingCurve],
    random_generator: np.random.Generator,
    day_index: int,
) -> List[SampledSession]:
    """
    Erzeugt Sessions für genau einen Tag.
    """
    time_resolution_min = int(scenario["time_resolution_min"])
    
    timezone = getattr(timestamps, "timezone", None)
    day_start = pd.Timestamp(simulation_day_start)
    if timezone is not None:
        if day_start.tzinfo is None:
            day_start = day_start.tz_localize(timezone)
        else:
            day_start = day_start.tz_convert(timezone)

    day_key = day_start.normalize()
    day_timestamps = timestamps[timestamps.normalize() == day_key]
    steps_per_day = int(len(day_timestamps))
    if steps_per_day <= 0:
        return []
    day_start_abs_step = int(timestamps.get_loc(day_timestamps[0]))

    site_configuration = scenario["site"]
    number_chargers = int(site_configuration["number_chargers"])

    expected_sessions_range = site_configuration.get("expected_sessions_per_charger_per_day", [1.0, 1.0])
    expected_sessions_per_charger = float(
        random_generator.uniform(float(expected_sessions_range[0]), float(expected_sessions_range[1]))
    )
    expected_sessions_per_charger = max(expected_sessions_per_charger, 0.0)

    day_type = _get_day_type(simulation_day_start, holiday_dates)

    arrival_time_distribution = scenario.get("arrival_time_distribution", {}) or {}
    weekday_weight_spec = (arrival_time_distribution.get("weekday_weight", {}) or {}).get(day_type, [1.0, 1.0])
    weekday_weight = float(random_generator.uniform(float(weekday_weight_spec[0]), float(weekday_weight_spec[1])))
    weekday_weight = max(weekday_weight, 0.0)

    expected_total_sessions = expected_sessions_per_charger * float(number_chargers) * weekday_weight
    expected_total_sessions = max(expected_total_sessions, 0.0)
    number_sessions = int(random_generator.poisson(lam=expected_total_sessions))

    components_per_weekday = arrival_time_distribution.get("components_per_weekday", {}) or {}
    arrival_components = components_per_weekday.get(day_type, [])
    if number_sessions > 0 and (not isinstance(arrival_components, list) or len(arrival_components) == 0):
        return []

    parking_duration_distribution = scenario.get("parking_duration_distribution", {}) or {}
    parking_duration_components = parking_duration_distribution.get("components", []) or []
    min_duration_minutes = float(parking_duration_distribution.get("min_duration_minutes", 0.0))
    max_duration_minutes = float(parking_duration_distribution.get("max_duration_minutes", 24 * 60))

    state_of_charge_distribution = scenario.get("soc_at_arrival_distribution", {}) or {}
    state_of_charge_components = state_of_charge_distribution.get("components", []) or []
    max_state_of_charge = float(state_of_charge_distribution.get("max_soc", 1.0))  # YAML nutzt "max_soc"

    vehicles_section = scenario.get("vehicles", {}) or {}
    fleet_mix = vehicles_section.get("fleet_mix", {}) or {}

    allow_cross_day_charging = bool(site_configuration.get("allow_cross_day_charging", False))

    sampled_sessions: List[SampledSession] = []

    for session_number in range(number_sessions):
        arrival_component_specification = {"type": "mixture", "components": arrival_components}
        arrival_hours = sample_from_distribution_specification(arrival_component_specification, random_generator)

        arrival_minutes = float(arrival_hours) * 60.0
        arrival_minutes = float(np.clip(arrival_minutes, 0.0, 24.0 * 60.0 - 1e-9))

        arrival_step = int(np.floor(arrival_minutes / float(time_resolution_min)))
        arrival_step = int(np.clip(arrival_step, 0, steps_per_day - 1))
        arrival_abs_step = day_start_abs_step + int(arrival_step)
        arrival_time = pd.to_datetime(timestamps[arrival_abs_step]).to_pydatetime()

        duration_minutes = sample_from_distribution_specification(
            {"type": "mixture", "components": parking_duration_components},
            random_generator,
        )
        duration_minutes = float(np.clip(duration_minutes, min_duration_minutes, max_duration_minutes))

        departure_time = arrival_time + timedelta(minutes=float(duration_minutes))
        departure_minutes_from_midnight = (departure_time - simulation_day_start).total_seconds() / 60.0
        departure_step = int(np.ceil(departure_minutes_from_midnight / float(time_resolution_min)))
        departure_step = int(np.clip(departure_step, arrival_step + 1, steps_per_day))

        if not allow_cross_day_charging:
            departure_step = int(min(departure_step, steps_per_day))

        departure_abs_step = day_start_abs_step + int(departure_step)

        # departure_step may equal steps_per_day (exclusive end) -> could be first step of next day
        if departure_abs_step < len(timestamps):
            departure_time = pd.to_datetime(timestamps[departure_abs_step]).to_pydatetime()
        else:
            # end boundary outside horizon: set to last timestamp + one step
            departure_time = (pd.to_datetime(timestamps[-1]) + pd.Timedelta(minutes=time_resolution_min)).to_pydatetime()

        sampled_state_of_charge = sample_from_distribution_specification(
            {"type": "mixture", "components": state_of_charge_components},
            random_generator,
        )
        state_of_charge_at_arrival = float(np.clip(sampled_state_of_charge, 0.0, max_state_of_charge))

        vehicle_name, vehicle_class = _sample_vehicle_by_class(
            vehicle_curves_by_name=vehicle_curves_by_name,
            fleet_mix=fleet_mix,
            random_generator=random_generator,
        )

        session_id = f"{simulation_day_start.date()}_{day_index:03d}_{session_number:05d}"

        sampled_sessions.append(
            SampledSession(
                session_id=session_id,
                arrival_time=arrival_time,
                departure_time=departure_time,
                arrival_step=int(arrival_step),
                departure_step=int(departure_step),
                duration_minutes=float(duration_minutes),
                state_of_charge_at_arrival=float(state_of_charge_at_arrival),
                vehicle_name=str(vehicle_name),
                vehicle_class=str(vehicle_class),
                day_type=str(day_type),
            )
        )

    sampled_sessions = sorted(sampled_sessions, key=lambda session: session.arrival_time)
    return sampled_sessions



# =============================================================================
# 3) Strategie: Reservierungsbasierte Session-Planung (immediate / market / generation)
# =============================================================================

def _charger_limit_site_kwh_per_step(scenario: dict) -> float:
    """
    Maximale abgebbare Energie pro Zeitschritt am Standort (Charger-Leistungsgrenze pro Ladepunkt).
    Hinweis: Im Modell wird die Charger-Grenze pro Session/Ladepunkt angewendet.
    """
    return float(scenario["site"]["rated_power_kw"]) * _step_hours(int(scenario["time_resolution_min"]))


def _grid_limit_site_kwh_per_step(scenario: dict) -> float:
    """
    Maximale beziehbare Netzenergie pro Zeitschritt am Standort (Grid-Limit gesamt Standort).
    """
    return float(scenario["site"]["grid_limit_p_avb_kw"]) * _step_hours(int(scenario["time_resolution_min"]))


def _vehicle_site_limit_kwh_per_step_from_curve(
    curve: VehicleChargingCurve,
    state_of_charge_fraction: float,
    scenario: dict,
) -> float:
    """
    Fahrzeuglimit pro Schritt auf Standortseite (kWh/step) aus der Master-Ladekurve,
    unter Berücksichtigung des Charger-Wirkungsgrads.
    """
    time_resolution_min = int(scenario["time_resolution_min"])

    state_of_charge_fraction = float(np.clip(state_of_charge_fraction, 0.0, 1.0))

    power_kw_at_battery = float(np.interp(state_of_charge_fraction, curve.state_of_charge_fraction, curve.power_kw))
    power_kw_at_battery = max(power_kw_at_battery, 0.0)

    power_kw_site = power_kw_at_battery
    return max(power_kw_site, 0.0) * _step_hours(time_resolution_min)


def _required_battery_energy_kwh(
    curve: VehicleChargingCurve,
    state_of_charge_at_arrival: float,
    state_of_charge_target: float,
) -> float:
    """
    Benötigte Batterie-Energie (kWh) von Ankunfts-SoC auf Ziel-SoC.
    """
    state_of_charge_start = float(np.clip(state_of_charge_at_arrival, 0.0, 1.0))
    state_of_charge_target = float(np.clip(state_of_charge_target, 0.0, 1.0))
    if state_of_charge_target <= state_of_charge_start:
        return 0.0
    return float(curve.battery_capacity_kwh) * (state_of_charge_target - state_of_charge_start)


def _available_site_energy_kwh_for_new_reservation(
    pv_generation_kwh_per_step: np.ndarray,
    base_load_kwh_per_step: np.ndarray,
    reserved_total_ev_energy_kwh_per_step: np.ndarray,
    scenario: dict,
    step_index: int,
) -> float:
    """
    Verfügbare Standort-Energie (PV + Netz) für neue Reservierungen in einem Schritt (kWh/step),
    nach Abzug Grundlast und bereits reservierter EV-Energie.
    """
    grid_limit_kwh_per_step = _grid_limit_site_kwh_per_step(scenario)
    supply_headroom = (
        float(pv_generation_kwh_per_step[step_index])
        + float(grid_limit_kwh_per_step)
        - float(base_load_kwh_per_step[step_index])
        - float(reserved_total_ev_energy_kwh_per_step[step_index])
    )
    return max(supply_headroom, 0.0)


def _available_grid_only_energy_kwh_for_new_reservation(
    base_load_kwh_per_step: np.ndarray,
    reserved_total_ev_energy_kwh_per_step: np.ndarray,
    scenario: dict,
    step_index: int,
) -> float:
    """
    Verfügbare Netz-Energie für neue Reservierungen (kWh/step),
    nach Abzug Grundlast und bereits reservierter EV-Energie.
    """
    grid_limit_kwh_per_step = _grid_limit_site_kwh_per_step(scenario)
    headroom = (
        float(grid_limit_kwh_per_step)
        - float(base_load_kwh_per_step[step_index])
        - float(reserved_total_ev_energy_kwh_per_step[step_index])
    )
    return max(headroom, 0.0)


def _pv_available_site_energy_kwh_for_new_reservation(
    pv_generation_kwh_per_step: np.ndarray,
    base_load_kwh_per_step: np.ndarray,
    reserved_pv_ev_energy_kwh_per_step: np.ndarray,
    step_index: int,
) -> float:
    """
    Noch nicht reservierte PV-Energie nach Abzug Grundlast (kWh/step) in einem Schritt.
    """
    pv_after_base_load = float(pv_generation_kwh_per_step[step_index]) - float(base_load_kwh_per_step[step_index])
    pv_after_base_load = max(pv_after_base_load, 0.0)

    pv_remaining = pv_after_base_load - float(reserved_pv_ev_energy_kwh_per_step[step_index])
    return max(pv_remaining, 0.0)


def _track_pv_share_for_slot(
    allocated_site_kwh: float,
    pv_generation_kwh_per_step: np.ndarray,
    base_load_kwh_per_step: np.ndarray,
    reserved_pv_ev_energy_kwh_per_step: np.ndarray,
    step_index: int,
) -> float:
    """
    PV-Anteil einer Slot-Allokation nach Regel:
    "PV wird physikalisch zuerst genutzt", d.h. PV-Anteil = min(Allocated, PV-Remaining-after-base-and-PV-reservations).
    """
    pv_available_kwh = _pv_available_site_energy_kwh_for_new_reservation(
        pv_generation_kwh_per_step=pv_generation_kwh_per_step,
        base_load_kwh_per_step=base_load_kwh_per_step,
        reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
        step_index=int(step_index),
    )
    pv_share_kwh = float(np.minimum(float(allocated_site_kwh), float(pv_available_kwh)))
    return max(pv_share_kwh, 0.0)


# -----------------------------------------------------------------------------
# Schritt-Reihenfolge (immediate / market)
# -----------------------------------------------------------------------------

def _get_charging_step_order_immediate(session_arrival_step: int, session_departure_step: int) -> np.ndarray:
    """
    Gibt die Zeitschritt-Reihenfolge für die Strategie „immediate“ zurück.

    Es wird strikt chronologisch geladen: vom Ankunftsschritt bis zum Abfahrtsschritt
    (Abfahrtsschritt ist exklusiv). Dadurch wird die Energie so früh wie möglich in der
    Standzeit eingeplant.
    """
    return np.arange(int(session_arrival_step), int(session_departure_step), dtype=int)


def _get_charging_step_order_market(
    session_arrival_step: int,
    session_departure_step: int,
    market_price_eur_per_kwh: Optional[np.ndarray],
) -> np.ndarray:
    """
    Gibt die Zeitschritt-Reihenfolge für die Strategie „market“ zurück.

    Grundidee:
    - Innerhalb der Standzeit werden alle möglichen Zeitschritte betrachtet.
    - Wenn ein Marktpreissignal vorhanden ist, werden diese Zeitschritte nach Preis sortiert
      (günstigste zuerst), damit die Reservierungslogik bevorzugt in billigen Slots lädt.
    - Wenn kein Preissignal vorhanden ist, wird als Fallback chronologisch geladen.

    Hinweis:
    Diese Funktion bestimmt nur die Sortierung der Slots. Die physikalisch korrekte
    SoC-Kopplung wird später im generischen Planner über die bereits zugewiesenen
    Energiemengen pro Zeitschritt sichergestellt.
    """
    step_indices = np.arange(int(session_arrival_step), int(session_departure_step), dtype=int)
    if market_price_eur_per_kwh is None:
        return step_indices

    prices_in_window = np.array([float(market_price_eur_per_kwh[i]) for i in step_indices], dtype=float)
    return step_indices[np.argsort(prices_in_window)]


# -----------------------------------------------------------------------------
# Generischer Reservierungs-Planner (SoC-konsistent auch bei market-order)
# -----------------------------------------------------------------------------

def _compute_state_of_charge_for_step_from_allocations(
    state_of_charge_at_arrival: float,
    curve: VehicleChargingCurve,
    scenario: dict,
    allocated_site_kwh_by_absolute_step: Dict[int, float],
    absolute_step_index: int,
) -> float:
    """
    Berechnet den Ladezustand (SoC) zu Beginn eines bestimmten Zeitschritts.

    Idee:
    Für den SoC in einem Slot ist entscheidend, wie viel Energie das Fahrzeug in allen
    vorherigen Slots bereits bekommen hat. Deshalb wird hier die bereits zugewiesene
    Energie aus `allocated_site_kwh_by_absolute_step` für alle Schritte < `absolute_step_index`
    aufsummiert und auf die Batteriekapazität umgerechnet.

    Warum ist das wichtig?
    - Bei der Market-Strategie werden Slots nach Preis sortiert geplant (nicht chronologisch).
      Trotzdem muss der SoC für jeden Slot so berechnet werden, als würde die Zeit normal
      voranschreiten – sonst könnten zu hohe Ladeleistungen bei eigentlich hohem SoC entstehen.
    - Beim Generation-Ansatz mit Grid-Fallback wird erst PV geplant und danach Netz.
      Der Netz-Plan muss dabei den SoC berücksichtigen, der durch die PV-Zuteilung schon erreicht wurde.

    Ergebnis:
    Ein SoC-Wert zwischen 0.0 und 1.0 (geclippt).
    """
    battery_capacity_kwh = max(float(curve.battery_capacity_kwh), 1e-9)

    delivered_site_kwh_before = 0.0
    for step_key, site_kwh in allocated_site_kwh_by_absolute_step.items():
        if int(step_key) < int(absolute_step_index):
            delivered_site_kwh_before += float(site_kwh)

    delivered_battery_kwh_before = delivered_site_kwh_before
    state_of_charge = float(state_of_charge_at_arrival) + delivered_battery_kwh_before / battery_capacity_kwh
    return float(np.clip(state_of_charge, 0.0, 1.0))


def _reserve_session_energy_generic(
    session_arrival_step: int,
    session_departure_step: int,
    remaining_site_energy_kwh: float,
    ordered_step_indices: np.ndarray,
    supply_headroom_function: Callable[[int], float],
    pv_share_function: Callable[[float, int], float],
    reserved_total_ev_energy_kwh_per_step: np.ndarray,
    reserved_pv_ev_energy_kwh_per_step: np.ndarray,
    curve: VehicleChargingCurve,
    state_of_charge_at_arrival: float,
    scenario: dict,
    initial_allocations_site_kwh_by_absolute_step: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, float, Dict[int, float]]:
    """
    Universeller Reservierungs-Planner:
    - plant kWh/step in reserved_total_ev_energy_kwh_per_step
    - trackt optional PV-Anteil in reserved_pv_ev_energy_kwh_per_step
    - SoC-Limit ist physikalisch korrekt (siehe _compute_state_of_charge_for_step_from_allocations)

    Rückgabe:
      plan_site_kwh_per_step, remaining_site_energy_kwh, allocations_dict(absolute_step->allocated_site_kwh)
    """
    number_steps = int(session_departure_step) - int(session_arrival_step)
    plan_site_kwh_per_step = np.zeros(number_steps, dtype=float)

    if remaining_site_energy_kwh <= 0.0 or number_steps <= 0:
        return (
            plan_site_kwh_per_step,
            float(remaining_site_energy_kwh),
            (initial_allocations_site_kwh_by_absolute_step or {}),
        )

    charger_limit_kwh_per_step = _charger_limit_site_kwh_per_step(scenario)

    allocations: Dict[int, float] = dict(initial_allocations_site_kwh_by_absolute_step or {})

    for absolute_step_index in ordered_step_indices:
        if remaining_site_energy_kwh <= 1e-12:
            break

        relative_step_index = int(absolute_step_index - int(session_arrival_step))
        if relative_step_index < 0 or relative_step_index >= number_steps:
            continue

        current_state_of_charge = _compute_state_of_charge_for_step_from_allocations(
            state_of_charge_at_arrival=float(state_of_charge_at_arrival),
            curve=curve,
            scenario=scenario,
            allocated_site_kwh_by_absolute_step=allocations,
            absolute_step_index=int(absolute_step_index),
        )

        vehicle_limit_kwh_per_step = _vehicle_site_limit_kwh_per_step_from_curve(
            curve=curve,
            state_of_charge_fraction=float(current_state_of_charge),
            scenario=scenario,
        )

        supply_headroom_kwh_per_step = float(supply_headroom_function(int(absolute_step_index)))

        already_allocated_site_kwh = float(allocations.get(int(absolute_step_index), 0.0))
        charger_headroom_kwh_per_step = max(charger_limit_kwh_per_step - already_allocated_site_kwh, 0.0)

        allocated_site_kwh = float(
            np.minimum(
                np.minimum(
                    np.minimum(supply_headroom_kwh_per_step, charger_headroom_kwh_per_step),
                    vehicle_limit_kwh_per_step,
                ),
                remaining_site_energy_kwh,
            )
        )
        allocated_site_kwh = max(allocated_site_kwh, 0.0)

        if allocated_site_kwh <= 0.0:
            continue

        plan_site_kwh_per_step[relative_step_index] += allocated_site_kwh
        reserved_total_ev_energy_kwh_per_step[int(absolute_step_index)] += allocated_site_kwh
        allocations[int(absolute_step_index)] = float(allocations.get(int(absolute_step_index), 0.0)) + allocated_site_kwh

        pv_share_kwh = float(pv_share_function(float(allocated_site_kwh), int(absolute_step_index)))
        pv_share_kwh = max(pv_share_kwh, 0.0)
        if pv_share_kwh > 0.0:
            reserved_pv_ev_energy_kwh_per_step[int(absolute_step_index)] += pv_share_kwh

        remaining_site_energy_kwh -= allocated_site_kwh

    return plan_site_kwh_per_step, float(remaining_site_energy_kwh), allocations

# -----------------------------------------------------------------------------
# Strategy Planner: immediate / market
# -----------------------------------------------------------------------------

def plan_charging_immediate(
    session_arrival_step: int,
    session_departure_step: int,
    remaining_site_energy_kwh: float,
    pv_generation_kwh_per_step: np.ndarray,
    base_load_kwh_per_step: np.ndarray,
    reserved_total_ev_energy_kwh_per_step: np.ndarray,
    reserved_pv_ev_energy_kwh_per_step: np.ndarray,
    curve: VehicleChargingCurve,
    state_of_charge_at_arrival: float,
    scenario: dict,
) -> Tuple[np.ndarray, float]:
    """
    Immediate:
    - lädt ab Ankunft so früh wie möglich
    - nutzt insgesamt verfügbare Energie (PV + Netz)
    - PV wird in jedem Slot physikalisch zuerst genutzt (Tracking via _track_pv_share_for_slot)
    """
    ordered_steps = _get_charging_step_order_immediate(session_arrival_step, session_departure_step)

    def supply_headroom_function(step_index: int) -> float:
        return _available_site_energy_kwh_for_new_reservation(
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
            scenario=scenario,
            step_index=int(step_index),
        )

    def pv_share_function(allocated_site_kwh: float, step_index: int) -> float:
        return _track_pv_share_for_slot(
            allocated_site_kwh=float(allocated_site_kwh),
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
            step_index=int(step_index),
        )

    plan, remaining, _ = _reserve_session_energy_generic(
        session_arrival_step=session_arrival_step,
        session_departure_step=session_departure_step,
        remaining_site_energy_kwh=remaining_site_energy_kwh,
        ordered_step_indices=ordered_steps,
        supply_headroom_function=supply_headroom_function,
        pv_share_function=pv_share_function,
        reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
        reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
        curve=curve,
        state_of_charge_at_arrival=float(state_of_charge_at_arrival),
        scenario=scenario,
        initial_allocations_site_kwh_by_absolute_step=None,
    )
    return plan, remaining


def plan_charging_market_price_optimized(
    session_arrival_step: int,
    session_departure_step: int,
    remaining_site_energy_kwh: float,
    market_price_eur_per_kwh: Optional[np.ndarray],
    pv_generation_kwh_per_step: np.ndarray,
    base_load_kwh_per_step: np.ndarray,
    reserved_total_ev_energy_kwh_per_step: np.ndarray,
    reserved_pv_ev_energy_kwh_per_step: np.ndarray,
    curve: VehicleChargingCurve,
    state_of_charge_at_arrival: float,
    scenario: dict,
) -> Tuple[np.ndarray, float]:
    """
    Plant die Ladung einer Session nach der Strategie „market“ (preisoptimiert).

    Vorgehen:
    - Innerhalb der Standzeit werden alle möglichen Zeitschritte betrachtet.
    - Wenn ein Marktpreissignal vorhanden ist, werden diese Zeitschritte nach Preis sortiert
      (günstigste zuerst). Das heißt: Energie wird bevorzugt in billige Zeitfenster gelegt.
    - Wenn kein Preissignal vorhanden ist, wird als Fallback chronologisch geplant.

    Restriktionen / Physik:
    - Es wird immer die insgesamt verfügbare Standortenergie genutzt: PV + Netz,
      abzüglich Grundlast und bereits reservierter EV-Energie.
    - PV wird nicht „strategisch“ gesucht (das ist nur bei generation der Fall), aber
      bei jeder geplanten Energiemenge wird der physikalisch mögliche PV-Anteil getrackt
      („PV zuerst“ nach Grundlast).
    - Trotz der Preis-Sortierung bleibt die SoC-Logik korrekt, weil der generische
      Reservierungs-Planner den SoC für jeden Slot aus allen bereits geplanten früheren
      Slots berechnet (zeitlich konsistent).

    Rückgabe:
    - `plan`: Array mit kWh pro Session-Zeitschritt (relativ zur Session), also die geplante Energiemenge je Slot.
    - `remaining`: Restenergie (kWh), die innerhalb der Standzeit nicht mehr eingeplant werden konnte.
    """
    ordered_steps = _get_charging_step_order_market(session_arrival_step, session_departure_step, market_price_eur_per_kwh)

    def supply_headroom_function(step_index: int) -> float:
        return _available_site_energy_kwh_for_new_reservation(
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
            scenario=scenario,
            step_index=int(step_index),
        )

    def pv_share_function(allocated_site_kwh: float, step_index: int) -> float:
        return _track_pv_share_for_slot(
            allocated_site_kwh=float(allocated_site_kwh),
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
            step_index=int(step_index),
        )

    plan, remaining, _ = _reserve_session_energy_generic(
        session_arrival_step=session_arrival_step,
        session_departure_step=session_departure_step,
        remaining_site_energy_kwh=remaining_site_energy_kwh,
        ordered_step_indices=ordered_steps,
        supply_headroom_function=supply_headroom_function,
        pv_share_function=pv_share_function,
        reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
        reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
        curve=curve,
        state_of_charge_at_arrival=float(state_of_charge_at_arrival),
        scenario=scenario,
        initial_allocations_site_kwh_by_absolute_step=None,
    )
    return plan, remaining


def plan_charging_market_price_optimized_grid_only(
    session_arrival_step: int,
    session_departure_step: int,
    remaining_site_energy_kwh: float,
    market_price_eur_per_kwh: Optional[np.ndarray],
    base_load_kwh_per_step: np.ndarray,
    reserved_total_ev_energy_kwh_per_step: np.ndarray,
    curve: VehicleChargingCurve,
    state_of_charge_at_arrival: float,
    scenario: dict,
    initial_allocations_site_kwh_by_absolute_step: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, float]:
    """
    Grid-only Market-Fallback (für generation):
    - nutzt NUR Netz (keine PV-Reservierung in dieser Teilplanung)
    - wählt günstigste Slots zuerst (market)
    - SoC-Kopplung: berücksichtigt initial_allocations_site_kwh_by_absolute_step (z.B. aus PV-Plan)
    """
    ordered_steps = _get_charging_step_order_market(session_arrival_step, session_departure_step, market_price_eur_per_kwh)

    def supply_headroom_function(step_index: int) -> float:
        return _available_grid_only_energy_kwh_for_new_reservation(
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
            scenario=scenario,
            step_index=int(step_index),
        )

    def pv_share_function(allocated_site_kwh: float, step_index: int) -> float:
        return 0.0

    # PV-Tracking ist hier bewusst deaktiviert (Grid-only). Es wird ein Dummy-Array übergeben,
    # damit _reserve_session_energy_generic(...) dieselbe Signatur nutzen kann, ohne PV-Reservierungen zu verändern.
    dummy_reserved_pv = np.zeros_like(reserved_total_ev_energy_kwh_per_step, dtype=float)

    plan, remaining, _ = _reserve_session_energy_generic(
        session_arrival_step=session_arrival_step,
        session_departure_step=session_departure_step,
        remaining_site_energy_kwh=remaining_site_energy_kwh,
        ordered_step_indices=ordered_steps,
        supply_headroom_function=supply_headroom_function,
        pv_share_function=pv_share_function,
        reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
        reserved_pv_ev_energy_kwh_per_step=dummy_reserved_pv,
        curve=curve,
        state_of_charge_at_arrival=float(state_of_charge_at_arrival),
        scenario=scenario,
        initial_allocations_site_kwh_by_absolute_step=(initial_allocations_site_kwh_by_absolute_step or {}),
    )
    return plan, remaining


# -----------------------------------------------------------------------------
# Strategy Planner: generation (PV-first fair share + grid-only market fallback)
# -----------------------------------------------------------------------------

def plan_charging_pv_first_fair_share(
    session_arrival_step: int,
    session_departure_step: int,
    remaining_site_energy_kwh: float,
    pv_generation_kwh_per_step: np.ndarray,
    base_load_kwh_per_step: np.ndarray,
    reserved_total_ev_energy_kwh_per_step: np.ndarray,
    reserved_pv_ev_energy_kwh_per_step: np.ndarray,
    curve: VehicleChargingCurve,
    state_of_charge_at_arrival: float,
    scenario: dict,
) -> Tuple[np.ndarray, float, Dict[int, float]]:
    """
    Plant die Ladeenergie einer einzelnen Session nach der Strategie „PV-first fair share“.

    Idee:
    - Es wird nur PV-Energie verwendet, die nach Abzug der Grundlast noch übrig ist
      (und die nicht bereits als PV→EV für andere Sessions reserviert wurde).
    - Diese verfügbare PV wird zunächst „fair“ über alle PV-Zeitschritte innerhalb der Standzeit verteilt.
    - Falls danach noch Energiebedarf übrig ist, wird zusätzlich in den PV-stärksten Zeitschritten nachgeladen.
    - Netzbezug wird hier NICHT genutzt (der kommt ggf. später über einen separaten Grid-only-Fallback).

    Ergebnis:
    - Ein Ladeplan (kWh pro Schritt innerhalb des Session-Fensters),
    - die verbleibende nicht geplante Energie,
    - sowie ein Dict der bereits geplanten Allokationen je absolutem Schritt
      (wichtig für SoC-Kopplung in nachgelagerten Planern).
    """
    number_steps = int(session_departure_step) - int(session_arrival_step)
    plan_site_kwh_per_step = np.zeros(number_steps, dtype=float)

    if remaining_site_energy_kwh <= 0.0 or number_steps <= 0:
        return plan_site_kwh_per_step, float(remaining_site_energy_kwh), {}

    charger_limit_kwh_per_step = _charger_limit_site_kwh_per_step(scenario)

    # Merkt sich, wie viel in jedem absoluten Simulationsschritt dieser Session bereits eingeplant wurde.
    # Damit kann später ein physikalisch korrekter SoC pro Slot bestimmt werden (auch bei mehrstufiger Planung).
    allocations: Dict[int, float] = {}

    step_indices = np.arange(int(session_arrival_step), int(session_departure_step), dtype=int)

    # PV-Rest je Schritt: PV nach Grundlast minus bereits reservierter PV→EV-Anteil anderer Sessions.
    pv_available_each_step = np.array(
        [
            _pv_available_site_energy_kwh_for_new_reservation(
                pv_generation_kwh_per_step=pv_generation_kwh_per_step,
                base_load_kwh_per_step=base_load_kwh_per_step,
                reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
                step_index=int(step_index),
            )
            for step_index in step_indices
        ],
        dtype=float,
    )

    # PV-Slots sind nur die Zeitschritte, in denen tatsächlich PV-Überschuss für EV verfügbar ist.
    pv_slot_mask = pv_available_each_step > 1e-12
    pv_slot_indices = step_indices[pv_slot_mask]
    if len(pv_slot_indices) == 0:
        return plan_site_kwh_per_step, float(remaining_site_energy_kwh), allocations

    # Fair-Share: gleiche Energiemenge pro PV-Slot als „Zielwert“ (wird pro Slot durch Limits begrenzt).
    fair_share_kwh_per_step = float(remaining_site_energy_kwh) / float(len(pv_slot_indices))

    # 1) Fair Share über alle PV-Slots (chronologisch)
    for absolute_step_index in pv_slot_indices:
        if remaining_site_energy_kwh <= 1e-12:
            break

        relative_step_index = int(absolute_step_index - int(session_arrival_step))

        pv_available_kwh = _pv_available_site_energy_kwh_for_new_reservation(
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
            step_index=int(absolute_step_index),
        )
        if pv_available_kwh <= 1e-12:
            continue

        current_state_of_charge = _compute_state_of_charge_for_step_from_allocations(
            state_of_charge_at_arrival=float(state_of_charge_at_arrival),
            curve=curve,
            scenario=scenario,
            allocated_site_kwh_by_absolute_step=allocations,
            absolute_step_index=int(absolute_step_index),
        )
        vehicle_limit_kwh_per_step = _vehicle_site_limit_kwh_per_step_from_curve(
            curve, current_state_of_charge, scenario
        )

        # Standort-Headroom: schützt davor, dass durch Grundlast/Reservierungen das Standortlimit verletzt wird.
        # Auch wenn wir „PV-only“ planen wollen, bleibt die Gesamtbilanz (PV + Grid-Limit - Grundlast - EV-Reservierung) relevant.
        site_headroom_kwh_per_step = _available_site_energy_kwh_for_new_reservation(
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
            scenario=scenario,
            step_index=int(absolute_step_index),
        )

        # Effektive Allokation im Slot ist die minimale Grenze aus:
        # PV-Rest, Fair-Share-Ziel, Standort-Headroom, Charger-Limit, Fahrzeuglimit, Restbedarf.
        allocated_site_kwh = float(
            np.minimum(
                np.minimum(
                    np.minimum(np.minimum(pv_available_kwh, fair_share_kwh_per_step), site_headroom_kwh_per_step),
                    charger_limit_kwh_per_step,
                ),
                np.minimum(vehicle_limit_kwh_per_step, remaining_site_energy_kwh),
            )
        )
        allocated_site_kwh = max(allocated_site_kwh, 0.0)

        if allocated_site_kwh <= 0.0:
            continue

        plan_site_kwh_per_step[relative_step_index] += allocated_site_kwh
        reserved_total_ev_energy_kwh_per_step[int(absolute_step_index)] += allocated_site_kwh

        # PV-only: dieser Teilplan reserviert PV→EV explizit (kein Grid-Anteil).
        reserved_pv_ev_energy_kwh_per_step[int(absolute_step_index)] += allocated_site_kwh

        allocations[int(absolute_step_index)] = float(allocations.get(int(absolute_step_index), 0.0)) + allocated_site_kwh
        remaining_site_energy_kwh -= allocated_site_kwh

    if remaining_site_energy_kwh <= 1e-12:
        return plan_site_kwh_per_step, float(remaining_site_energy_kwh), allocations

    # 2) Extra-Laden in PV-starken Slots (absteigend nach PV-Verfügbarkeit)
    pv_strength = np.array(
        [
            _pv_available_site_energy_kwh_for_new_reservation(
                pv_generation_kwh_per_step=pv_generation_kwh_per_step,
                base_load_kwh_per_step=base_load_kwh_per_step,
                reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
                step_index=int(step_index),
            )
            for step_index in pv_slot_indices
        ],
        dtype=float,
    )
    pv_strength_ordered = pv_slot_indices[np.argsort(-pv_strength)]

    for absolute_step_index in pv_strength_ordered:
        if remaining_site_energy_kwh <= 1e-12:
            break

        relative_step_index = int(absolute_step_index - int(session_arrival_step))

        pv_available_kwh = _pv_available_site_energy_kwh_for_new_reservation(
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
            step_index=int(absolute_step_index),
        )
        if pv_available_kwh <= 1e-12:
            continue

        current_state_of_charge = _compute_state_of_charge_for_step_from_allocations(
            state_of_charge_at_arrival=float(state_of_charge_at_arrival),
            curve=curve,
            scenario=scenario,
            allocated_site_kwh_by_absolute_step=allocations,
            absolute_step_index=int(absolute_step_index),
        )
        vehicle_limit_kwh_per_step = _vehicle_site_limit_kwh_per_step_from_curve(
            curve, current_state_of_charge, scenario
        )

        site_headroom_kwh_per_step = _available_site_energy_kwh_for_new_reservation(
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
            scenario=scenario,
            step_index=int(absolute_step_index),
        )

        allocated_site_kwh = float(
            np.minimum(
                np.minimum(np.minimum(pv_available_kwh, site_headroom_kwh_per_step), charger_limit_kwh_per_step),
                np.minimum(vehicle_limit_kwh_per_step, remaining_site_energy_kwh),
            )
        )
        allocated_site_kwh = max(allocated_site_kwh, 0.0)

        if allocated_site_kwh <= 0.0:
            continue

        plan_site_kwh_per_step[relative_step_index] += allocated_site_kwh
        reserved_total_ev_energy_kwh_per_step[int(absolute_step_index)] += allocated_site_kwh
        reserved_pv_ev_energy_kwh_per_step[int(absolute_step_index)] += allocated_site_kwh

        allocations[int(absolute_step_index)] = float(allocations.get(int(absolute_step_index), 0.0)) + allocated_site_kwh
        remaining_site_energy_kwh -= allocated_site_kwh

    return plan_site_kwh_per_step, float(remaining_site_energy_kwh), allocations


def plan_ev_charging_session(
    session: SampledSession,
    curve: VehicleChargingCurve,
    scenario: dict,
    pv_generation_kwh_per_step: np.ndarray,
    base_load_kwh_per_step: np.ndarray,
    reserved_total_ev_energy_kwh_per_step: np.ndarray,
    reserved_pv_ev_energy_kwh_per_step: np.ndarray,
    market_price_eur_per_kwh: Optional[np.ndarray],
) -> dict:
    """
    Plant eine einzelne EV-Ladesession gemäß scenario.site.charging_strategy:
      - immediate: frühestmöglich, PV+Netz, Limits beachten
      - market: billigste Slots, PV+Netz, Limits beachten
      - generation: PV-first fair share, dann grid-only market fallback

    Für alle Strategien gilt:
      - in jedem Slot wird PV physikalisch zuerst genutzt (Tracking via reserved_pv_ev_energy_kwh_per_step),
        aber nur generation "zielt" aktiv auf PV (fair share).
    """
    charging_strategy = str(scenario["site"].get("charging_strategy", "immediate")).strip().lower()

    state_of_charge_target = float(scenario["vehicles"]["soc_target"])
    required_battery_kwh = _required_battery_energy_kwh(curve, session.state_of_charge_at_arrival, state_of_charge_target)
    required_site_kwh = required_battery_kwh
    remaining_site_kwh = float(required_site_kwh)

    plan_immediate: Optional[np.ndarray] = None
    plan_market: Optional[np.ndarray] = None
    plan_pv: Optional[np.ndarray] = None

    charged_pv_site_kwh = 0.0
    charged_market_site_kwh = 0.0
    charged_immediate_site_kwh = 0.0

    if charging_strategy == "immediate":
        plan_immediate, remaining_site_kwh = plan_charging_immediate(
            session_arrival_step=int(session.arrival_step),
            session_departure_step=int(session.departure_step),
            remaining_site_energy_kwh=float(remaining_site_kwh),
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
            reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
            curve=curve,
            state_of_charge_at_arrival=float(session.state_of_charge_at_arrival),
            scenario=scenario,
        )
        total_plan = plan_immediate
        charged_immediate_site_kwh = float(np.sum(plan_immediate))

    elif charging_strategy == "market":
        plan_market, remaining_site_kwh = plan_charging_market_price_optimized(
            session_arrival_step=int(session.arrival_step),
            session_departure_step=int(session.departure_step),
            remaining_site_energy_kwh=float(remaining_site_kwh),
            market_price_eur_per_kwh=market_price_eur_per_kwh,
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
            reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
            curve=curve,
            state_of_charge_at_arrival=float(session.state_of_charge_at_arrival),
            scenario=scenario,
        )
        total_plan = plan_market
        charged_market_site_kwh = float(np.sum(plan_market))

    elif charging_strategy == "generation":
        plan_pv, remaining_site_kwh, pv_allocations = plan_charging_pv_first_fair_share(
            session_arrival_step=int(session.arrival_step),
            session_departure_step=int(session.departure_step),
            remaining_site_energy_kwh=float(remaining_site_kwh),
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
            reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
            curve=curve,
            state_of_charge_at_arrival=float(session.state_of_charge_at_arrival),
            scenario=scenario,
        )
        charged_pv_site_kwh = float(np.sum(plan_pv))

        plan_market = np.zeros_like(plan_pv)
        if remaining_site_kwh > 1e-9:
            plan_market, remaining_site_kwh = plan_charging_market_price_optimized_grid_only(
                session_arrival_step=int(session.arrival_step),
                session_departure_step=int(session.departure_step),
                remaining_site_energy_kwh=float(remaining_site_kwh),
                market_price_eur_per_kwh=market_price_eur_per_kwh,
                base_load_kwh_per_step=base_load_kwh_per_step,
                reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
                curve=curve,
                state_of_charge_at_arrival=float(session.state_of_charge_at_arrival),
                scenario=scenario,
                initial_allocations_site_kwh_by_absolute_step=pv_allocations,  # <-- SoC-Kopplung PV -> Grid-Fallback
            )
            charged_market_site_kwh = float(np.sum(plan_market))

        total_plan = plan_pv + plan_market

    else:
        raise ValueError(f"Unbekannte charging_strategy: '{charging_strategy}'")

    return {
        "session_id": session.session_id,
        "vehicle_name": session.vehicle_name,
        "vehicle_class": session.vehicle_class,
        "arrival_step": int(session.arrival_step),
        "departure_step": int(session.departure_step),
        "required_battery_kwh": float(required_battery_kwh),
        "required_site_kwh": float(required_site_kwh),
        "charged_site_kwh": float(np.sum(total_plan)),
        "charged_pv_site_kwh": float(charged_pv_site_kwh),
        "charged_market_site_kwh": float(charged_market_site_kwh),
        "charged_immediate_site_kwh": float(charged_immediate_site_kwh),
        "remaining_site_kwh": float(remaining_site_kwh),
        "plan_site_kwh_per_step": total_plan,
        "plan_pv_site_kwh_per_step": (plan_pv if plan_pv is not None else np.zeros_like(total_plan)),
        "plan_market_site_kwh_per_step": (plan_market if plan_market is not None else np.zeros_like(total_plan)),
        "plan_immediate_site_kwh_per_step": (plan_immediate if plan_immediate is not None else np.zeros_like(total_plan)),
    }


# =============================================================================
# 4) Physik + Simulation: FCFS, Ladepunkt-Zuordnung, Reservierungs-Lastgang
# =============================================================================

@dataclass
class ChargerTraceRow:
    """
    Einzelner Trace-Datensatz für einen Ladepunkt zu einem konkreten Zeitstempel.

    Die Klasse wird typischerweise als „Log-Zeile“ verwendet, um pro Zeitschritt festzuhalten:
    - welcher Charger gerade welcher Session bzw. welchem Fahrzeug zugeordnet ist,
    - welche Leistung am Standort anliegt (gesamt) und welcher Anteil davon aus PV stammt,
    - in welchem Betriebsmodus die Leistung zustande kommt (z.B. PV-first, market, grid-only),
    - und wie hoch der Ladezustand (SoC) des Fahrzeugs zu diesem Zeitpunkt ist.
    """
    timestamp: datetime
    charger_id: int
    session_id: str
    vehicle_name: str
    power_kw_site: float
    pv_power_kw_site: float
    mode: str
    state_of_charge_fraction: float


def _find_free_charger_for_interval(
    charger_occupied_until_step: List[int],
    arrival_step: int,
    departure_step: int,
) -> Optional[int]:
    """
    Findet deterministisch den ersten freien Charger für ein Zeitintervall (FCFS, kleinste ID zuerst).
    """
    for charger_id, occupied_until_step in enumerate(charger_occupied_until_step):
        if int(occupied_until_step) <= int(arrival_step):
            return int(charger_id)
    return None


def simulate_charging_sessions_fcfs(
    sessions: List[SampledSession],
    vehicle_curves_by_name: Dict[str, VehicleChargingCurve],
    scenario: dict,
    timestamps: pd.DatetimeIndex,
    pv_generation_kwh_per_step: np.ndarray,
    base_load_kwh_per_step: np.ndarray,
    market_price_eur_per_kwh: Optional[np.ndarray],
    record_debug: bool = False,
    record_charger_traces: bool = False,
) -> Tuple[np.ndarray, List[dict], List[dict], Optional[List[dict]]]:
    """
    Simuliert den Standort-Lastgang, indem Sessions nach First-Come-First-Served (FCFS) verarbeitet werden.

    Kernpunkte:
    - Session-Zeiten (arrival_time/departure_time) werden robust auf das Simulationsraster gemappt.
    - Daraus werden arrival_step/departure_step berechnet und auf der Session gesetzt.
    - Ladepunkte werden deterministisch belegt (kleinste Charger-ID zuerst).
    - Die Energie wird als Reservierungen pro Zeitschritt geführt und am Ende in kW umgerechnet.
    """
    number_steps_total = int(len(timestamps))
    number_chargers = int(scenario["site"]["number_chargers"])
    time_resolution_min = int(scenario["time_resolution_min"])
    step_hours = _step_hours(time_resolution_min)

    reserved_total_ev_energy_kwh_per_step = np.zeros(number_steps_total, dtype=float)
    reserved_pv_ev_energy_kwh_per_step = np.zeros(number_steps_total, dtype=float)

    charger_occupied_until_step: List[int] = [0 for _ in range(number_chargers)]

    charger_traces: List[dict] = []
    debug_rows: List[dict] = []
    sessions_out: List[dict] = []

    timestamps_timezone = getattr(timestamps, "timezone", None)

    def _normalize_datetime_to_timestamps_timezone(datetime_value) -> pd.Timestamp:
        """
        Normalisiert eine Zeitangabe auf:
        - gleiche Timezone wie timestamps
        - Minutenraster (floor auf time_resolution_min)
        - keine Sekunden/Mikrosekunden
        """
        timestamp_value = pd.Timestamp(datetime_value)

        if timestamps_timezone is not None:
            if timestamp_value.tzinfo is None:
                timestamp_value = timestamp_value.tz_localize(timestamps_timezone)
            else:
                timestamp_value = timestamp_value.tz_convert(timestamps_timezone)
        else:
            if timestamp_value.tzinfo is not None:
                timestamp_value = timestamp_value.tz_convert(None).tz_localize(None)

        timestamp_value = timestamp_value.replace(second=0, microsecond=0)
        timestamp_value = timestamp_value.floor(f"{time_resolution_min}min")
        return timestamp_value

    def _map_datetime_to_simulation_step(datetime_value) -> int:
        """
        Mappt eine Zeit auf einen gültigen Simulationsschritt-Index [0, number_steps_total-1].

        Vorgehen:
        - Zeit normalisieren (Timezone + Raster)
        - exakten Treffer in timestamps suchen
        - falls nicht enthalten: nearest index bestimmen (robust)
        - auf Grenzen clampen
        """
        normalized_timestamp = _normalize_datetime_to_timestamps_timezone(datetime_value)

        try:
            position = timestamps.get_loc(normalized_timestamp)
            if isinstance(position, slice):
                # falls duplicates: nimm den ersten Index der Slice
                step_index = int(position.start)
            elif isinstance(position, (np.ndarray, list)):
                # selten: mehrere Treffer
                step_index = int(position[0])
            else:
                step_index = int(position)
        except KeyError:
            # robust: nearest
            nearest_positions = timestamps.get_indexer([normalized_timestamp], method="nearest")
            step_index = int(nearest_positions[0])

        if step_index < 0:
            step_index = 0
        if step_index >= number_steps_total:
            step_index = number_steps_total - 1

        return int(step_index)

    sessions_sorted = sorted(
        sessions,
        key=lambda session: pd.Timestamp(session.arrival_time) if getattr(session, "arrival_time", None) is not None else pd.Timestamp.min,
    )

    for session in sessions_sorted:
        if getattr(session, "arrival_time", None) is None or getattr(session, "departure_time", None) is None:
            sessions_out.append(
                {
                    "session_id": session.session_id,
                    "vehicle_name": session.vehicle_name,
                    "vehicle_class": session.vehicle_class,
                    "_plug_in_time": None,
                    "_plug_out_time": None,
                    "arrival_time": getattr(session, "arrival_time", None),
                    "departure_time": getattr(session, "departure_time", None),
                    "arrival_step": None,
                    "departure_step": None,
                    "charger_id": None,
                    "status": "invalid_time",
                    "charged_site_kwh": 0.0,
                    "charged_pv_site_kwh": 0.0,
                    "charged_market_site_kwh": 0.0,
                    "charged_immediate_site_kwh": 0.0,
                    "remaining_site_kwh": None,
                    "state_of_charge_at_arrival": float(session.state_of_charge_at_arrival),
                    "state_of_charge_end": float(session.state_of_charge_at_arrival),
                }
            )
            continue

        arrival_step = _map_datetime_to_simulation_step(session.arrival_time)
        departure_step = _map_datetime_to_simulation_step(session.departure_time)

        # Sicherstellen: departure_step muss nach arrival_step liegen
        if int(departure_step) <= int(arrival_step):
            departure_step = min(int(arrival_step) + 1, number_steps_total - 1)

        session.arrival_step = int(arrival_step)
        session.departure_step = int(departure_step)

        chosen_charger_id = _find_free_charger_for_interval(
            charger_occupied_until_step=charger_occupied_until_step,
            arrival_step=int(session.arrival_step),
            departure_step=int(session.departure_step),
        )

        if chosen_charger_id is None:
            sessions_out.append(
                {
                    "session_id": session.session_id,
                    "vehicle_name": session.vehicle_name,
                    "vehicle_class": session.vehicle_class,
                    "_plug_in_time": None,
                    "_plug_out_time": None,
                    "arrival_time": session.arrival_time,
                    "departure_time": session.departure_time,
                    "arrival_step": int(session.arrival_step),
                    "departure_step": int(session.departure_step),
                    "charger_id": None,
                    "status": "drive_off",
                    "charged_site_kwh": 0.0,
                    "charged_pv_site_kwh": 0.0,
                    "charged_market_site_kwh": 0.0,
                    "charged_immediate_site_kwh": 0.0,
                    "remaining_site_kwh": None,
                    "state_of_charge_at_arrival": float(session.state_of_charge_at_arrival),
                    "state_of_charge_end": float(session.state_of_charge_at_arrival),
                }
            )
            continue

        charger_occupied_until_step[int(chosen_charger_id)] = int(session.departure_step)

        curve = vehicle_curves_by_name.get(session.vehicle_name)
        if curve is None:
            raise ValueError(f"Fahrzeugkurve nicht gefunden für vehicle_name='{session.vehicle_name}'")

        plan_result = plan_ev_charging_session(
            session=session,
            curve=curve,
            scenario=scenario,
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
            reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
            market_price_eur_per_kwh=market_price_eur_per_kwh,
        )

        charged_site_kwh = float(plan_result["charged_site_kwh"])
        charged_battery_kwh = charged_site_kwh

        state_of_charge_end = float(
            np.clip(
                float(session.state_of_charge_at_arrival)
                + charged_battery_kwh / max(float(curve.battery_capacity_kwh), 1e-9),
                0.0,
                1.0,
            )
        )

        sessions_out.append(
            {
                "session_id": session.session_id,
                "vehicle_name": session.vehicle_name,
                "vehicle_class": session.vehicle_class,
                "_plug_in_time": session.arrival_time,
                "_plug_out_time": session.departure_time,
                "arrival_time": session.arrival_time,
                "departure_time": session.departure_time,
                "arrival_step": int(session.arrival_step),
                "departure_step": int(session.departure_step),
                "charger_id": int(chosen_charger_id),
                "status": "plugged",
                "required_site_kwh": float(plan_result["required_site_kwh"]),
                "required_battery_kwh": float(plan_result["required_battery_kwh"]),
                "charged_site_kwh": float(plan_result["charged_site_kwh"]),
                "charged_pv_site_kwh": float(plan_result["charged_pv_site_kwh"]),
                "charged_market_site_kwh": float(plan_result["charged_market_site_kwh"]),
                "charged_immediate_site_kwh": float(plan_result["charged_immediate_site_kwh"]),
                "remaining_site_kwh": float(plan_result["remaining_site_kwh"]),
                "state_of_charge_at_arrival": float(session.state_of_charge_at_arrival),
                "state_of_charge_end": state_of_charge_end,
                "plan_site_kwh_per_step": plan_result["plan_site_kwh_per_step"],
                "plan_pv_site_kwh_per_step": plan_result["plan_pv_site_kwh_per_step"],
                "plan_market_site_kwh_per_step": plan_result["plan_market_site_kwh_per_step"],
                "plan_immediate_site_kwh_per_step": plan_result["plan_immediate_site_kwh_per_step"],
            }
        )

        if record_charger_traces:
            plan_site_kwh_per_step = np.array(plan_result["plan_site_kwh_per_step"], dtype=float)
            plan_pv_kwh_per_step = np.array(plan_result["plan_pv_site_kwh_per_step"], dtype=float)

            charging_strategy = str(scenario["site"].get("charging_strategy", "immediate")).strip().lower()

            battery_added_cumulative_kwh = np.cumsum(plan_site_kwh_per_step)
            state_of_charge_trace = float(session.state_of_charge_at_arrival) + battery_added_cumulative_kwh / max(
                float(curve.battery_capacity_kwh),
                1e-9,
            )
            state_of_charge_trace = np.clip(state_of_charge_trace, 0.0, 1.0)

            for relative_step_index in range(len(plan_site_kwh_per_step)):
                site_kwh = float(plan_site_kwh_per_step[relative_step_index])
                if site_kwh <= 1e-12:
                    continue

                absolute_step_index = int(session.arrival_step) + int(relative_step_index)
                if absolute_step_index < 0 or absolute_step_index >= number_steps_total:
                    continue

                timestamp = pd.to_datetime(timestamps[absolute_step_index]).to_pydatetime()

                power_kw_site = site_kwh / max(step_hours, 1e-9)
                pv_power_kw_site = float(plan_pv_kwh_per_step[relative_step_index]) / max(step_hours, 1e-9)

                rated_power_kw = float(scenario["site"]["rated_power_kw"])
                max_site_kwh_this_step = rated_power_kw * step_hours

                site_kwh = float(plan_site_kwh_per_step[relative_step_index])
                if site_kwh <= 1e-12:
                    continue

                # Cap energy per step so physics + traces are consistent
                site_kwh_capped = min(max(site_kwh, 0.0), max_site_kwh_this_step)

                pv_kwh = float(plan_pv_kwh_per_step[relative_step_index])
                pv_kwh_capped = min(max(pv_kwh, 0.0), site_kwh_capped)

                power_kw_site_capped = site_kwh_capped / max(step_hours, 1e-9)
                pv_power_kw_site_capped = pv_kwh_capped / max(step_hours, 1e-9)

                charger_traces.append(
                    {
                        "timestamp": timestamp,
                        "charger_id": int(chosen_charger_id),
                        "session_id": session.session_id,
                        "vehicle_name": session.vehicle_name,
                        "power_kw": power_kw_site_capped,
                        "pv_power_kw": pv_power_kw_site_capped,
                        "mode": charging_strategy,
                        "state_of_charge": float(state_of_charge_trace[relative_step_index]),
                    }
                )


    if record_debug:
        grid_limit_kwh_per_step = _grid_limit_site_kwh_per_step(scenario)
        charger_rated_power_kw = float(scenario["site"]["rated_power_kw"])

        for step_index in range(number_steps_total):
            pv_kwh = float(pv_generation_kwh_per_step[step_index])
            base_kwh = float(base_load_kwh_per_step[step_index])
            ev_kwh = float(reserved_total_ev_energy_kwh_per_step[step_index])
            pv_ev_kwh = float(reserved_pv_ev_energy_kwh_per_step[step_index])

            pv_to_base_kwh = float(np.minimum(pv_kwh, base_kwh))
            base_remaining_kwh = base_kwh - pv_to_base_kwh

            pv_to_ev_kwh = float(np.minimum(pv_ev_kwh, max(pv_kwh - pv_to_base_kwh, 0.0)))
            ev_remaining_kwh = ev_kwh - pv_to_ev_kwh

            grid_to_base_kwh = float(np.minimum(base_remaining_kwh, grid_limit_kwh_per_step))
            grid_to_ev_kwh = float(np.minimum(ev_remaining_kwh, max(grid_limit_kwh_per_step - grid_to_base_kwh, 0.0)))

            debug_rows.append(
                {
                    "timestamp": pd.to_datetime(timestamps[step_index]),
                    "pv_generation_kwh_per_step": pv_kwh,
                    "base_load_kwh_per_step": base_kwh,
                    "ev_load_kwh_per_step": ev_kwh,
                    "pv_ev_kwh_per_step": pv_ev_kwh,
                    "pv_to_base_kwh_per_step": pv_to_base_kwh,
                    "pv_to_ev_kwh_per_step": pv_to_ev_kwh,
                    "grid_to_base_kwh_per_step": grid_to_base_kwh,
                    "grid_to_ev_kwh_per_step": grid_to_ev_kwh,
                    "grid_limit_kwh_per_step": float(grid_limit_kwh_per_step),
                    "charger_rated_power_kw": float(charger_rated_power_kw),
                }
            )

    ev_load_kw = reserved_total_ev_energy_kwh_per_step / max(step_hours, 1e-9)
    return ev_load_kw, sessions_out, debug_rows, (charger_traces if record_charger_traces else None)


# =============================================================================
# 5) Analyse / Validierung / Notebook-Helper
# =============================================================================

def build_timeseries_dataframe(
    timestamps: pd.DatetimeIndex,
    ev_load_kw: np.ndarray,
    scenario: dict,
    debug_rows: Optional[List[dict]] = None,
    generation_series: Optional[pd.Series] = None,
    market_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Baut eine zentrale Zeitreihen-Tabelle (DataFrame), die für Plots und Auswertungen im Notebook genutzt wird.

    Die Funktion erzeugt eine Zeile pro Simulations-Zeitstempel und schreibt die wichtigsten Signale in Spalten:
    - EV-Ladeleistung (ev_load_kw)
    - Grundlast (base_load_kw)
    - PV-Erzeugung (pv_generation_kw)
    - optional: Aufteilung der EV-Leistung nach Quelle (pv_to_ev_kw, grid_to_ev_kw)
    - optional: Marktpreis (market_price_eur_per_kwh)

    Wenn `debug_rows` vorhanden ist, werden PV-/Netz-Aufteilungen direkt daraus berechnet (kWh/step -> kW).
    Wenn `debug_rows` fehlt, wird stattdessen eine konstante Grundlast aus dem Szenario verwendet und
    PV-Erzeugung optional aus `generation_series` übernommen. In diesem Fall bleiben pv_to_ev_kw/grid_to_ev_kw leer.
    """
    time_resolution_min = int(scenario["time_resolution_min"])
    step_hours = _step_hours(time_resolution_min)

    dataframe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "ev_load_kw": np.array(ev_load_kw, dtype=float),
        }
    )

    if debug_rows is not None and len(debug_rows) > 0:
        debug_dataframe = pd.DataFrame(debug_rows)
        debug_dataframe = debug_dataframe.sort_values("timestamp").reset_index(drop=True)

        dataframe["base_load_kw"] = np.array(debug_dataframe["base_load_kwh_per_step"], dtype=float) / max(step_hours, 1e-9)
        dataframe["pv_generation_kw"] = np.array(debug_dataframe["pv_generation_kwh_per_step"], dtype=float) / max(step_hours, 1e-9)

        dataframe["pv_to_ev_kw"] = np.array(debug_dataframe["pv_to_ev_kwh_per_step"], dtype=float) / max(step_hours, 1e-9)
        dataframe["grid_to_ev_kw"] = np.array(debug_dataframe["grid_to_ev_kwh_per_step"], dtype=float) / max(step_hours, 1e-9)

    else:
        base_load_kw = float(scenario["site"].get("base_load_kw", 0.0))
        dataframe["base_load_kw"] = base_load_kw

        if generation_series is not None:
            dataframe["pv_generation_kw"] = np.array(generation_series, dtype=float) / max(step_hours, 1e-9)
        else:
            dataframe["pv_generation_kw"] = 0.0

        dataframe["pv_to_ev_kw"] = np.nan
        dataframe["grid_to_ev_kw"] = np.nan

    if market_series is not None:
        dataframe["market_price_eur_per_kwh"] = np.array(market_series, dtype=float)
    else:
        dataframe["market_price_eur_per_kwh"] = np.nan

    return dataframe


def summarize_sessions(sessions_out: List[dict]) -> dict:
    """
    Erstellt eine KPI-Zusammenfassung (gesamt, plugged, rejected, sowie Liste nicht erfüllter Sessions).
    """
    number_sessions_total = int(len(sessions_out))
    plugged_sessions = [session for session in sessions_out if session.get("status") == "plugged"]
    rejected_sessions = [session for session in sessions_out if session.get("status") != "plugged"]

    not_reached_rows: List[dict] = []
    for session in plugged_sessions:
        remaining_site_kwh = session.get("remaining_site_kwh", 0.0)
        if remaining_site_kwh is not None and float(remaining_site_kwh) > 1e-6:
            not_reached_rows.append(
                {
                    "session_id": session.get("session_id"),
                    "vehicle_name": session.get("vehicle_name"),
                    "charger_id": session.get("charger_id"),
                    "remaining_energy_kwh": float(remaining_site_kwh),
                    "state_of_charge_at_arrival": float(session.get("state_of_charge_at_arrival", np.nan)),
                    "state_of_charge_end": float(session.get("state_of_charge_end", np.nan)),
                }
            )

    return {
        "num_sessions_total": number_sessions_total,
        "num_sessions_plugged": int(len(plugged_sessions)),
        "num_sessions_rejected": int(len(rejected_sessions)),
        "not_reached_rows": not_reached_rows,
    }


def build_strategy_signal_series(
    scenario: dict,
    timestamps: pd.DatetimeIndex,
    charging_strategy: str,
    normalize_to_internal: bool = True,
    strategy_resolution_min: int = 15,
) -> Tuple[pd.Series, str]:
    """
    Lädt ein strategieabhängiges Signal (generation: pv, market: Preis) und gibt Serie + Achsenlabel zurück.
    """
    charging_strategy = str(charging_strategy).strip().lower()
    time_resolution_min = int(scenario["time_resolution_min"])

    if charging_strategy == "generation":
        site_configuration = scenario["site"]
        series = read_generation_profile_from_csv(
            csv_path=str(site_configuration["generation_strategy_csv"]),
            value_column_one_based=int(site_configuration["generation_strategy_value_col"]),
            value_unit=str(site_configuration["generation_strategy_unit"]),
            pv_profile_reference_kwp=float(site_configuration["pv_profile_reference_kwp"]),
            pv_system_size_kwp=float(site_configuration["pv_system_size_kwp"]),
            time_resolution_min=time_resolution_min,
            timestamps=timestamps,
        )
        if normalize_to_internal:
            return series, "PV [kWh/step]"
        return series / max(_step_hours(time_resolution_min), 1e-9), "PV [kW]"

    if charging_strategy == "market":
        site_configuration = scenario["site"]
        series = read_market_profile_from_csv(
            csv_path=str(site_configuration["market_strategy_csv"]),
            value_column_one_based=int(site_configuration["market_strategy_value_col"]),
            value_unit=str(site_configuration["market_strategy_unit"]),
            timestamps=timestamps,
        )
        return series, "Preis [€/kWh]"

    raise ValueError(f"Unknown charging_strategy for signal series: {charging_strategy}")


def build_plugged_sessions_preview_table(sessions_out: List[dict], n: int = 20) -> pd.DataFrame:
    """
    Baut eine kompakte Vorschau-Tabelle für plugged Sessions (für Notebook-Checks).
    """
    rows: List[dict] = []
    for session in sessions_out:
        if session.get("status") != "plugged":
            continue
        rows.append(
            {
                "session_id": session.get("session_id"),
                "charger_id": session.get("charger_id"),
                "vehicle_name": session.get("vehicle_name"),
                "arrival_time": session.get("arrival_time"),
                "departure_time": session.get("departure_time"),
                "state_of_charge_at_arrival": session.get("state_of_charge_at_arrival"),
                "state_of_charge_end": session.get("state_of_charge_end"),
                "charged_site_kwh": session.get("charged_site_kwh"),
                "charged_pv_site_kwh": session.get("charged_pv_site_kwh"),
                "charged_market_site_kwh": session.get("charged_market_site_kwh"),
                "charged_immediate_site_kwh": session.get("charged_immediate_site_kwh"),
                "remaining_site_kwh": session.get("remaining_site_kwh"),
            }
        )

    dataframe = pd.DataFrame(rows)
    if len(dataframe) == 0:
        return dataframe
    dataframe = dataframe.sort_values(["arrival_time", "charger_id"]).reset_index(drop=True)
    return dataframe.head(int(n))


POWER_COLUMN_CANDIDATES: List[str] = ["power_kw", "power_kw_site", "site_power_kw"]  # mögliche Spaltennamen für die Ladeleistung (kW), je nach Quelle/Export unterschiedlich benannt
SOC_COLUMN_CANDIDATES: List[str] = ["state_of_charge", "state_of_charge_fraction", "soc", "state_of_charge_trace"]  # mögliche Spaltennamen für den Ladezustand (State of Charge), je nach Quelle/Export unterschiedlich benannt


def _get_column_name(dataframe: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Findet in einem DataFrame den ersten passenden Spaltennamen aus einer priorisierten Kandidatenliste.
    So kann der Code mit unterschiedlich benannten CSV-/Trace-Spalten arbeiten, ohne dass du überall feste Namen erzwingen musst.
    """
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def _require_non_empty_dataframe(dataframe: Optional[pd.DataFrame], dataframe_name: str) -> pd.DataFrame:
    """
    Stellt sicher, dass ein DataFrame vorhanden ist und mindestens eine Zeile enthält.
    Falls nicht, wird ein gut lesbarer Fehler mit dem angegebenen Namen ausgelöst.
    """
    if dataframe is None or dataframe.empty:
        raise ValueError(f"{dataframe_name} ist leer.")
    return dataframe


def _filter_time_window(
    dataframe: pd.DataFrame,
    timestamp_column_name: str,
    start: Optional[datetime],
    end: Optional[datetime],
) -> pd.DataFrame:
    """
    Filtert ein DataFrame auf ein (inklusive) Zeitfenster [start, end] anhand einer Zeitstempel-Spalte.

    Zeitstempel werden mit `errors="coerce"` geparst; ungültige Werte werden verworfen.
    Falls nach dem Filtern keine Zeilen übrig bleiben, wird ein Fehler ausgelöst.
    """
    dataframe = dataframe.copy()

    if timestamp_column_name not in dataframe.columns:
        raise ValueError(f"Timestamp column '{timestamp_column_name}' fehlt im DataFrame.")

    dataframe[timestamp_column_name] = pd.to_datetime(dataframe[timestamp_column_name], errors="coerce")
    dataframe = dataframe.dropna(subset=[timestamp_column_name])

    if start is not None:
        dataframe = dataframe[dataframe[timestamp_column_name] >= pd.to_datetime(start)]
    if end is not None:
        dataframe = dataframe[dataframe[timestamp_column_name] <= pd.to_datetime(end)]

    if len(dataframe) == 0:
        raise ValueError("Keine Daten im gewählten Zeitfenster.")
    return dataframe


def _clean_non_empty_strings(values: List[Any]) -> List[str]:
    """
    Gibt nur die Werte zurück, die wirklich brauchbare Strings sind:
    - Typ `str`
    - nach `strip()` nicht leer
    """
    out: List[str] = []
    for value in values:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped != "":
                out.append(stripped)
    return out


def build_power_per_charger_timeseries(
    charger_traces_dataframe: pd.DataFrame,
    charger_id: int,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Erstellt eine Zeitreihe der Ladeleistung für genau einen Charger.

    Die Funktion:
    - filtert optional auf ein Zeitfenster,
    - filtert auf eine charger_id,
    - sucht automatisch die passende Leistungsspalte (z.B. power_kw / power_kw_site / site_power_kw),
    - gibt eine sauber sortierte Tabelle zurück, die im Notebook direkt geplottet werden kann.
    """
    charger_traces_dataframe = _require_non_empty_dataframe(charger_traces_dataframe, "charger_traces_dataframe")
    dataframe = _filter_time_window(charger_traces_dataframe, "timestamp", start=start, end=end)

    if "charger_id" not in dataframe.columns:
        raise ValueError("charger_traces_dataframe: Spalte 'charger_id' fehlt.")

    dataframe = dataframe[dataframe["charger_id"] == int(charger_id)]
    if len(dataframe) == 0:
        raise ValueError(f"Keine Daten für charger_id={int(charger_id)} im gewählten Zeitfenster.")

    power_column_name = _get_column_name(dataframe, POWER_COLUMN_CANDIDATES)
    if power_column_name is None:
        raise ValueError("Keine Leistungsspalte gefunden (power_kw / power_kw_site / site_power_kw).")

    columns_to_take = ["timestamp", power_column_name]
    if "session_id" in dataframe.columns:
        columns_to_take.append("session_id")
    if "vehicle_name" in dataframe.columns:
        columns_to_take.append("vehicle_name")

    out = dataframe[columns_to_take].copy()
    out = out.rename(columns={power_column_name: "power_kw"})
    out["power_kw"] = pd.to_numeric(out["power_kw"], errors="coerce").astype(float).fillna(0.0)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")

    return out


def build_ev_power_by_source_timeseries(timeseries_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt eine Zeitreihe der EV-Leistung, aufgeteilt nach Quelle (PV vs. Netz).

    Erwartet, dass im zentralen Timeseries-DataFrame bereits die Spalten für PV->EV
    und Grid->EV enthalten sind (typischerweise nur wenn Debug-Tracking aktiv war).
    """
    timeseries_dataframe = _require_non_empty_dataframe(timeseries_dataframe, "timeseries_dataframe")

    if "pv_to_ev_kw" not in timeseries_dataframe.columns:
        raise ValueError("timeseries_dataframe: Spalte 'pv_to_ev_kw' fehlt (record_debug=True?).")
    if "grid_to_ev_kw" not in timeseries_dataframe.columns:
        raise ValueError("timeseries_dataframe: Spalte 'grid_to_ev_kw' fehlt (record_debug=True?).")

    dataframe = timeseries_dataframe.copy()
    if "timestamp" not in dataframe.columns:
        raise ValueError("timeseries_dataframe: Spalte 'timestamp' fehlt.")

    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], errors="coerce")
    dataframe = dataframe.dropna(subset=["timestamp"])

    return pd.DataFrame(
        {
            "timestamp": dataframe["timestamp"],
            "ev_from_pv_kw": pd.to_numeric(dataframe["pv_to_ev_kw"], errors="coerce").astype(float).fillna(0.0),
            "ev_from_grid_kw": pd.to_numeric(dataframe["grid_to_ev_kw"], errors="coerce").astype(float).fillna(0.0),
        }
    )


def build_soc_timeseries_by_charger(
    charger_traces_dataframe: pd.DataFrame,
    charger_ids: List[int],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Dict[int, pd.DataFrame]:
    """
    Baut pro Charger eine SoC-Zeitreihe (State of Charge) für Notebook-Plots.

    Die Ausgabe enthält – falls vorhanden – session_id, damit man im Notebook Linien
    pro Session trennen kann (unplug/replug).
    """
    charger_traces_dataframe = _require_non_empty_dataframe(charger_traces_dataframe, "charger_traces_dataframe")
    dataframe = _filter_time_window(charger_traces_dataframe, "timestamp", start=start, end=end)

    if "charger_id" not in dataframe.columns:
        raise ValueError("charger_traces_dataframe: Spalte 'charger_id' fehlt.")

    soc_column_name = _get_column_name(dataframe, SOC_COLUMN_CANDIDATES)
    if soc_column_name is None:
        raise ValueError(
            "Keine SoC-Spalte gefunden (state_of_charge/state_of_charge_fraction/soc/state_of_charge_trace)."
        )

    has_session_id = "session_id" in dataframe.columns
    has_vehicle_name = "vehicle_name" in dataframe.columns

    output_by_charger_id: Dict[int, pd.DataFrame] = {}

    for charger_id in charger_ids:
        charger_dataframe = dataframe[dataframe["charger_id"] == int(charger_id)]
        if len(charger_dataframe) == 0:
            continue

        columns_to_take = ["timestamp", soc_column_name]
        if has_session_id:
            columns_to_take.append("session_id")
        if has_vehicle_name:
            columns_to_take.append("vehicle_name")

        out = charger_dataframe[columns_to_take].copy()
        out = out.rename(columns={soc_column_name: "soc"})

        out["soc"] = pd.to_numeric(out["soc"], errors="coerce").astype(float)
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp", "soc"]).sort_values("timestamp")

        output_by_charger_id[int(charger_id)] = out

    return output_by_charger_id


def validate_against_master_curves(
    charger_traces_dataframe: pd.DataFrame,
    sessions_out: List[dict],
    scenario: dict,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prüft Trace-Leistungen gegen zwei obere Grenzen:

    1) Charger-Limit aus dem Scenario (rated_power_kw)
    2) Fahrzeug-Limit aus der Master-Charging-Curve (max. Leistung abhängig vom SoC)

    Ergebnis:
    - validation_dataframe enthält pro Trace-Zeile die Limits und OK-Flags
    - violations_dataframe enthält nur die Zeilen, die mindestens ein Limit verletzen
    """
    if charger_traces_dataframe is None or len(charger_traces_dataframe) == 0:
        return pd.DataFrame(), pd.DataFrame()

    dataframe = _filter_time_window(charger_traces_dataframe, "timestamp", start=start, end=end)

    power_column_name = _get_column_name(dataframe, POWER_COLUMN_CANDIDATES)
    if power_column_name is None:
        raise ValueError(
            "charger_traces_dataframe: keine Leistungsspalte gefunden (power_kw / power_kw_site / site_power_kw)."
        )

    soc_column_name = _get_column_name(dataframe, SOC_COLUMN_CANDIDATES)
    if soc_column_name is None:
        raise ValueError("charger_traces_dataframe: keine SoC-Spalte gefunden (state_of_charge / soc / ...).")

    if "vehicle_name" not in dataframe.columns:
        raise ValueError("charger_traces_dataframe: Spalte 'vehicle_name' fehlt.")

    rated_power_kw = float(scenario["site"]["rated_power_kw"])

    vehicle_curves_by_name = read_vehicle_load_profiles_from_csv(str(scenario["vehicles"]["vehicle_curve_csv"]))

    validation_rows: List[dict] = []

    for _, row in dataframe.iterrows():
        vehicle_name = str(row["vehicle_name"])
        curve = vehicle_curves_by_name.get(vehicle_name)

        power_kw = float(pd.to_numeric(row[power_column_name], errors="coerce"))
        soc_value = float(pd.to_numeric(row[soc_column_name], errors="coerce"))

        if not np.isfinite(power_kw) or not np.isfinite(soc_value):
            continue

        charger_limit_ok = power_kw <= rated_power_kw + 1e-9

        vehicle_limit_kw = np.nan
        vehicle_limit_ok = True

        if curve is not None:
            soc_clipped = float(np.clip(soc_value, 0.0, 1.0))
            max_power_kw_from_curve = float(
                np.interp(
                    soc_clipped,
                    curve.state_of_charge_fraction,
                    curve.power_kw,
                )
            )
            max_power_kw_from_curve = max(max_power_kw_from_curve, 0.0)
            vehicle_limit_kw = max_power_kw_from_curve
            vehicle_limit_ok = power_kw <= vehicle_limit_kw + 1e-9

        validation_rows.append(
            {
                "timestamp": pd.to_datetime(row["timestamp"]),
                "charger_id": int(row["charger_id"]) if "charger_id" in row else np.nan,
                "session_id": str(row["session_id"]) if "session_id" in row else "",
                "vehicle_name": vehicle_name,
                "state_of_charge": soc_value,
                "power_kw": power_kw,
                "charger_limit_kw": rated_power_kw,
                "vehicle_limit_kw": float(vehicle_limit_kw) if np.isfinite(vehicle_limit_kw) else np.nan,
                "ok_charger_limit": bool(charger_limit_ok),
                "ok_vehicle_limit": bool(vehicle_limit_ok),
            }
        )

    validation_dataframe = pd.DataFrame(validation_rows)
    if len(validation_dataframe) == 0:
        return validation_dataframe, validation_dataframe.copy()

    violations_dataframe = validation_dataframe[
        (~validation_dataframe["ok_charger_limit"]) | (~validation_dataframe["ok_vehicle_limit"])
    ].copy()

    return validation_dataframe, violations_dataframe


def build_charger_power_heatmap_matrix(
    charger_traces_dataframe: pd.DataFrame,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Bereitet eine Heatmap-Matrix für Charger-Leistungen vor.

    Ausgabe-Format:
    - Zeilen = Charger (nach charger_id)
    - Spalten = Zeitstempel
    - Zellen = aggregierte Leistung pro Zeitpunkt (Summe, falls mehrere Einträge pro Zeitstempel existieren)
    """
    charger_traces_dataframe = _require_non_empty_dataframe(charger_traces_dataframe, "charger_traces_dataframe")
    dataframe = _filter_time_window(charger_traces_dataframe, "timestamp", start=start, end=end)

    if "charger_id" not in dataframe.columns:
        raise ValueError("charger_traces_dataframe: Spalte 'charger_id' fehlt.")

    power_column_name = _get_column_name(dataframe, POWER_COLUMN_CANDIDATES)
    if power_column_name is None:
        raise ValueError("Keine Leistungsspalte gefunden (power_kw / power_kw_site / site_power_kw).")

    pivot = dataframe.pivot_table(
        index="charger_id",
        columns="timestamp",
        values=power_column_name,
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index(axis=0).sort_index(axis=1)

    return {
        "matrix": pivot.values,
        "charger_ids": list(pivot.index),
        "timestamps": list(pivot.columns),
    }


def build_site_overview_plot_data(
    timeseries_dataframe: pd.DataFrame,
    scenario: dict,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Bereitet Daten für einen „Site Overview“-Plot vor (ohne zu plotten).

    Die Funktion filtert optional ein Zeitfenster, liest relevante Spalten (falls vorhanden)
    und berechnet außerdem hilfreiche Summen-/Limit-Serien (z.B. Gesamtlast, Grid-Limit, Gesamt-Charger-Limit).
    """
    timeseries_dataframe = _require_non_empty_dataframe(timeseries_dataframe, "timeseries_dataframe")
    dataframe = _filter_time_window(timeseries_dataframe, "timestamp", start=start, end=end)

    grid_limit_kw = float(scenario["site"]["grid_limit_p_avb_kw"])
    charger_limit_kw_total = float(scenario["site"]["rated_power_kw"]) * float(scenario["site"]["number_chargers"])

    if "base_load_kw" in dataframe.columns:
        base_load_kw = pd.to_numeric(dataframe["base_load_kw"], errors="coerce").astype(float).fillna(0.0)
    else:
        base_load_kw = pd.Series(np.zeros(len(dataframe), dtype=float), index=dataframe.index)

    if "ev_load_kw" in dataframe.columns:
        ev_load_kw = pd.to_numeric(dataframe["ev_load_kw"], errors="coerce").astype(float).fillna(0.0)
    else:
        ev_load_kw = pd.Series(np.zeros(len(dataframe), dtype=float), index=dataframe.index)

    total_load_kw = (base_load_kw + ev_load_kw).fillna(0.0)

    pv_generation_kw: Optional[pd.Series] = None
    if "pv_generation_kw" in dataframe.columns:
        pv_generation_kw = pd.to_numeric(dataframe["pv_generation_kw"], errors="coerce").astype(float).fillna(0.0)

    market_price_eur_per_kwh: Optional[pd.Series] = None
    if "market_price_eur_per_kwh" in dataframe.columns:
        market_price_eur_per_kwh = pd.to_numeric(dataframe["market_price_eur_per_kwh"], errors="coerce").astype(float)
    elif "market_price" in dataframe.columns:
        market_price_eur_per_kwh = pd.to_numeric(dataframe["market_price"], errors="coerce").astype(float)

    return {
        "dataframe": dataframe,
        "base_load_kw": base_load_kw,
        "ev_load_kw": ev_load_kw,
        "total_load_kw": total_load_kw,
        "pv_generation_kw": pv_generation_kw,
        "market_price_eur_per_kwh": market_price_eur_per_kwh,
        "grid_limit_kw": grid_limit_kw,
        "charger_limit_kw_total": charger_limit_kw_total,
    }


def build_ev_power_by_mode_timeseries_dataframe(
    timeseries_dataframe: pd.DataFrame,
    sessions_out: Optional[List[dict]] = None,
    scenario: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Erstellt eine EV-Leistungszeitreihe, aufgeteilt nach Lademodus (generation/market/immediate).

    Ablauf:
    - Wenn die Mode-Spalten schon im timeseries_dataframe existieren, werden sie direkt verwendet.
    - Sonst werden die Session-Pläne (kWh pro Step) über alle Sessions aufsummiert und in kW umgerechnet.
    """
    timeseries_dataframe = _require_non_empty_dataframe(timeseries_dataframe, "timeseries_dataframe")

    dataframe = timeseries_dataframe.copy()
    if "timestamp" not in dataframe.columns:
        raise ValueError("timeseries_dataframe: Spalte 'timestamp' fehlt.")
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], errors="coerce")
    dataframe = dataframe.dropna(subset=["timestamp"])

    has_generation = "ev_generation_kw" in dataframe.columns
    has_market = "ev_market_kw" in dataframe.columns
    has_immediate = "ev_immediate_kw" in dataframe.columns

    if has_generation and has_market and has_immediate:
        return pd.DataFrame(
            {
                "timestamp": dataframe["timestamp"],
                "ev_generation_kw": pd.to_numeric(dataframe["ev_generation_kw"], errors="coerce").astype(float).fillna(0.0),
                "ev_market_kw": pd.to_numeric(dataframe["ev_market_kw"], errors="coerce").astype(float).fillna(0.0),
                "ev_immediate_kw": pd.to_numeric(dataframe["ev_immediate_kw"], errors="coerce").astype(float).fillna(0.0),
            }
        )

    if sessions_out is None or scenario is None:
        raise ValueError(
            "Mode-Daten fehlen: Weder Mode-Spalten im timeseries_dataframe "
            "noch sessions_out+scenario für Aggregation verfügbar."
        )

    time_resolution_min = int(scenario["time_resolution_min"])
    step_hours = float(time_resolution_min) / 60.0

    number_steps_total = int(len(dataframe))
    generation_kwh_per_step = np.zeros(number_steps_total, dtype=float)
    market_kwh_per_step = np.zeros(number_steps_total, dtype=float)
    immediate_kwh_per_step = np.zeros(number_steps_total, dtype=float)

    for session in sessions_out:
        if session.get("status") != "plugged":
            continue

        arrival_step = int(session["arrival_step"])
        departure_step = int(session["departure_step"])
        window_length = int(departure_step - arrival_step)
        if window_length <= 0:
            continue

        plan_generation_kwh = np.array(session.get("plan_pv_site_kwh_per_step", []), dtype=float)
        plan_market_kwh = np.array(session.get("plan_market_site_kwh_per_step", []), dtype=float)
        plan_immediate_kwh = np.array(session.get("plan_immediate_site_kwh_per_step", []), dtype=float)

        if len(plan_generation_kwh) == window_length:
            generation_kwh_per_step[arrival_step:departure_step] += plan_generation_kwh
        if len(plan_market_kwh) == window_length:
            market_kwh_per_step[arrival_step:departure_step] += plan_market_kwh
        if len(plan_immediate_kwh) == window_length:
            immediate_kwh_per_step[arrival_step:departure_step] += plan_immediate_kwh

    generation_kw = generation_kwh_per_step / max(step_hours, 1e-12)
    market_kw = market_kwh_per_step / max(step_hours, 1e-12)
    immediate_kw = immediate_kwh_per_step / max(step_hours, 1e-12)

    return pd.DataFrame(
        {
            "timestamp": dataframe["timestamp"],
            "ev_generation_kw": generation_kw,
            "ev_market_kw": market_kw,
            "ev_immediate_kw": immediate_kw,
        }
    )


def get_most_used_vehicle_name(
    sessions_out: Optional[List[dict]] = None,
    charger_traces_dataframe: Optional[pd.DataFrame] = None,
    only_plugged_sessions: bool = True,
) -> str:
    """
    Ermittelt den am häufigsten verwendeten Fahrzeugnamen.

    Bevorzugt werden die Session-Daten (sessions_out), weil dort pro Ladevorgang genau ein vehicle_name gezählt wird.
    Falls keine nutzbaren Sessions vorhanden sind, wird als Fallback im Trace-DataFrame gezählt (vehicle_name-Häufigkeit
    über alle Trace-Zeitpunkte).
    """
    if sessions_out is not None and len(sessions_out) > 0:
        if only_plugged_sessions:
            names = [s.get("vehicle_name") for s in sessions_out if s.get("status") == "plugged"]
        else:
            names = [s.get("vehicle_name") for s in sessions_out]
        names = _clean_non_empty_strings(names)
        if len(names) > 0:
            return pd.Series(names).value_counts().idxmax()

    if charger_traces_dataframe is None or len(charger_traces_dataframe) == 0:
        raise ValueError("Weder sessions_out (nutzbar) noch charger_traces_dataframe (nicht leer) wurde übergeben.")

    if "vehicle_name" not in charger_traces_dataframe.columns:
        raise ValueError("charger_traces_dataframe: Spalte 'vehicle_name' fehlt.")

    names = _clean_non_empty_strings(list(charger_traces_dataframe["vehicle_name"].tolist()))
    if len(names) == 0:
        raise ValueError("charger_traces_dataframe: keine nutzbaren vehicle_name-Werte gefunden.")
    return pd.Series(names).value_counts().idxmax()


def build_master_curve_and_actual_points_for_vehicle(
    charger_traces_dataframe: pd.DataFrame,
    scenario: dict,
    vehicle_name: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """
    Bereitet Daten auf, um für ein Fahrzeug die Master-Charging-Curve mit realen Trace-Punkten zu vergleichen.

    Enthalten sind:
    - Masterkurve aus der Vehicle-CSV (max. Leistung in kW über dem SoC)
    - Reale Punkte aus den Charger-Traces (gemessene/zugewiesene Leistung in kW über dem SoC)
    - Eine Verletzungsmaske, wo die reale Leistung über der erlaubten Master-Leistung liegt

    Optional kann über start/end auf ein Zeitfenster gefiltert werden.
    """
    charger_traces_dataframe = _require_non_empty_dataframe(charger_traces_dataframe, "charger_traces_dataframe")
    dataframe = charger_traces_dataframe.copy()

    for column_name in ["timestamp", "vehicle_name"]:
        if column_name not in dataframe.columns:
            raise ValueError(f"charger_traces_dataframe: Spalte '{column_name}' fehlt.")

    power_column_name = _get_column_name(dataframe, POWER_COLUMN_CANDIDATES)
    if power_column_name is None:
        raise ValueError("charger_traces_dataframe: keine Leistungsspalte gefunden (power_kw/power_kw_site/site_power_kw).")

    soc_column_name = _get_column_name(dataframe, SOC_COLUMN_CANDIDATES)
    if soc_column_name is None:
        raise ValueError("charger_traces_dataframe: keine SoC-Spalte gefunden (state_of_charge/.../soc).")

    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], errors="coerce")
    dataframe = dataframe.dropna(subset=["timestamp"])

    if start is not None:
        dataframe = dataframe[dataframe["timestamp"] >= pd.to_datetime(start)]
    if end is not None:
        dataframe = dataframe[dataframe["timestamp"] <= pd.to_datetime(end)]

    dataframe = dataframe[dataframe["vehicle_name"] == str(vehicle_name)]
    if len(dataframe) == 0:
        raise ValueError(f"Keine Trace-Punkte für vehicle_name='{vehicle_name}' im gewählten Zeitfenster gefunden.")

    actual_soc = pd.to_numeric(dataframe[soc_column_name], errors="coerce").to_numpy(dtype=float)
    actual_power_kw = pd.to_numeric(dataframe[power_column_name], errors="coerce").to_numpy(dtype=float)

    valid_mask = np.isfinite(actual_soc) & np.isfinite(actual_power_kw)
    actual_soc = actual_soc[valid_mask]
    actual_power_kw = actual_power_kw[valid_mask]

    if len(actual_soc) == 0:
        raise ValueError("Alle realen Punkte sind nach der Umwandlung ungültig (SoC/Power).")

    vehicle_curves_by_name = read_vehicle_load_profiles_from_csv(str(scenario["vehicles"]["vehicle_curve_csv"]))
    if str(vehicle_name) not in vehicle_curves_by_name:
        example_names = list(vehicle_curves_by_name.keys())[:10]
        raise ValueError(f"Fahrzeug '{vehicle_name}' nicht in den Masterkurven gefunden. Beispiele: {example_names}")

    curve = vehicle_curves_by_name[str(vehicle_name)]
    master_soc = np.array(curve.state_of_charge_fraction, dtype=float)
    master_power_battery_kw = np.array(curve.power_kw, dtype=float)

    allowed_power_kw_at_actual = np.interp(actual_soc, master_soc, master_power_battery_kw)
    violation_mask = actual_power_kw > (allowed_power_kw_at_actual + 1e-9)

    return {
        "vehicle_name": str(vehicle_name),
        "master_soc": master_soc,
        "master_power_battery_kw": master_power_battery_kw,
        "actual_soc": actual_soc,
        "actual_power_kw": actual_power_kw,
        "allowed_power_kw_at_actual": allowed_power_kw_at_actual,
        "violation_mask": violation_mask,
        "number_violations": int(np.sum(violation_mask)),
        "number_points": int(len(actual_soc)),
    }


def choose_vehicle_for_master_curve_plot(
    sessions_out: List[dict],
    charger_traces_dataframe: pd.DataFrame,
    scenario: dict,
) -> Dict[str, Any]:
    """
    Wählt das insgesamt am häufigsten verwendete Fahrzeug über den kompletten Simulationshorizont
    und erstellt dafür den Vergleich „Masterkurve vs. reale Punkte“ über die gesamte Zeit.

    Wichtig:
    - Es wird bewusst NICHT auf ein Zoom-Zeitfenster geachtet.
    - Die Auswahl basiert auf der Gesamtnutzung (Sessions), nicht auf zufälligen Ausschnitten.
    """
    vehicle_name = get_most_used_vehicle_name(
        sessions_out=sessions_out,
        charger_traces_dataframe=charger_traces_dataframe,
        only_plugged_sessions=True,
    )

    plot_data = build_master_curve_and_actual_points_for_vehicle(
        charger_traces_dataframe=charger_traces_dataframe,
        scenario=scenario,
        vehicle_name=vehicle_name,
        start=None,
        end=None,
    )

    return {"vehicle_name": vehicle_name, "plot_data": plot_data}


# -----------------------------------------------------------------------------
# Notebook Helpers (kleine Utilities)
# -----------------------------------------------------------------------------

def show_strategy_status(charging_strategy: str, strategy_status: str) -> None:
    """
    Gibt einen kurzen Statusblock zur aktuell gewählten Lademanagementstrategie aus.
    """
    strategy_name = (charging_strategy or "immediate").capitalize()
    status_name = (strategy_status or "immediate").capitalize()
    print(f"Charging Strategy: {strategy_name}")
    print(f"Strategy Status: {status_name}")


def decorate_title_with_status(base_title: str, charging_strategy: str, strategy_status: str) -> str:
    """
    Erzeugt einen Plot-Titel, der den Basistitel um Strategie-Name und Status ergänzt.
    """
    strategy_name = (charging_strategy or "immediate").capitalize()
    status_name = (strategy_status or "immediate").capitalize()
    return f"{base_title} ({strategy_name}, {status_name})"


def initialize_time_window(
    timestamps: pd.DatetimeIndex,
    scenario: dict,
    days: int = 1,
) -> Tuple[Optional[int], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Erzeugt ein Zoom-Zeitfenster basierend auf den ersten N Tagen der Simulation.

    Das Fenster wird DST-/Zeitzonen-sicher bestimmt, indem das Ende über „Start + N Tage“
    berechnet wird (anstatt über eine feste Schrittanzahl pro Tag). Dadurch werden Tage mit
    23 oder 25 Stunden (Sommer-/Winterzeitumstellung) korrekt abgedeckt.
    """
    if timestamps is None or len(timestamps) == 0:
        return None, None, None

    time_resolution_min = int(scenario["time_resolution_min"])
    days = int(max(1, days))

    window_start = pd.to_datetime(timestamps[0])

    # Ziel: Fenster = [window_start, window_start + days)
    # window_end ist der letzte gültige Zeitstempel innerhalb dieses Fensters.
    window_end_target = window_start + pd.Timedelta(days=days) - pd.Timedelta(minutes=time_resolution_min)

    idx = int(timestamps.get_indexer([window_end_target], method="nearest")[0])
    idx = max(0, min(idx, len(timestamps) - 1))

    window_end = pd.to_datetime(timestamps[idx])

    # steps_per_day ist bei DST nicht konstant -> None
    return None, window_start, window_end


def get_holiday_dates_from_scenario(scenario: dict, timestamps: pd.DatetimeIndex) -> List[datetime]:
    """
    Liest Feiertage aus dem Szenario und gibt eine eindeutige Liste von Feiertagsdaten zurück.

    Es werden manuell angegebene ISO-Daten aus dem Szenario berücksichtigt und – falls verfügbar –
    zusätzlich Feiertage über das optionale Paket `holidays` für das Land und ggf. die Region erzeugt.
    """
    holidays_configuration = (scenario.get("holidays") or {})
    manual_dates = holidays_configuration.get("dates") or []
    holiday_dates: List[datetime] = []

    for date_text in manual_dates:
        try:
            holiday_dates.append(datetime.fromisoformat(str(date_text)))
        except Exception:
            pass

    try:
        import holidays as python_holidays  # optional dependency

        country_code = str(holidays_configuration.get("country", "DE"))
        subdivision_code = holidays_configuration.get("subdivision", None)
        years = sorted({pd.to_datetime(timestamp).year for timestamp in timestamps})
        holiday_calendar = python_holidays.country_holidays(country_code, subdiv=subdivision_code, years=years)
        for day in holiday_calendar.keys():
            holiday_dates.append(datetime(day.year, day.month, day.day))
    except Exception:
        pass

    unique_by_date = sorted({d.date(): d for d in holiday_dates}.values(), key=lambda x: x.date())
    return unique_by_date


def get_daytype_calendar(start_datetime: datetime, horizon_days: int, holiday_dates: List[datetime]) -> Dict[str, List[Any]]:
    """
    Erstellt für den gesamten Simulationshorizont einen Kalender, der jeden Tag einem Tagtyp zuordnet
    (Arbeitstag, Samstag, Sonntag/Feiertag).

    Die Einordnung erfolgt über die interne Logik von `_get_day_type(...)`, wobei explizite Feiertage
    (holiday_dates) berücksichtigt werden.
    """
    out: Dict[str, List[Any]] = {"working_day": [], "saturday": [], "sunday_holiday": []}
    for day_index in range(int(horizon_days)):
        day_start = start_datetime + timedelta(days=day_index)
        day_type = _get_day_type(day_start, holiday_dates)  # expects function in this module
        out[day_type].append(day_start.date())
    return out


def group_sessions_by_day(sessions_out: List[dict], only_plugged: bool = False) -> Dict[Any, List[dict]]:
    """
    Gruppiert Sessions nach dem Kalendertag ihrer Ankunftszeit.

    Optional können nur Sessions berücksichtigt werden, die tatsächlich „plugged“ sind.
    """
    grouped: Dict[Any, List[dict]] = {}
    for session in sessions_out:
        if only_plugged and session.get("status") != "plugged":
            continue
        arrival_time = session.get("arrival_time")
        day = pd.to_datetime(arrival_time).date()
        grouped.setdefault(day, []).append(session)
    return grouped


def resolve_paths_relative_to_yaml(scenario: dict, scenario_path: str) -> dict:
    """
    Löst relative Dateipfade innerhalb des Szenarios relativ zum Ordner der YAML-Datei auf.

    Dadurch können CSV-Pfade im YAML relativ angegeben werden und werden hier in absolute Pfade
    umgewandelt, sodass das Laden unabhängig vom aktuellen Working Directory funktioniert.
    """
    base_directory = Path(scenario_path).resolve().parent
    scenario_copy = dict(scenario)

    site_section = dict(scenario_copy.get("site", {}))
    vehicles_section = dict(scenario_copy.get("vehicles", {}))

    for key in ["generation_strategy_csv", "market_strategy_csv", "base_load_csv"]:
        if key in site_section and site_section[key]:
            path_value = Path(str(site_section[key]))
            if not path_value.is_absolute():
                site_section[key] = str((base_directory / path_value).resolve())

    if "vehicle_curve_csv" in vehicles_section and vehicles_section["vehicle_curve_csv"]:
        path_value = Path(str(vehicles_section["vehicle_curve_csv"]))
        if not path_value.is_absolute():
            vehicles_section["vehicle_curve_csv"] = str((base_directory / path_value).resolve())

    scenario_copy["site"] = site_section
    scenario_copy["vehicles"] = vehicles_section
    return scenario_copy


def make_timeseries_dataframe(
    timestamps: pd.DatetimeIndex,
    ev_load_kw: np.ndarray,
    scenario: dict,
    debug_rows=None,
    pv_generation_series=None,
    market_price_series=None,
) -> pd.DataFrame:
    """
    Erzeugt das zentrale Timeseries-DataFrame über `build_timeseries_dataframe(...)` und stellt sicher,
    dass zusätzlich eine gut nutzbare Datetime-Spalte `ts` vorhanden ist.

    Diese Funktion ist als bequemer Wrapper gedacht, um Notebook- und Plot-Code zu vereinfachen.
    """
    dataframe = build_timeseries_dataframe(
        timestamps=timestamps,
        ev_load_kw=ev_load_kw,
        scenario=scenario,
        debug_rows=debug_rows,
        generation_series=pv_generation_series,
        market_series=market_price_series,
    )
    if "timestamp" in dataframe.columns and "ts" not in dataframe.columns:
        dataframe["ts"] = pd.to_datetime(dataframe["timestamp"], errors="coerce")
    return dataframe
