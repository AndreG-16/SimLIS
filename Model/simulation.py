from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import Dict, Optional, List, Tuple, Any, Set, Literal
from pathlib import Path
from collections import Counter

# =============================================================================
# Modulüberblick
# =============================================================================
# Dieses Modul simuliert den Lastgang eines Ladepark-Standorts über einen konfigurierten Zeithorizont.
# Es trennt bewusst:
#   (1) Grundmodell / Session-Modellierung (Flotte, Ankünfte, Standzeiten, Energiebedarf),
#   (2) Lademanagement (immediate, market, generation inkl. Fallbacks).
#
# Ziel ist:
#   - Zeitprofil der EV-Ladeleistung (kW)
#   - Session-Details für KPI-Analysen
#   - optional Debug-Zeitreihen für Notebook-Auswertungen

# =============================================================================
# 0) Hilfsfunktionen: Zeit / Einheiten
# =============================================================================

def build_simulation_timestamps(scenario: dict) -> pd.DatetimeIndex:
    """Erzeugt den tz-aware Simulationszeitindex (DST-robust).

    Die Simulation arbeitet konsequent tz-aware (z. B. "Europe/Berlin").
    `simulation_horizon_days` wird als Kalendertage interpretiert, damit DST-Tage
    automatisch 23h bzw. 25h haben.

    Wichtiger Hinweis
    -----------------
    Wenn `start_datetime` tz-naiv ist und in eine *ambige* lokale Zeit fällt
    (DST-Ende, typischerweise 02:xx), wird mit `ambiguous="raise"` ein Fehler
    geworfen. In dem Fall `start_datetime` im YAML auf eine eindeutige Zeit legen
    (z. B. 00:00 oder 03:00) oder explizit einen Offset angeben.

    Raises
    ------
    ValueError
        Wenn `start_datetime` tz-naiv und ambig ist (DST-Ende).
    """
    tz = str(scenario.get("timezone", "Europe/Berlin"))
    step_min = int(scenario["time_resolution_min"])
    horizon_days = int(scenario["simulation_horizon_days"])

    start = pd.to_datetime(scenario["start_datetime"])

    if start.tzinfo is None:
        # Bei einem einzelnen Timestamp ist ambiguous="infer" NICHT erlaubt.
        start = start.tz_localize(tz, ambiguous="raise", nonexistent="shift_forward")
    else:
        start = start.tz_convert(tz)

    end = start + pd.DateOffset(days=horizon_days)  # Kalendertage (DST-sicher)
    return pd.date_range(start=start, end=end, freq=f"{step_min}min", inclusive="left")


def _localize_wall_time_index(
    idx: pd.DatetimeIndex,
    tz: str,
) -> pd.DatetimeIndex:
    """Localize a tz-naive wall-time index to a timezone in a DST-robust way.

    This helper assumes that the CSV timestamps represent local wall time
    (e.g., German local time without timezone info).

    DST handling
    ------------
    - Spring forward (nonexistent local times): shift forward.
      In many datasets the missing times are simply not present in the CSV.
    - Fall back (ambiguous local times): if the CSV contains duplicate timestamps
      (e.g., 02:00 twice), the first occurrence is interpreted as DST (True),
      the second as standard time (False). This yields a unique tz-aware index.

    Parameters
    ----------
    idx:
        Naive DatetimeIndex (local wall time).
    tz:
        IANA timezone name (e.g., "Europe/Berlin").

    Returns
    -------
    pd.DatetimeIndex
        tz-aware localized index.
    """
    idx = pd.DatetimeIndex(idx)

    if idx.tz is not None:
        return idx.tz_convert(tz)

    if idx.has_duplicates:
        dup = idx.duplicated(keep=False)

        # Count occurrences per timestamp: 0 for first, 1 for second, ...
        occ = pd.Series(np.arange(len(idx))).groupby(idx).cumcount().to_numpy()

        # For ambiguous duplicated wall times:
        # first occurrence -> DST=True, second -> DST=False
        ambiguous = np.where(dup, occ == 0, False)

        return idx.tz_localize(
            tz,
            ambiguous=ambiguous,
            nonexistent="shift_forward",
        )

    return idx.tz_localize(
        tz,
        ambiguous="infer",
        nonexistent="shift_forward",
    )


# =============================================================================
# 1) Daten-Reader: Gebäudeprofil / Marktpreise / Ladekurven / PV-Generation
# =============================================================================
@dataclass
class VehicleChargingCurve:
    """
    Container für eine fahrzeugspezifische Ladekennlinie.

    Attributes
    ----------
    vehicle_name:
        Anzeigename des Fahrzeugs (z.B. "Audi Q4 e-tron").
    manufacturer:
        Hersteller.
    model:
        Modellbezeichnung.
    vehicle_class:
        Fahrzeugklasse (z.B. "PKW", "Transporter").
    battery_capacity_kwh:
        Maximale Batteriekapazität in kWh.
    state_of_charge_fraction:
        SoC-Stützstellen als Anteil [0..1].
    power_kw:
        Maximale Ladeleistung (Batterieseite) in kW für die jeweiligen SoC-Stützstellen.
    """
    vehicle_name: str
    manufacturer: str
    model: str
    vehicle_class: str
    battery_capacity_kwh: float
    state_of_charge_fraction: np.ndarray
    power_kw: np.ndarray


def read_scenario_from_yaml(scenario_path: str) -> Dict[str, Any]:
    """
    Liest eine YAML-Szenario-Datei ein, validiert Pflichtfelder und ergänzt Defaults.

    Erwartete Struktur (Auszug)
    ---------------------------
    Top-Level Pflichtfelder:
    - time_resolution_min
    - simulation_horizon_days
    - start_datetime
    - site
    - vehicles

    Pflichtfelder in ``site``:
    - number_chargers
    - rated_power_kw
    - grid_limit_p_avb_kw
    - expected_sessions_per_charger_per_day
    - pv_system_size_kwp
    - base_load_annual_kwh

    Pflichtfelder in ``vehicles``:
    - vehicle_curve_csv

    Parameters
    ----------
    scenario_path:
        Pfad zur YAML-Datei.

    Returns
    -------
    dict
        Szenario als Dictionary.

    Raises
    ------
    ValueError
        Wenn YAML ungültig ist oder Pflichtfelder fehlen.
    """
    with open(scenario_path, "r", encoding="utf-8") as file_handle:
        scenario = yaml.safe_load(file_handle)

    if not isinstance(scenario, dict):
        raise ValueError("YAML konnte nicht als Dictionary gelesen werden.")

    required_top_level = [
        "time_resolution_min",
        "simulation_horizon_days",
        "start_datetime",
        "charging_strategy",
        "localload_pv_market_csv",
        "site",
        "vehicles",
    ]
    missing_top_level = [key for key in required_top_level if key not in scenario]
    if missing_top_level:
        raise ValueError(f"Pflichtfelder fehlen in YAML (top-level): {missing_top_level}")

    site_configuration = scenario.get("site")
    if not isinstance(site_configuration, dict):
        raise ValueError("YAML.site fehlt oder ist kein Dictionary.")

    required_site = [
        "number_chargers",
        "rated_power_kw",
        "grid_limit_p_avb_kw",
        "expected_sessions_per_charger_per_day",
        "pv_system_size_kwp",
        "base_load_annual_kwh",
    ]

    missing_site = [key for key in required_site if key not in site_configuration]
    if missing_site:
        raise ValueError(f"Pflichtfelder fehlen in YAML.site: {missing_site}")

    vehicles_configuration = scenario.get("vehicles")
    if not isinstance(vehicles_configuration, dict):
        raise ValueError("YAML.vehicles fehlt oder ist kein Dictionary.")

    required_vehicles = ["vehicle_curve_csv"]
    missing_vehicles = [key for key in required_vehicles if key not in vehicles_configuration]
    if missing_vehicles:
        raise ValueError(f"Pflichtfelder fehlen in YAML.vehicles: {missing_vehicles}")

    scenario["site"] = site_configuration
    scenario["vehicles"] = vehicles_configuration

    return scenario


def resolve_paths_relative_to_yaml(scenario: dict, scenario_path: str) -> dict:
    """
    Löst relative Dateipfade im Szenario relativ zum Speicherort der YAML-Datei auf.

    Diese Funktion sorgt dafür, dass Pfade aus der YAML (z. B. CSV-Dateien) unabhängig vom
    aktuellen Working Directory korrekt gefunden werden. Relative Pfade werden relativ zum
    Ordner der YAML-Datei in absolute Pfade umgewandelt. Absolute Pfade bleiben unverändert.

    Aktuell werden folgende Szenario-Felder aufgelöst:
    - scenario["localload_pv_market_csv"]
    - scenario["vehicles"]["vehicle_curve_csv"]

    Parameters
    ----------
    scenario:
        Szenario-Dictionary, wie es aus der YAML eingelesen wurde. Das Dictionary wird
        in-place aktualisiert.
    scenario_path:
        Pfad zur YAML-Datei, die als Referenz für relative Pfade dient.

    Returns
    -------
    dict
        Das aktualisierte Szenario-Dictionary mit absoluten Pfaden.

    Raises
    ------
    KeyError
        Wenn eines der erwarteten Pfadfelder im Szenario fehlt.
    ValueError
        Wenn ein Pfad leer ist oder nur aus Whitespaces besteht.
    """
    base_directory = Path(scenario_path).resolve().parent

    def to_absolute(path_value: Any) -> str:
        path_text = str(path_value).strip()
        if not path_text:
            raise ValueError("CSV-Pfad darf nicht leer sein.")
        path_object = Path(path_text).expanduser()
        if path_object.is_absolute():
            return str(path_object)
        return str((base_directory / path_object).resolve())

    scenario["localload_pv_market_csv"] = to_absolute(scenario["localload_pv_market_csv"])
    scenario["vehicles"]["vehicle_curve_csv"] = to_absolute(scenario["vehicles"]["vehicle_curve_csv"])
    return scenario


def read_local_load_pv_market_from_csv(
    csv_path: str,
    timestamps: pd.DatetimeIndex,
    base_load_annual_kwh: float,
    pv_system_size_kwp: float,
    profiles_are_normalized: bool = True,
    input_time_resolution_min: int = 15,
    datetime_format: str = "%d.%m.%y %H:%M",
    separator: str = ";",
    decimal: str = ",",
    timezone: Optional[str] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Liest Grundlast-, PV- und Marktpreis-Zeitreihen aus CSV und richtet sie am Simulationsraster aus.

    Annahmen (tz-aware Simulation)
    ------------------------------
    - `timestamps` ist *tz-aware* (z. B. "Europe/Berlin") und definiert das Simulationsraster.
    - Die CSV enthält in Spalte 0 lokale Zeitstempel als *Wandzeit* (tz-naiv).
      Dadurch existieren im Frühjahr (Sommerzeitbeginn) bestimmte Zeiten nicht,
      und im Herbst (Winterzeitbeginn) kommen Zeiten (02:xx) doppelt vor.
    - Die Lokalisierung der CSV erfolgt DST-robust über `_localize_wall_time_index()`:
      - Nicht-existente Zeiten: werden nach vorn geschoben (`nonexistent="shift_forward"`).
      - Doppelte Zeiten (Herbst): werden disambiguiert (erste Vorkommen DST, zweite Standardzeit).

    CSV-Format (mindestens 4 Spalten)
    ---------------------------------
    Spalte 0: Datum/Zeit als String (Format `datetime_format`)
    Spalte 1: Grundlast-Profil
    Spalte 2: PV-Profil
    Spalte 3: Marktpreis [€/MWh]

    Interpretation und Skalierung
    -----------------------------
    Wenn `profiles_are_normalized=True`:
    - Grundlast (Spalte 1) ist ein normiertes Profil (gewichtete Anteile) und wird so skaliert,
      dass die Summe über das Jahr `base_load_annual_kwh` entspricht.
    - PV (Spalte 2) ist ein Kapazitätsfaktor (0..1) und wird mit `pv_system_size_kwp`
      in Leistung [kW] umgerechnet und dann auf Energie [kWh] je CSV-Zeitschritt.

    Rückgabe
    --------
    - Grundlast und PV als *kWh pro Simulationsschritt*.
    - Marktpreis als *€/MWh* je Simulationsschritt.

    Resampling / Alignment
    ----------------------
    - Wenn Simulationstakt == CSV-Takt: reindex (ohne Interpolation).
    - Wenn Simulation feiner als CSV (Upsampling): Forward-Fill der Leistung.
    - Wenn Simulation gröber als CSV (Downsampling): Resample der Leistung per Mittelwert.

    Parameter
    ---------
    csv_path:
        Pfad zur CSV-Datei.
    timestamps:
        Simulations-Zeitindex (tz-aware erforderlich).
    base_load_annual_kwh:
        Jährlicher Energieverbrauch des Standorts [kWh] (für Skalierung bei normalisiertem Profil).
    pv_system_size_kwp:
        PV-Anlagengröße [kWp] (für Skalierung bei normalisiertem Profil).
    profiles_are_normalized:
        True: Spalten 1/2 sind normierte Profile; False: bereits Energiemengen je CSV-Schritt.
    input_time_resolution_min:
        Zeitauflösung der CSV in Minuten.
    datetime_format:
        Parsformat für Spalte 0.
    separator:
        Trennzeichen der CSV.
    decimal:
        Dezimaltrennzeichen der CSV.
    timezone:
        Optionales Override für die Lokalisierung der CSV-Wandzeit (Standard: `timestamps.tz`).

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        (base_load_kwh_per_step, pv_generation_kwh_per_step, market_price_eur_per_mwh)

    Raises
    ------
    ValueError
        - wenn `timestamps` tz-naiv ist (harte Prüfung)
        - wenn CSV < 4 Spalten hat
        - wenn Numerik-Parsen fehlschlägt
    """
    ts_index = pd.DatetimeIndex(timestamps)

    # Harte Prüfung: konsequent tz-aware Simulation.
    if ts_index.tz is None:
        raise ValueError(
            "Die Simulation ist nicht tz-aware: `timestamps.tz` ist None. "
            "Bitte den Simulations-Zeitindex z. B. mit 'Europe/Berlin' lokalisieren."
        )

    if len(ts_index) < 2:
        raise ValueError("`timestamps` muss mindestens zwei Einträge enthalten.")

    # CSV einlesen und Grundstruktur prüfen.
    df = pd.read_csv(csv_path, sep=separator, decimal=decimal)
    if df.shape[1] < 4:
        raise ValueError(
            "CSV hat zu wenige Spalten. Erwartet: datetime | base load | PV | market price."
        )

    dt_col = df.columns[0]
    base_col = df.columns[1]
    pv_col = df.columns[2]
    price_col = df.columns[3]

    # Datums-/Zeitspalte parsen und als Index setzen.
    # Hinweis: CSV-Zeit ist tz-naive Wandzeit -> wird danach lokalisiert.
    parsed_dt = pd.to_datetime(
        df[dt_col].astype(str).str.strip(),
        format=datetime_format,
        errors="raise",
    )

    df = df.copy()
    df[dt_col] = parsed_dt
    df = df.sort_values(dt_col).set_index(dt_col)

    # CSV-Index in Simulations-TZ lokalisieren (DST-robust).
    target_tz = str(timezone) if timezone else str(ts_index.tz)
    df.index = _localize_wall_time_index(df.index, target_tz)

    # Numerische Spalten parsen (und Profile auf >= 0 clampen).
    df[base_col] = pd.to_numeric(df[base_col], errors="raise").astype(float).clip(lower=0.0)
    df[pv_col] = pd.to_numeric(df[pv_col], errors="raise").astype(float).clip(lower=0.0)
    df[price_col] = pd.to_numeric(df[price_col], errors="raise").astype(float)

    df = df[[base_col, pv_col, price_col]].sort_index()

    # Zeitschrittgrößen bestimmen.
    sim_step_min = int(round((ts_index[1] - ts_index[0]).total_seconds() / 60.0))
    sim_step_hours = float(sim_step_min) / 60.0
    input_step_hours = float(input_time_resolution_min) / 60.0

    # Profile interpretieren / skalieren (auf Input-Gitter).
    base_profile = df[base_col]
    pv_profile = df[pv_col]
    price_series = df[price_col]

    if profiles_are_normalized:
        # Grundlast: normierte Gewichte -> skaliere auf Jahresenergie.
        total_weight = float(base_profile.sum())
        if base_load_annual_kwh <= 0.0 or total_weight <= 0.0:
            base_kwh_input = base_profile * 0.0
        else:
            base_kwh_input = (base_profile / total_weight) * float(base_load_annual_kwh)

        # PV: Kapazitätsfaktor -> kW -> kWh je Input-Schritt.
        pv_power_kw = pv_profile * float(max(pv_system_size_kwp, 0.0))
        pv_kwh_input = pv_power_kw * input_step_hours
    else:
        # Profile sind bereits Energiemengen je CSV-Schritt.
        base_kwh_input = base_profile
        pv_kwh_input = pv_profile

    # Energie je Input-Schritt -> Leistung [kW] (für korrektes Resampling).
    base_kw = base_kwh_input / input_step_hours
    pv_kw = pv_kwh_input / input_step_hours

    # Leistung auf Simulationsraster ausrichten (resample/reindex).
    if sim_step_min == int(input_time_resolution_min):
        # Gleiche Auflösung: direkt auf Raster legen.
        base_kw_aligned = base_kw.reindex(ts_index)
        pv_kw_aligned = pv_kw.reindex(ts_index)
        price_aligned = price_series.reindex(ts_index, method="ffill")

    elif sim_step_min < int(input_time_resolution_min):
        # Simulation feiner als CSV: stückweise konstante Leistung bis zur nächsten Messung.
        base_kw_aligned = base_kw.reindex(ts_index, method="ffill")
        pv_kw_aligned = pv_kw.reindex(ts_index, method="ffill")
        price_aligned = price_series.reindex(ts_index, method="ffill")

    else:
        # Simulation gröber als CSV: mittlere Leistung über das größere Intervall.
        rule = f"{sim_step_min}min"
        base_kw_aligned = base_kw.resample(rule).mean().reindex(ts_index)
        pv_kw_aligned = pv_kw.resample(rule).mean().reindex(ts_index)
        price_aligned = price_series.resample(rule).mean().reindex(ts_index, method="ffill")

    # Fehlende Werte in Leistung als 0 interpretieren (z. B. außerhalb CSV-Bereich).
    base_kw_aligned = base_kw_aligned.fillna(0.0)
    pv_kw_aligned = pv_kw_aligned.fillna(0.0)

    # Leistung [kW] -> Energie [kWh] je Simulationsschritt.
    base_kwh_step = base_kw_aligned * sim_step_hours
    pv_kwh_step = pv_kw_aligned * sim_step_hours

    return (
        base_kwh_step.astype(float),
        pv_kwh_step.astype(float),
        price_aligned.astype(float),
    )


def read_vehicle_load_profiles_from_csv(
    vehicle_curve_csv_path: str,
) -> Dict[str, VehicleChargingCurve]:
    """
    Reads vehicle charging curves from a CSV in a wide format (one vehicle per column).

    The CSV is expected to have fixed labels in column 0, rows 0..3:
    - "Hersteller"
    - "Modell"
    - "Fahrzeugklasse"
    - "max. Kapazität"

    Notes
    -----
    Some CSV exports include a UTF-8 BOM marker at the beginning of the file.
    This can turn the first label into "\\ufeffHersteller". We handle this by
    reading with "utf-8-sig" and stripping BOM characters defensively.

    Parameters
    ----------
    vehicle_curve_csv_path:
        Path to the CSV file.

    Returns
    -------
    Dict[str, VehicleChargingCurve]
        Mapping from unique vehicle name to its charging curve data.

    Raises
    ------
    ValueError
        If the CSV does not match the expected format or contains invalid data.
    """
    vehicle_curve_table = pd.read_csv(
        vehicle_curve_csv_path,
        sep=None,
        engine="python",
        header=None,
        dtype=str,
        encoding="utf-8-sig",  # <-- NEW: removes UTF-8 BOM if present
    )

    if vehicle_curve_table.shape[0] < 6 or vehicle_curve_table.shape[1] < 2:
        raise ValueError(
            "Vehicleloadprofile-CSV hat nicht das erwartete Format "
            "(benötigt mindestens 6 Zeilen und 2 Spalten)."
        )

    expected_labels = ["Hersteller", "Modell", "Fahrzeugklasse", "max. Kapazität"]

    def _clean_label(x: object) -> str:
        # <-- NEW: defensive BOM stripping (in case encoding wasn't applied)
        return str(x).strip().lstrip("\ufeff")

    found_labels = [_clean_label(vehicle_curve_table.iat[row_index, 0]) for row_index in range(4)]
    if found_labels != expected_labels:
        raise ValueError(
            "Vehicleloadprofile-CSV entspricht nicht dem erwarteten Format.\n"
            f"Erwartet in Spalte 0, Zeilen 0..3: {expected_labels}\n"
            f"Gefunden: {found_labels}"
        )

    manufacturer_by_vehicle = vehicle_curve_table.iloc[0, 1:].astype(str).str.strip().to_list()
    model_by_vehicle = vehicle_curve_table.iloc[1, 1:].astype(str).str.strip().to_list()
    vehicle_class_by_vehicle = vehicle_curve_table.iloc[2, 1:].astype(str).str.strip().to_list()

    battery_capacity_kwh_by_vehicle = pd.to_numeric(
        vehicle_curve_table.iloc[3, 1:].astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    ).to_numpy(dtype=float)

    soc_raw = pd.to_numeric(
        vehicle_curve_table.iloc[4:, 0].astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    ).to_numpy(dtype=float)

    valid_soc_mask = np.isfinite(soc_raw)
    if np.count_nonzero(valid_soc_mask) < 2:
        raise ValueError("Vehicleloadprofile-CSV: SoC-Spalte enthält zu wenige gültige Zahlenwerte.")

    soc_values = soc_raw[valid_soc_mask]
    if not np.all(np.diff(soc_values) > 0):
        raise ValueError(
            "Vehicleloadprofile-CSV: SoC-Stützstellen müssen streng monoton steigend sein "
            "(z. B. 0, 0.5, 1, ...)."
        )

    # SoC: wenn max > 1 -> Prozent -> in Anteil umrechnen
    if float(np.nanmax(soc_values)) > 1.0:
        state_of_charge_fraction = np.clip(soc_values / 100.0, 0.0, 1.0)
    else:
        state_of_charge_fraction = np.clip(soc_values, 0.0, 1.0)

    curves_by_vehicle_name: Dict[str, VehicleChargingCurve] = {}

    number_vehicle_columns = int(vehicle_curve_table.shape[1] - 1)
    for vehicle_column_offset in range(number_vehicle_columns):
        column_index = 1 + vehicle_column_offset

        manufacturer = str(manufacturer_by_vehicle[vehicle_column_offset]).strip()
        model = str(model_by_vehicle[vehicle_column_offset]).strip()
        vehicle_class = str(vehicle_class_by_vehicle[vehicle_column_offset]).strip()

        battery_capacity_kwh = float(battery_capacity_kwh_by_vehicle[vehicle_column_offset])
        if not np.isfinite(battery_capacity_kwh) or battery_capacity_kwh <= 0.0:
            raise ValueError(
                "Vehicleloadprofile-CSV: Ungültige Batteriekapazität gefunden "
                f"(Spalte {column_index}: Hersteller='{manufacturer}', Modell='{model}')."
            )

        power_raw = pd.to_numeric(
            vehicle_curve_table.iloc[4:, column_index].astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        ).to_numpy(dtype=float)

        power_values_kw = power_raw[valid_soc_mask]
        if np.count_nonzero(np.isfinite(power_values_kw)) < 2:
            raise ValueError(
                "Vehicleloadprofile-CSV: Zu wenige gültige Leistungswerte "
                f"(Spalte {column_index}: Hersteller='{manufacturer}', Modell='{model}')."
            )

        power_values_kw = np.maximum(np.nan_to_num(power_values_kw, nan=0.0), 0.0)

        vehicle_name = f"{manufacturer} {model}".strip()
        if vehicle_name == "" or vehicle_name.lower() == "nan nan":
            raise ValueError(f"Vehicleloadprofile-CSV: Ungültiger Fahrzeugname in Spalte {column_index}.")

        unique_vehicle_name = vehicle_name
        suffix_counter = 2
        while unique_vehicle_name in curves_by_vehicle_name:
            unique_vehicle_name = f"{vehicle_name} ({suffix_counter})"
            suffix_counter += 1

        curves_by_vehicle_name[unique_vehicle_name] = VehicleChargingCurve(
            vehicle_name=unique_vehicle_name,
            manufacturer=manufacturer,
            model=model,
            vehicle_class=vehicle_class,  # z.B. "PKW" oder "Transporter"
            battery_capacity_kwh=battery_capacity_kwh,
            state_of_charge_fraction=state_of_charge_fraction.copy(),
            power_kw=power_values_kw.copy(),
        )

    if not curves_by_vehicle_name:
        raise ValueError("Vehicleloadprofile-CSV: Es konnten keine Fahrzeugkurven extrahiert werden.")

    return curves_by_vehicle_name


# =============================================================================
# 2) Sampling: Sessions (Ankunft / Standzeit / SoC / Fleet-Mix)
# =============================================================================

@dataclass
class SampledSession:
    """
    Repräsentiert eine zufällig erzeugte Ladesession innerhalb der Simulation.

    Attributes
    ----------
    session_id:
        Eindeutige ID der Session.
    arrival_time:
        Ankunftszeitpunkt (auf Simulationsraster gerastert).
    departure_time:
        Abfahrtszeitpunkt (auf Simulationsraster gerastert, Ende exklusiv).
    arrival_step:
        absolute Schritte im gesamten Horizont
    departure_step:
        absolute Schritte im gesamten Horizont
    duration_minutes:
        Standzeit in Minuten (aus gerasterten Zeiten berechnet).
    state_of_charge_at_arrival:
        SoC bei Ankunft als Anteil [0..1].
    vehicle_name:
        Fahrzeugname (Key aus vehicle_curves_by_name).
    vehicle_class:
        Fahrzeugklasse (z. B. "PKW", "Transporter").
    day_type:
        Tagtyp ("working_day", "saturday", "sunday_holiday").
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


def get_day_type(simulation_day_start: datetime, holiday_dates: Set[date]) -> str:
    """
    Klassifiziert einen Simulationstag als "working_day", "saturday" oder "sunday_holiday".

    Parameters
    ----------
    simulation_day_start:
        Startzeitpunkt des Tages (Datumsteil ist relevant).
    holiday_dates:
        Menge von Feiertagen als datetime.date.

    Returns
    -------
    str
        "working_day", "saturday" oder "sunday_holiday".
    """
    if simulation_day_start.date() in holiday_dates:
        return "sunday_holiday"
    weekday_index = int(simulation_day_start.weekday())  # 0=Mo .. 6=So
    if weekday_index == 5:
        return "saturday"
    if weekday_index == 6:
        return "sunday_holiday"
    return "working_day"


def define_sample_from_distribution(spec: Any, random_generator: np.random.Generator) -> float:
    """
    Zieht genau einen Sample-Wert aus einer Verteilungsspezifikation.

    Unterstützte Eingaben
    ---------------------
    - Konstante: 3.5
    - Uniform-Range: [min, max]
    - Einzelkomponente: {"distribution": "normal|beta|lognormal", ...}
    - Mixture: {"type": "mixture", "components": [...]} oder direkt [...]-Liste

    Returns
    -------
    float
        Gesampelter Wert.

    Raises
    ------
    ValueError
        Wenn die Spezifikation ungültig ist.
    """
    if isinstance(spec, (int, float, np.number)):
        return float(spec)

    if isinstance(spec, list):
        if len(spec) == 2 and all(isinstance(x, (int, float, np.number)) for x in spec):
            minimum_value, maximum_value = float(spec[0]), float(spec[1])
            if maximum_value < minimum_value:
                raise ValueError("Uniform-Range erwartet [min, max] mit max >= min.")
            return float(random_generator.uniform(minimum_value, maximum_value))
        spec = {"type": "mixture", "components": spec}

    if not isinstance(spec, dict):
        raise ValueError(f"Distribution spec muss Zahl, list oder dict sein, bekommen: {type(spec)}")

    if str(spec.get("type", "")).strip().lower() == "mixture":
        components = spec.get("components")
        if not isinstance(components, list) or len(components) == 0 or not all(isinstance(c, dict) for c in components):
            raise ValueError("mixture benötigt eine nicht-leere Liste 'components' aus dicts.")

        weights = np.array([float(c.get("weight", 1.0)) for c in components], dtype=float)
        if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
            raise ValueError("mixture.weights müssen >= 0 sein und eine positive Summe haben.")

        chosen_index = int(random_generator.choice(len(components), p=weights / float(weights.sum())))
        return define_sample_from_distribution(components[chosen_index], random_generator)

    distribution_name = str(spec.get("distribution", "")).strip().lower()
    if not distribution_name:
        raise ValueError("Einzelkomponente benötigt key 'distribution'.")

    def require(key: str) -> Any:
        if key not in spec:
            raise ValueError(f"Distribution '{distribution_name}' benötigt Parameter '{key}'.")
        return spec[key]

    if distribution_name == "normal":
        mean_value = define_sample_from_distribution(require("mu"), random_generator)
        standard_deviation = define_sample_from_distribution(require("sigma"), random_generator)
        if standard_deviation <= 0.0:
            raise ValueError("Normalverteilung benötigt sigma > 0.")
        return float(random_generator.normal(loc=float(mean_value), scale=float(standard_deviation)))

    if distribution_name == "beta":
        alpha_value = define_sample_from_distribution(require("alpha"), random_generator)
        beta_value = define_sample_from_distribution(require("beta"), random_generator)
        if alpha_value <= 0.0 or beta_value <= 0.0:
            raise ValueError("Betaverteilung benötigt alpha > 0 und beta > 0.")
        return float(random_generator.beta(a=float(alpha_value), b=float(beta_value)))

    if distribution_name == "lognormal":
        mean_value = define_sample_from_distribution(require("mu"), random_generator)
        standard_deviation = define_sample_from_distribution(require("sigma"), random_generator)
        if standard_deviation <= 0.0:
            raise ValueError("Lognormalverteilung benötigt sigma > 0.")
        return float(random_generator.lognormal(mean=float(mean_value), sigma=float(standard_deviation)))

    raise ValueError(f"Unbekannte distribution in spec: '{distribution_name}'")


def sample_vehicle_for_session(
    vehicle_curves_by_name: Dict[str, "VehicleChargingCurve"],
    fleet_mix: Dict[str, float],
    random_generator: np.random.Generator,
    vehicle_type: Optional[Any] = None,
) -> Tuple[str, str]:
    """
    Wählt ein Fahrzeug für eine Ladesession (Klasse nach fleet_mix, Fahrzeug gleichverteilt).

    Parameters
    ----------
    vehicle_curves_by_name:
        Mapping {vehicle_name: VehicleChargingCurve}.
    fleet_mix:
        Klassen-Gewichte, z. B. {"PKW": 0.98, "Transporter": 0.02}. Kann leer sein.
    random_generator:
        Numpy RNG.
    vehicle_type:
        Optionaler Override: String oder Liste[String], max. 5 Fahrzeugnamen.

    Returns
    -------
    Tuple[str, str]
        (vehicle_name, vehicle_class)

    Raises
    ------
    ValueError
        Wenn keine Fahrzeuge verfügbar sind oder Konfiguration ungültig ist.
    """
    if len(vehicle_curves_by_name) == 0:
        raise ValueError("vehicle_curves_by_name ist leer.")

    selected_vehicle_names: Optional[List[str]] = None
    if vehicle_type is not None:
        if isinstance(vehicle_type, str):
            selected_vehicle_names = [vehicle_type.strip()]
        elif isinstance(vehicle_type, list):
            selected_vehicle_names = [str(x).strip() for x in vehicle_type if str(x).strip()]
        else:
            raise ValueError("vehicles.vehicle_type muss String oder Liste von Strings sein.")

        selected_vehicle_names = list(dict.fromkeys(selected_vehicle_names))
        if len(selected_vehicle_names) == 0:
            selected_vehicle_names = None
        elif len(selected_vehicle_names) > 5:
            raise ValueError("vehicles.vehicle_type darf maximal 5 Fahrzeuge enthalten.")

    if selected_vehicle_names is None:
        candidate_curves = vehicle_curves_by_name
    else:
        unknown_names = [name for name in selected_vehicle_names if name not in vehicle_curves_by_name]
        if unknown_names:
            raise ValueError(f"vehicles.vehicle_type enthält unbekannte Fahrzeugnamen: {unknown_names}")
        candidate_curves = {name: vehicle_curves_by_name[name] for name in selected_vehicle_names}

    available_classes = sorted({curve.vehicle_class for curve in candidate_curves.values()})
    if len(available_classes) == 0:
        raise ValueError("Keine Fahrzeugklassen in candidate_curves gefunden.")

    if fleet_mix:
        class_names = [class_name for class_name in fleet_mix.keys() if class_name in available_classes]
        if len(class_names) == 0:
            raise ValueError("fleet_mix passt zu keiner verfügbaren Fahrzeugklasse.")
        class_weights = np.array([float(fleet_mix[name]) for name in class_names], dtype=float)
        if np.any(class_weights < 0.0) or float(class_weights.sum()) <= 0.0:
            raise ValueError("fleet_mix Gewichte müssen >= 0 sein und eine positive Summe haben.")
        chosen_class = str(random_generator.choice(class_names, p=class_weights / float(class_weights.sum())))
    else:
        chosen_class = str(random_generator.choice(available_classes))

    vehicle_names_in_class = [
        curve.vehicle_name for curve in candidate_curves.values() if str(curve.vehicle_class) == chosen_class
    ]
    if len(vehicle_names_in_class) == 0:
        raise ValueError(f"Keine Fahrzeuge in Klasse '{chosen_class}' verfügbar.")
    chosen_vehicle_name = str(random_generator.choice(vehicle_names_in_class))

    return chosen_vehicle_name, chosen_class


def choose_parameters_for_simulation_step(
    scenario: dict,
    day_type: str,
    random_generator: np.random.Generator,
    vehicle_curves_by_name: Dict[str, "VehicleChargingCurve"],
) -> Tuple[float, float, float, str, str]:
    """
    Zieht die Parameter einer einzelnen Session aus den Szenario-Verteilungen.

    Returns
    -------
    Tuple[float, float, float, str, str]
        (arrival_hours, parking_duration_minutes, soc_at_arrival, vehicle_name, vehicle_class)
    """
    site_configuration = scenario["site"]
    vehicles_section = scenario["vehicles"]

    arrival_distribution = scenario.get("arrival_time_distribution", {}) or {}
    components_per_weekday = arrival_distribution.get("components_per_weekday", {}) or {}
    arrival_spec = components_per_weekday.get(day_type)
    if not isinstance(arrival_spec, (list, dict)):
        raise ValueError(f"arrival_time_distribution für day_type='{day_type}' fehlt oder ist ungültig.")

    parking_distribution = scenario.get("parking_duration_distribution", {}) or {}
    min_duration_minutes = float(parking_distribution.get("min_duration_minutes", 0.0))
    max_duration_minutes = float(parking_distribution.get("max_duration_minutes", 24.0 * 60.0))
    if max_duration_minutes < min_duration_minutes:
        raise ValueError("parking_duration_distribution: max_duration_minutes < min_duration_minutes.")
    parking_spec = parking_distribution.get("components", parking_distribution.get("spec"))
    if not isinstance(parking_spec, (list, dict)):
        raise ValueError("parking_duration_distribution benötigt 'components' (list) oder 'spec' (dict).")

    soc_distribution = scenario.get("soc_at_arrival_distribution", {}) or {}
    max_soc = float(soc_distribution.get("max_soc", 1.0))
    if not (0.0 <= max_soc <= 1.0):
        raise ValueError("soc_at_arrival_distribution.max_soc muss in [0, 1] liegen.")
    soc_spec = soc_distribution.get("components", soc_distribution.get("spec"))
    if not isinstance(soc_spec, (list, dict)):
        raise ValueError("soc_at_arrival_distribution benötigt 'components' (list) oder 'spec' (dict).")

    arrival_hours = float(define_sample_from_distribution(arrival_spec, random_generator))
    parking_duration_minutes = float(define_sample_from_distribution(parking_spec, random_generator))
    soc_at_arrival = float(define_sample_from_distribution(soc_spec, random_generator))

    parking_duration_minutes = float(np.clip(parking_duration_minutes, min_duration_minutes, max_duration_minutes))
    soc_at_arrival = float(np.clip(soc_at_arrival, 0.0, max_soc))

    fleet_mix = vehicles_section.get("fleet_mix", {}) or {}
    vehicle_name, vehicle_class = sample_vehicle_for_session(
        vehicle_curves_by_name=vehicle_curves_by_name,
        fleet_mix=fleet_mix,
        random_generator=random_generator,
        vehicle_type=vehicles_section.get("vehicle_type"),
    )

    return arrival_hours, parking_duration_minutes, soc_at_arrival, vehicle_name, vehicle_class


def sample_sessions_for_simulation_day(
    scenario: dict,
    simulation_day_start: datetime,
    timestamps: pd.DatetimeIndex,
    holiday_dates: Set[date],
    vehicle_curves_by_name: Dict[str, "VehicleChargingCurve"],
    random_generator: np.random.Generator,
    day_index: int,
) -> List["SampledSession"]:
    """
    Erzeugt zufällige Ladesessions für genau einen Kalendertag der Simulation.

    Voraussetzungen (konsequente tz-aware Simulation)
    -------------------------------------------------
    - `timestamps` ist tz-aware (z. B. Europe/Berlin).
    - `simulation_day_start` ist ebenfalls tz-aware und in derselben Zeitzone wie `timestamps`.
      (Diese Funktion konvertiert NICHT und lokalisiert NICHT.)

    Hinweise
    --------
    - `arrival_step` und `departure_step` sind *absolute* Indizes im Simulationsraster.
    - `departure_step` ist Ende-exklusiv.
    - Ankunft wird auf den nächstgelegenen Simulationszeitpunkt gerastert (nearest).
    - Abfahrt wird als `arrival_wall + parkdauer` berechnet und dann per **ceil** gerastert
      (d. h. der erste Rasterpunkt >= Abfahrt-Zeit; dadurch wird die Parkdauer nicht verkürzt).
    - Wenn `allow_cross_day_charging=False`, wird die Session am Tagesende abgeschnitten.

    Returns
    -------
    list[SampledSession]
        Liste gesampelter Sessions (nach arrival_time sortiert).
    """
    ts_index = pd.DatetimeIndex(timestamps)

    # Du wolltest den tz-aware Check außerhalb machen. Optionaler Guard (sehr empfehlenswert):
    # if ts_index.tz is None:
    #     raise ValueError("timestamps muss tz-aware sein.")

    time_resolution_min = int(scenario["time_resolution_min"])
    step_minutes = float(time_resolution_min)

    # -------------------------------------------------------------------------
    # Tagesfenster im Simulationsraster bestimmen (TZ-sicher über normalize()).
    # Wichtig: simulation_day_start muss bereits in Simulations-TZ sein.
    # -------------------------------------------------------------------------
    day_start_ts = pd.Timestamp(simulation_day_start)
    day_key = day_start_ts.normalize()

    day_mask = ts_index.normalize() == day_key
    if not np.any(day_mask):
        return []

    day_timestamps = ts_index[day_mask]
    steps_per_day = int(len(day_timestamps))
    if steps_per_day <= 0:
        return []

    day_start_abs_step = int(ts_index.get_loc(day_timestamps[0]))
    day_end_excl = day_start_abs_step + steps_per_day  # Ende-exklusiv (erster Step des Folgetags)

    # -------------------------------------------------------------------------
    # Anzahl Sessions via Erwartungswert * Tagesgewichtung.
    # -------------------------------------------------------------------------
    site_configuration = scenario["site"]
    number_chargers = int(site_configuration["number_chargers"])

    expected_sessions_spec = site_configuration["expected_sessions_per_charger_per_day"]
    expected_sessions_per_charger = float(
        define_sample_from_distribution(expected_sessions_spec, random_generator)
    )
    if expected_sessions_per_charger < 0.0:
        raise ValueError("expected_sessions_per_charger_per_day muss >= 0 sein.")

    day_type = get_day_type(simulation_day_start, holiday_dates)

    arrival_distribution = scenario.get("arrival_time_distribution", {}) or {}
    weekday_weight_table = arrival_distribution.get("weekday_weight", {}) or {}
    weekday_weight_spec = weekday_weight_table.get(day_type, 1.0)
    weekday_weight = float(define_sample_from_distribution(weekday_weight_spec, random_generator))
    if weekday_weight < 0.0:
        raise ValueError("weekday_weight muss >= 0 sein.")

    expected_total_sessions = expected_sessions_per_charger * float(number_chargers) * float(weekday_weight)
    number_sessions = int(random_generator.poisson(lam=float(max(expected_total_sessions, 0.0))))

    allow_cross_day_charging = bool(site_configuration.get("allow_cross_day_charging", False))

    sampled_sessions: List[SampledSession] = []

    # -------------------------------------------------------------------------
    # Hilfsfunktion: "ceil" Rasterung auf ts_index (erster Index mit ts >= target)
    # -------------------------------------------------------------------------
    def ceil_index(target: pd.Timestamp) -> int:
        # searchsorted funktioniert nur korrekt, wenn ts_index sortiert ist (sollte es sein).
        idx = int(ts_index.searchsorted(target, side="left"))
        return idx

    for session_number in range(number_sessions):
        arrival_hours, parking_duration_minutes, soc_at_arrival, vehicle_name, vehicle_class = (
            choose_parameters_for_simulation_step(
                scenario=scenario,
                day_type=day_type,
                random_generator=random_generator,
                vehicle_curves_by_name=vehicle_curves_by_name,
            )
        )

        # ---------------------------------------------------------------------
        # Arrival: Stunden -> Minuten -> Wandzeit -> Raster (nearest)
        # ---------------------------------------------------------------------
        arrival_minutes = float(np.clip(arrival_hours * 60.0, 0.0, 24.0 * 60.0 - 1e-9))
        arrival_wall = day_start_ts + pd.Timedelta(minutes=arrival_minutes)

        arrival_abs_step = int(ts_index.get_indexer([arrival_wall], method="nearest")[0])

        # In den Tagesbereich clampen, damit die Session diesem Kalendertag zugeordnet bleibt
        arrival_abs_step = int(np.clip(arrival_abs_step, day_start_abs_step, day_end_excl - 1))
        arrival_time = pd.to_datetime(ts_index[arrival_abs_step]).to_pydatetime()

        # ---------------------------------------------------------------------
        # Departure: Wandzeit + Parkdauer -> Raster (CEIL, Ende-exklusiv)
        # ---------------------------------------------------------------------
        # Parkdauer mind. 1 Zeitschritt (damit departure > arrival)
        parking_duration_minutes = float(max(parking_duration_minutes, step_minutes))
        departure_wall = arrival_wall + pd.Timedelta(minutes=parking_duration_minutes)

        departure_abs_step = ceil_index(pd.Timestamp(departure_wall))

        # Mindestens 1 Step Standzeit erzwingen
        if departure_abs_step <= arrival_abs_step:
            departure_abs_step = arrival_abs_step + 1

        # Optional: über Mitternacht hinaus abschneiden
        if not allow_cross_day_charging:
            departure_abs_step = int(min(departure_abs_step, day_end_excl))
            if departure_abs_step <= arrival_abs_step:
                departure_abs_step = min(arrival_abs_step + 1, day_end_excl)

        # departure_time bestimmen (Ende-exklusiv => Timestamp an departure_abs_step)
        if departure_abs_step < len(ts_index):
            departure_time = pd.to_datetime(ts_index[departure_abs_step]).to_pydatetime()
        else:
            # Falls ganz am Ende des Horizonts: virtueller Endpunkt ein Schritt danach
            departure_time = (
                pd.to_datetime(ts_index[-1]) + pd.Timedelta(minutes=time_resolution_min)
            ).to_pydatetime()

        duration_minutes = float((departure_time - arrival_time).total_seconds() / 60.0)
        session_id = f"{day_start_ts.date()}_{day_index:03d}_{session_number:05d}"

        sampled_sessions.append(
            SampledSession(
                session_id=session_id,
                arrival_time=arrival_time,
                departure_time=departure_time,
                arrival_step=int(arrival_abs_step),
                departure_step=int(departure_abs_step),
                duration_minutes=float(duration_minutes),
                state_of_charge_at_arrival=float(soc_at_arrival),
                vehicle_name=str(vehicle_name),
                vehicle_class=str(vehicle_class),
                day_type=str(day_type),
            )
        )

    sampled_sessions.sort(key=lambda session: session.arrival_time)
    return sampled_sessions


# =============================================================================
# 3) Strategie: Reservierungsbasierte Session-Planung (immediate / market / generation)
# =============================================================================

def available_energy_for_session_step(
    *,
    step_index: int,
    scenario: dict,
    curve: VehicleChargingCurve,
    state_of_charge_fraction: float,
    remaining_site_energy_kwh: float,
    pv_generation_kwh_per_step: np.ndarray,
    base_load_kwh_per_step: np.ndarray,
    reserved_total_ev_energy_kwh_per_step: np.ndarray,
    reserved_pv_ev_energy_kwh_per_step: Optional[np.ndarray] = None,
    already_allocated_on_this_charger_kwh: float = 0.0,
    supply_mode: Literal["site", "pv_only", "grid_only"] = "site",
) -> Tuple[float, float]:
    """
    Bestimmt die maximal allokierbare Ladeenergie (kWh/Step) für eine Session in einem Schritt.

    Die verfügbare Energie ergibt sich aus PV und Netzlimit nach Deckung der Grundlast sowie nach
    Abzug bereits reservierter EV-Energie. Zusätzlich wird die Allokation durch das Ladepunktlimit,
    die Fahrzeug-Ladekurve (SoC-abhängige maximale Ladeleistung) und den Restenergiebedarf begrenzt.

    Vor der EV-Allokation wird geprüft, ob die Grundlast im Schritt durch PV und Netzlimit versorgbar
    ist. Ist dies nicht möglich, wird ein Fehler ausgelöst.

    Parameters
    ----------
    step_index:
        Absoluter Index im Simulationsraster.
    scenario:
        Szenario-Dictionary mit mindestens ``site.rated_power_kw``, ``site.grid_limit_p_avb_kw``,
        ``site.charger_efficiency`` und ``time_resolution_min``.
    curve:
        Fahrzeug-Ladekurve (SoC-Stützstellen und Ladeleistung auf Batterieseite).
    state_of_charge_fraction:
        SoC zu Beginn des Schritts als Anteil [0..1].
    remaining_site_energy_kwh:
        Noch benötigte Energie auf Standortseite (kWh).
    pv_generation_kwh_per_step, base_load_kwh_per_step:
        Zeitreihen für PV-Erzeugung und Grundlast in kWh/Step.
    reserved_total_ev_energy_kwh_per_step, reserved_pv_ev_energy_kwh_per_step:
        Bereits reservierte EV-Energie (gesamt und PV-Anteil) in kWh/Step.
    already_allocated_on_this_charger_kwh:
        Bereits im selben Schritt auf demselben Ladepunkt allokierte Energie (kWh/Step).
    supply_mode:
        Versorgungsmodus: ``"site"`` (PV+Netz), ``"pv_only"`` (nur PV) oder ``"grid_only"`` (nur Netz).

    Returns
    -------
    Tuple[float, float]
        ``(allocated_site_kwh, pv_share_kwh)`` mit der allokierten Energie auf Standortseite und
        dem PV-Anteil daran, jeweils in kWh/Step.

    Raises
    ------
    IndexError
        Wenn ``step_index`` außerhalb der Zeitreihenlänge liegt.
    ValueError
        Wenn die Grundlast im Schritt nicht durch PV und Netzlimit versorgbar ist oder Parameter
        im Szenario ungültig sind.
    """
    if step_index < 0 or step_index >= len(base_load_kwh_per_step):
        raise IndexError(f"step_index={step_index} außerhalb der Zeitreihenlänge.")

    site_configuration = scenario["site"]
    time_resolution_min = int(scenario["time_resolution_min"])
    step_hours = float(time_resolution_min) / 60.0

    charger_efficiency = float(site_configuration.get("charger_efficiency", 1.0))
    if not (0.0 < charger_efficiency <= 1.0):
        raise ValueError("site.charger_efficiency muss im Bereich (0, 1] liegen.")

    pv_energy_kwh = float(pv_generation_kwh_per_step[step_index])
    grid_energy_limit_kwh = float(site_configuration["grid_limit_p_avb_kw"]) * step_hours
    building_energy_kwh = float(base_load_kwh_per_step[step_index])

    reserved_total_ev_energy_kwh = float(reserved_total_ev_energy_kwh_per_step[step_index])
    reserved_pv_ev_energy_kwh = (
        float(reserved_pv_ev_energy_kwh_per_step[step_index])
        if reserved_pv_ev_energy_kwh_per_step is not None
        else 0.0
    )

    # (1) Harte Prüfung: Gebäude muss versorgbar sein (PV zuerst, Rest aus Netz bis Limit).
    building_energy_from_grid_kwh = building_energy_kwh - float(np.minimum(pv_energy_kwh, building_energy_kwh))
    if building_energy_from_grid_kwh > grid_energy_limit_kwh + 1e-9:
        raise ValueError(
            "Strombedarf (Grundlast) höher als verfügbare Energie am Standort: "
            f"building_energy_kwh={building_energy_kwh:.6f}, pv_energy_kwh={pv_energy_kwh:.6f}, "
            f"grid_energy_limit_kwh={grid_energy_limit_kwh:.6f}."
        )

    # (2) Verfügbare Energie für EV (PV nach Grundlast + Grid nach Grundlast)
    pv_after_building_kwh = float(max(pv_energy_kwh - building_energy_kwh, 0.0))
    pv_remaining_for_ev_kwh = float(max(pv_after_building_kwh - reserved_pv_ev_energy_kwh, 0.0))

    grid_remaining_after_building_kwh = float(grid_energy_limit_kwh - building_energy_from_grid_kwh)

    # "Grid-EV bereits" = reserved_total - (physikalisch möglicher PV-Anteil)
    pv_ev_physical_kwh = float(np.minimum(reserved_pv_ev_energy_kwh, pv_after_building_kwh))
    grid_ev_already_kwh = float(max(reserved_total_ev_energy_kwh - pv_ev_physical_kwh, 0.0))
    grid_remaining_for_ev_kwh = float(max(grid_remaining_after_building_kwh - grid_ev_already_kwh, 0.0))

    if supply_mode == "pv_only":
        supply_headroom_kwh = pv_remaining_for_ev_kwh
    elif supply_mode == "grid_only":
        supply_headroom_kwh = grid_remaining_for_ev_kwh
    elif supply_mode == "site":
        supply_headroom_kwh = pv_remaining_for_ev_kwh + grid_remaining_for_ev_kwh
    else:
        raise ValueError("supply_mode muss 'site', 'pv_only' oder 'grid_only' sein.")

    # (3) Chargerlimit
    charger_limit_kwh = float(site_configuration["rated_power_kw"]) * step_hours
    charger_headroom_kwh = float(max(charger_limit_kwh - already_allocated_on_this_charger_kwh, 0.0))

    # (4) Fahrzeuglimit aus Ladekurve (Batterieseite -> Standortseite)
    soc = float(np.clip(state_of_charge_fraction, 0.0, 1.0))
    battery_power_kw = float(np.interp(soc, curve.state_of_charge_fraction, curve.power_kw))
    vehicle_site_limit_kwh = float(max((battery_power_kw / max(charger_efficiency, 1e-12)) * step_hours, 0.0))

    remaining_site_energy_kwh = float(max(remaining_site_energy_kwh, 0.0))

    allocated_site_kwh = float(
        np.minimum.reduce(
            np.array(
                [supply_headroom_kwh, charger_headroom_kwh, vehicle_site_limit_kwh, remaining_site_energy_kwh],
                dtype=float,
            )
        )
    )
    allocated_site_kwh = float(max(allocated_site_kwh, 0.0))
    if allocated_site_kwh <= 1e-12:
        return 0.0, 0.0

    if supply_mode == "pv_only":
        pv_share_kwh = allocated_site_kwh
    elif supply_mode == "grid_only":
        pv_share_kwh = 0.0
    else:
        pv_share_kwh = float(np.minimum(allocated_site_kwh, pv_remaining_for_ev_kwh))

    return allocated_site_kwh, float(max(pv_share_kwh, 0.0))


def charging_strategy_immediate(
    *,
    session_arrival_step: int,
    session_departure_step: int,
    required_site_energy_kwh: float,
    pv_generation_kwh_per_step: np.ndarray,
    base_load_kwh_per_step: np.ndarray,
    reserved_total_ev_energy_kwh_per_step: np.ndarray,
    reserved_pv_ev_energy_kwh_per_step: np.ndarray,
    curve: VehicleChargingCurve,
    state_of_charge_at_arrival: float,
    scenario: dict,
) -> Dict[str, Any]:
    """
    Vergibt Energie an eine Session nach der Immediate-Strategie (chronologisch laden, solange möglich).

    Pro Zeitschritt wird die maximal allokierbare Energie über
    ``available_energy_for_session_step(...)`` bestimmt und anschließend
    (a) in Plan-Arrays geschrieben und (b) in globale Reservierungen addiert.

    Returns
    -------
    Dict[str, Any]
        Enthält nur session-relevante Ergebnisse:
        - plan_site_kwh_per_step, plan_pv_site_kwh_per_step
        - charged_site_kwh, charged_pv_site_kwh
        - remaining_site_kwh, final_soc
    """
    n_total = int(len(reserved_total_ev_energy_kwh_per_step))

    site_cfg = scenario["site"]
    charger_efficiency = float(site_cfg.get("charger_efficiency", 1.0))

    start = int(max(0, session_arrival_step))
    end_excl = int(min(int(session_departure_step), n_total))
    window_len = int(max(0, end_excl - start))

    plan_site_kwh_per_step = np.zeros(window_len, dtype=float)
    plan_pv_site_kwh_per_step = np.zeros(window_len, dtype=float)

    remaining_site_kwh = float(max(required_site_energy_kwh, 0.0))
    soc = float(np.clip(state_of_charge_at_arrival, 0.0, 1.0))

    battery_capacity_kwh = float(max(curve.battery_capacity_kwh, 1e-12))
    eff = float(charger_efficiency)

    charged_site_kwh = 0.0
    charged_pv_site_kwh = 0.0

    for abs_step in range(start, end_excl):
        if remaining_site_kwh <= 1e-12 or soc >= 1.0 - 1e-12:
            break

        alloc_site_kwh, alloc_pv_kwh = available_energy_for_session_step(
            step_index=int(abs_step),
            scenario=scenario,
            curve=curve,
            state_of_charge_fraction=float(soc),
            remaining_site_energy_kwh=float(remaining_site_kwh),
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
            reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
            already_allocated_on_this_charger_kwh=0.0,
            supply_mode="site",
        )

        alloc_site_kwh = float(alloc_site_kwh)
        if alloc_site_kwh <= 1e-12:
            continue

        alloc_pv_kwh = float(np.clip(float(alloc_pv_kwh), 0.0, alloc_site_kwh))

        local_i = int(abs_step - start)
        plan_site_kwh_per_step[local_i] += alloc_site_kwh
        plan_pv_site_kwh_per_step[local_i] += alloc_pv_kwh

        reserved_total_ev_energy_kwh_per_step[abs_step] += alloc_site_kwh
        reserved_pv_ev_energy_kwh_per_step[abs_step] += alloc_pv_kwh

        charged_site_kwh += alloc_site_kwh
        charged_pv_site_kwh += alloc_pv_kwh
        remaining_site_kwh -= alloc_site_kwh

        # SoC-Update (notwendig für Ladekurvenlimit im nächsten Step)
        battery_energy_kwh = alloc_site_kwh * eff
        soc = float(min(1.0, soc + battery_energy_kwh / battery_capacity_kwh))

    return {
        "plan_site_kwh_per_step": plan_site_kwh_per_step,
        "plan_pv_site_kwh_per_step": plan_pv_site_kwh_per_step,
        "charged_site_kwh": float(charged_site_kwh),
        "charged_pv_site_kwh": float(charged_pv_site_kwh),
        "remaining_site_kwh": float(max(remaining_site_kwh, 0.0)),
        "final_soc": float(soc),
    }



def charging_strategy_market(
    *,
    session_arrival_step: int,
    session_departure_step: int,
    required_site_energy_kwh: float,
    market_price_eur_per_mwh: np.ndarray,
    pv_generation_kwh_per_step: np.ndarray,
    base_load_kwh_per_step: np.ndarray,
    reserved_total_ev_energy_kwh_per_step: np.ndarray,
    reserved_pv_ev_energy_kwh_per_step: np.ndarray,
    curve: VehicleChargingCurve,
    state_of_charge_at_arrival: float,
    scenario: dict,
) -> Dict[str, Any]:
    """
    Vergibt Energie an eine Session nach der Market-Strategie (nur günstige Slots nutzen).

    Vorgehen
    --------
    1) Alle Zeitschritte der Session-Standzeit werden bestimmt.
    2) Diese Schritte werden nach Marktpreis aufsteigend sortiert.
    3) Es werden die k günstigsten Schritte ausgewählt, bis eine optimistische Kapazitätsschätzung
       (ohne SoC-Limit) theoretisch ausreichen würde. Falls das nicht möglich ist, werden alle
       Schritte zugelassen.
    4) Die tatsächliche Zuweisung erfolgt chronologisch und ausschließlich in den zugelassenen
       Schritten über ``available_energy_for_session_step(...)``.

    Wenn die Session innerhalb ihrer Standzeit nicht genug Energie bekommt (weil z. B. keine
    Standortenergie verfügbar ist), bleibt Restbedarf übrig. Es gibt keinen "must-charge".

    Returns
    -------
    Dict[str, Any]
        Session-relevante Ergebnisse:
        - plan_site_kwh_per_step, plan_pv_site_kwh_per_step, plan_market_kwh_per_step
        - charged_site_kwh, charged_pv_site_kwh
        - remaining_site_kwh, final_soc
    """
    n_total = int(len(reserved_total_ev_energy_kwh_per_step))

    site_cfg = scenario["site"]
    charger_efficiency = float(site_cfg.get("charger_efficiency", 1.0))

    start = int(max(0, session_arrival_step))
    end_excl = int(min(int(session_departure_step), n_total))
    window_len = int(max(0, end_excl - start))

    plan_site_kwh_per_step = np.zeros(window_len, dtype=float)
    plan_pv_site_kwh_per_step = np.zeros(window_len, dtype=float)
    plan_market_kwh_per_step = np.zeros(window_len, dtype=float)

    remaining_site_kwh = float(max(required_site_energy_kwh, 0.0))
    soc = float(np.clip(state_of_charge_at_arrival, 0.0, 1.0))

    if window_len == 0 or remaining_site_kwh <= 1e-12:
        return {
            "plan_site_kwh_per_step": plan_site_kwh_per_step,
            "plan_pv_site_kwh_per_step": plan_pv_site_kwh_per_step,
            "plan_market_kwh_per_step": plan_market_kwh_per_step,
            "charged_site_kwh": 0.0,
            "charged_pv_site_kwh": 0.0,
            "remaining_site_kwh": float(max(remaining_site_kwh, 0.0)),
            "final_soc": float(soc),
        }

    # --- Slot-Auswahl (k günstigste Slots bis "optimistisch ausreichend") ---
    time_resolution_min = int(scenario["time_resolution_min"])
    step_hours = float(time_resolution_min) / 60.0

    grid_limit_kwh = float(site_cfg["grid_limit_p_avb_kw"]) * step_hours
    charger_cap_kwh = float(site_cfg["rated_power_kw"]) * step_hours

    session_steps = np.arange(start, end_excl, dtype=int)
    prices_window = market_price_eur_per_mwh[session_steps].astype(float)

    # Optimistische Cap-Schätzung ohne SoC-Limit:
    # PV + GridLimit - Base - bereits reservierte EV (alles pro Step), dann durch ChargerCap begrenzen.
    supply_headroom_est = (
        pv_generation_kwh_per_step[session_steps].astype(float)
        + grid_limit_kwh
        - base_load_kwh_per_step[session_steps].astype(float)
        - reserved_total_ev_energy_kwh_per_step[session_steps].astype(float)
    )
    supply_headroom_est = np.maximum(supply_headroom_est, 0.0)

    cap_est = np.minimum(supply_headroom_est, charger_cap_kwh)
    cap_est = np.maximum(cap_est, 0.0)

    order_by_price = np.argsort(prices_window)  # cheapest first

    allowed = np.zeros(window_len, dtype=bool)
    cap_acc = 0.0
    for idx in order_by_price:
        allowed[idx] = True
        cap_acc += float(cap_est[idx])
        if cap_acc >= remaining_site_kwh - 1e-9:
            break

    # Wenn selbst optimistisch nicht genug möglich ist -> alle Slots erlauben
    if float(np.sum(cap_est)) < remaining_site_kwh - 1e-9:
        allowed[:] = True

    # --- Chronologische Zuweisung nur in erlaubten Slots ---
    battery_capacity_kwh = float(max(curve.battery_capacity_kwh, 1e-12))
    eff = float(charger_efficiency)

    charged_site_kwh = 0.0
    charged_pv_site_kwh = 0.0

    for local_i, abs_step in enumerate(range(start, end_excl)):
        if remaining_site_kwh <= 1e-12 or soc >= 1.0 - 1e-12:
            break
        if not bool(allowed[local_i]):
            continue

        alloc_site_kwh, alloc_pv_kwh = available_energy_for_session_step(
            step_index=int(abs_step),
            scenario=scenario,
            curve=curve,
            state_of_charge_fraction=float(soc),
            remaining_site_energy_kwh=float(remaining_site_kwh),
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
            reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
            already_allocated_on_this_charger_kwh=0.0,
            supply_mode="site",
        )

        alloc_site_kwh = float(alloc_site_kwh)
        if alloc_site_kwh <= 1e-12:
            continue

        alloc_pv_kwh = float(np.clip(float(alloc_pv_kwh), 0.0, alloc_site_kwh))

        plan_site_kwh_per_step[local_i] += alloc_site_kwh
        plan_pv_site_kwh_per_step[local_i] += alloc_pv_kwh
        plan_market_kwh_per_step[local_i] += alloc_site_kwh

        reserved_total_ev_energy_kwh_per_step[abs_step] += alloc_site_kwh
        reserved_pv_ev_energy_kwh_per_step[abs_step] += alloc_pv_kwh

        charged_site_kwh += alloc_site_kwh
        charged_pv_site_kwh += alloc_pv_kwh
        remaining_site_kwh -= alloc_site_kwh

        battery_energy_kwh = alloc_site_kwh * eff
        soc = float(min(1.0, soc + battery_energy_kwh / battery_capacity_kwh))

    return {
        "plan_site_kwh_per_step": plan_site_kwh_per_step,
        "plan_pv_site_kwh_per_step": plan_pv_site_kwh_per_step,
        "plan_market_kwh_per_step": plan_market_kwh_per_step,
        "charged_site_kwh": float(charged_site_kwh),
        "charged_pv_site_kwh": float(charged_pv_site_kwh),
        "remaining_site_kwh": float(max(remaining_site_kwh, 0.0)),
        "final_soc": float(soc),
    }


def charging_strategy_pv(
    *,
    scenario: dict,
    session_arrival_step: int,
    session_departure_step: int,
    required_site_energy_kwh: float,
    market_price_eur_per_mwh: np.ndarray,
    pv_generation_kwh_per_step: np.ndarray,
    base_load_kwh_per_step: np.ndarray,
    reserved_total_ev_energy_kwh_per_step: np.ndarray,
    reserved_pv_ev_energy_kwh_per_step: np.ndarray,
    curve: VehicleChargingCurve,
    state_of_charge_at_arrival: float,
) -> Dict[str, Any]:
    """
    Vergibt Energie an eine Session nach der PV-First-Strategie.

    Ablauf
    ------
    1) Innerhalb des Session-Fensters wird pro Schritt zuerst PV-only allokiert
       (``supply_mode="pv_only"``).
    2) Reicht PV über das Session-Fenster voraussichtlich nicht aus, werden zusätzliche
       Grid-Fallback-Slots über Marktpreis-Ranking ausgewählt.
    3) In den ausgewählten Schritten wird nach der PV-Allokation optional zusätzlich
       Grid-only allokiert (``supply_mode="grid_only"``), begrenzt durch Charger-Headroom,
       Site-Headroom, Ladekurve und Restenergiebedarf.

    Hinweis
    -------
    - Es gibt **kein must-charge**: Wenn die Session in ihren Slots keine Energie bekommt,
      bleibt Restenergie übrig.
    - Die Simulation läuft **chronologisch**, damit SoC und Ladekurvenlimit zeitkonsistent bleiben.

    Returns
    -------
    Dict[str, Any]
        Enthält session-relevante Ergebnisse:
        - plan_site_kwh_per_step, plan_pv_site_kwh_per_step, plan_market_kwh_per_step
        - charged_site_kwh, charged_pv_site_kwh, charged_market_kwh
        - remaining_site_kwh, final_soc
    """
    n_total = int(len(reserved_total_ev_energy_kwh_per_step))

    site_cfg = scenario["site"]
    charger_efficiency = float(site_cfg.get("charger_efficiency", 1.0))

    start = int(max(0, session_arrival_step))
    end_excl = int(min(int(session_departure_step), n_total))
    window_len = int(max(0, end_excl - start))

    plan_site_kwh_per_step = np.zeros(window_len, dtype=float)
    plan_pv_site_kwh_per_step = np.zeros(window_len, dtype=float)
    plan_market_kwh_per_step = np.zeros(window_len, dtype=float)

    remaining_site_kwh = float(max(required_site_energy_kwh, 0.0))
    soc = float(np.clip(state_of_charge_at_arrival, 0.0, 1.0))

    if window_len == 0 or remaining_site_kwh <= 1e-12:
        return {
            "plan_site_kwh_per_step": plan_site_kwh_per_step,
            "plan_pv_site_kwh_per_step": plan_pv_site_kwh_per_step,
            "plan_market_kwh_per_step": plan_market_kwh_per_step,
            "charged_site_kwh": 0.0,
            "charged_pv_site_kwh": 0.0,
            "charged_market_kwh": 0.0,
            "remaining_site_kwh": float(remaining_site_kwh),
            "final_soc": float(soc),
        }

    time_resolution_min = int(scenario["time_resolution_min"])
    step_hours = float(time_resolution_min) / 60.0

    battery_capacity_kwh = float(max(curve.battery_capacity_kwh, 1e-12))

    # -------------------------------------------------------------------------
    # Grid-Fallback-Slots auswählen (preisgünstige Schritte), basierend auf
    # einem SoC-unabhängigen Kapazitäts-Estimate.
    # -------------------------------------------------------------------------
    window_steps = np.arange(start, end_excl, dtype=int)

    pv_energy = pv_generation_kwh_per_step[window_steps].astype(float)
    base_energy = base_load_kwh_per_step[window_steps].astype(float)
    reserved_total = reserved_total_ev_energy_kwh_per_step[window_steps].astype(float)
    reserved_pv = reserved_pv_ev_energy_kwh_per_step[window_steps].astype(float)

    pv_after_building = np.maximum(pv_energy - base_energy, 0.0)
    pv_remaining_for_ev = np.maximum(pv_after_building - reserved_pv, 0.0)

    building_from_grid = np.maximum(base_energy - pv_energy, 0.0)
    grid_limit_kwh = float(site_cfg["grid_limit_p_avb_kw"]) * step_hours
    grid_remaining_after_building = np.maximum(grid_limit_kwh - building_from_grid, 0.0)

    pv_ev_physical = np.minimum(reserved_pv, pv_after_building)
    grid_ev_already = np.maximum(reserved_total - pv_ev_physical, 0.0)
    grid_remaining_for_ev = np.maximum(grid_remaining_after_building - grid_ev_already, 0.0)

    charger_limit_kwh = float(site_cfg["rated_power_kw"]) * step_hours

    pv_cap_est = np.minimum(pv_remaining_for_ev, charger_limit_kwh)
    grid_cap_est = np.minimum(grid_remaining_for_ev, charger_limit_kwh)

    pv_total_est = float(np.sum(pv_cap_est))
    grid_needed_est = float(max(0.0, remaining_site_kwh - pv_total_est))

    preferred_grid = np.zeros(window_len, dtype=bool)
    if grid_needed_est > 1e-9:
        prices = market_price_eur_per_mwh[window_steps].astype(float)
        order = np.argsort(prices)  # cheapest first

        cap_sorted = grid_cap_est[order]
        cumsum = np.cumsum(cap_sorted)
        k = int(np.searchsorted(cumsum, grid_needed_est, side="left")) + 1
        k = int(np.clip(k, 0, window_len))

        if k > 0:
            preferred_grid[order[:k]] = True

    # -------------------------------------------------------------------------
    # Chronologische Allokation: erst PV-only, dann (wenn Slot ausgewählt) Grid-only
    # -------------------------------------------------------------------------
    charged_site_kwh = 0.0
    charged_pv_site_kwh = 0.0
    charged_market_kwh = 0.0

    for local_i, abs_step in enumerate(range(start, end_excl)):
        if remaining_site_kwh <= 1e-12 or soc >= 1.0 - 1e-12:
            break

        # (1) PV-only
        pv_alloc_kwh, pv_share_kwh = available_energy_for_session_step(
            step_index=int(abs_step),
            scenario=scenario,
            curve=curve,
            state_of_charge_fraction=float(soc),
            remaining_site_energy_kwh=float(remaining_site_kwh),
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
            reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
            already_allocated_on_this_charger_kwh=0.0,
            supply_mode="pv_only",
        )

        pv_alloc_kwh = float(max(pv_alloc_kwh, 0.0))
        if pv_alloc_kwh > 1e-12:
            # pv_only => pv_share == pv_alloc (sicherheitshalber clamp)
            pv_share_kwh = float(np.clip(float(pv_share_kwh), 0.0, pv_alloc_kwh))

            plan_site_kwh_per_step[local_i] += pv_alloc_kwh
            plan_pv_site_kwh_per_step[local_i] += pv_share_kwh

            reserved_total_ev_energy_kwh_per_step[abs_step] += pv_alloc_kwh
            reserved_pv_ev_energy_kwh_per_step[abs_step] += pv_share_kwh

            charged_site_kwh += pv_alloc_kwh
            charged_pv_site_kwh += pv_share_kwh
            remaining_site_kwh -= pv_alloc_kwh

            battery_energy_kwh = pv_alloc_kwh * charger_efficiency
            soc = float(min(1.0, soc + battery_energy_kwh / battery_capacity_kwh))

            if remaining_site_kwh <= 1e-12 or soc >= 1.0 - 1e-12:
                break

        # (2) Grid-only Fallback in ausgewählten Slots
        if not bool(preferred_grid[local_i]):
            continue

        grid_alloc_kwh, _grid_pv_share = available_energy_for_session_step(
            step_index=int(abs_step),
            scenario=scenario,
            curve=curve,
            state_of_charge_fraction=float(soc),
            remaining_site_energy_kwh=float(remaining_site_kwh),
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
            reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
            already_allocated_on_this_charger_kwh=float(pv_alloc_kwh),
            supply_mode="grid_only",
        )

        grid_alloc_kwh = float(max(grid_alloc_kwh, 0.0))
        if grid_alloc_kwh <= 1e-12:
            continue

        plan_site_kwh_per_step[local_i] += grid_alloc_kwh
        plan_market_kwh_per_step[local_i] += grid_alloc_kwh

        reserved_total_ev_energy_kwh_per_step[abs_step] += grid_alloc_kwh

        charged_site_kwh += grid_alloc_kwh
        charged_market_kwh += grid_alloc_kwh
        remaining_site_kwh -= grid_alloc_kwh

        battery_energy_kwh = grid_alloc_kwh * charger_efficiency
        soc = float(min(1.0, soc + battery_energy_kwh / battery_capacity_kwh))

    return {
        "plan_site_kwh_per_step": plan_site_kwh_per_step,
        "plan_pv_site_kwh_per_step": plan_pv_site_kwh_per_step,
        "plan_market_kwh_per_step": plan_market_kwh_per_step,
        "charged_site_kwh": float(charged_site_kwh),
        "charged_pv_site_kwh": float(charged_pv_site_kwh),
        "charged_market_kwh": float(charged_market_kwh),
        "remaining_site_kwh": float(max(remaining_site_kwh, 0.0)),
        "final_soc": float(soc),
    }


def order_sessions_for_planning(
    sessions: list[SampledSession],
    charging_strategy: str,
) -> list[SampledSession]:
    """
    Sortiert Sessions in der Reihenfolge, in der sie geplant werden sollen.

    - "immediate": FCFS (Ankunft zuerst)
    - sonst (z. B. "market", "pv", "generation"): kürzeste Standzeit zuerst

    Tie-Breaker: arrival_step, session_id (stabil & reproduzierbar).
    """
    strategy = str(charging_strategy).strip().lower()

    def duration_steps(s: SampledSession) -> int:
        return int(max(0, int(s.departure_step) - int(s.arrival_step)))

    if strategy == "immediate":
        return sorted(
            sessions,
            key=lambda s: (int(s.arrival_step), int(s.departure_step), str(s.session_id)),
        )

    return sorted(
        sessions,
        key=lambda s: (duration_steps(s), int(s.arrival_step), str(s.session_id)),
    )


def required_site_energy_for_session(
    *,
    scenario: dict,
    curve: VehicleChargingCurve,
    soc_at_arrival: float,
) -> float:
    """
    Berechnet die benötigte Energie auf Standortseite (kWh) bis zum Ziel-SoC.

    Formel
    ------
    needed_battery_kwh = max(0, (soc_target - soc0) * battery_capacity_kwh)
    needed_site_kwh    = needed_battery_kwh / charger_efficiency
    """
    vehicles_cfg = scenario["vehicles"]
    site_cfg = scenario["site"]

    soc_target = float(np.clip(float(vehicles_cfg.get("soc_target", 1.0)), 0.0, 1.0))
    soc0 = float(np.clip(float(soc_at_arrival), 0.0, 1.0))

    battery_capacity_kwh = float(max(curve.battery_capacity_kwh, 1e-12))
    charger_efficiency = float(site_cfg.get("charger_efficiency", 1.0))

    needed_battery_kwh = float(max(0.0, (soc_target - soc0) * battery_capacity_kwh))
    needed_site_kwh = float(needed_battery_kwh / max(charger_efficiency, 1e-12))
    return float(max(needed_site_kwh, 0.0))


def plan_charging_for_day(
    *,
    sessions: list[SampledSession],
    scenario: dict,
    vehicle_curves_by_name: dict[str, VehicleChargingCurve],
    pv_generation_kwh_per_step: np.ndarray,
    base_load_kwh_per_step: np.ndarray,
    reserved_total_ev_energy_kwh_per_step: np.ndarray,
    reserved_pv_ev_energy_kwh_per_step: np.ndarray,
    market_price_eur_per_mwh: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """
    Plant alle Sessions eines Tages abhängig von scenario["charging_strategy"].

    Reihenfolge:
    - immediate: FCFS
    - market/pv/generation: shortest parking duration first

    Gibt pro Session ein Ergebnisdict zurück (z. B. Pläne, geladene Energiemengen, final_soc).
    """
    strategy = str(scenario["charging_strategy"]).strip().lower()
    ordered = order_sessions_for_planning(sessions, strategy)

    charger_efficiency = float(scenario["site"].get("charger_efficiency", 1.0))

    results: list[dict[str, Any]] = []

    for s in ordered:
        curve = vehicle_curves_by_name.get(s.vehicle_name)
        if curve is None:
            raise ValueError(f"Keine Ladekurve für vehicle_name='{s.vehicle_name}' gefunden.")

        required_site_energy_kwh = required_site_energy_for_session(
            scenario=scenario,
            curve=curve,
            soc_at_arrival=float(s.state_of_charge_at_arrival),
        )

        if strategy == "immediate":
            res = charging_strategy_immediate(
                scenario=scenario,
                session_arrival_step=int(s.arrival_step),
                session_departure_step=int(s.departure_step),
                required_site_energy_kwh=float(required_site_energy_kwh),
                pv_generation_kwh_per_step=pv_generation_kwh_per_step,
                base_load_kwh_per_step=base_load_kwh_per_step,
                reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
                reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
                curve=curve,
                state_of_charge_at_arrival=float(s.state_of_charge_at_arrival),
            )

        elif strategy == "market":
            if market_price_eur_per_mwh is None:
                raise ValueError("charging_strategy='market' benötigt market_price_eur_per_mwh.")

            res = charging_strategy_market(
                scenario=scenario,
                session_arrival_step=int(s.arrival_step),
                session_departure_step=int(s.departure_step),
                required_site_energy_kwh=float(required_site_energy_kwh),
                market_price_eur_per_mwh=market_price_eur_per_mwh,
                pv_generation_kwh_per_step=pv_generation_kwh_per_step,
                base_load_kwh_per_step=base_load_kwh_per_step,
                reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
                reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
                curve=curve,
                state_of_charge_at_arrival=float(s.state_of_charge_at_arrival),
            )

        elif strategy in ("pv", "generation"):
            if market_price_eur_per_mwh is None:
                raise ValueError(
                    "charging_strategy='pv' benötigt market_price_eur_per_mwh (Fallback-Ranking)."
                )

            res = charging_strategy_pv(
                scenario=scenario,
                session_arrival_step=int(s.arrival_step),
                session_departure_step=int(s.departure_step),
                required_site_energy_kwh=float(required_site_energy_kwh),
                market_price_eur_per_mwh=market_price_eur_per_mwh,
                pv_generation_kwh_per_step=pv_generation_kwh_per_step,
                base_load_kwh_per_step=base_load_kwh_per_step,
                reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
                reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
                curve=curve,
                state_of_charge_at_arrival=float(s.state_of_charge_at_arrival),
            )

        else:
            raise ValueError(f"Unbekannte charging_strategy='{strategy}'")

        # Meta-Infos ergänzen (praktisch fürs Logging/Export)
        res["session_id"] = s.session_id
        res["vehicle_name"] = s.vehicle_name
        res["vehicle_class"] = s.vehicle_class
        res["arrival_step"] = int(s.arrival_step)
        res["departure_step"] = int(s.departure_step)
        res["required_site_kwh"] = float(required_site_energy_kwh)

        results.append(res)

    return results


# =============================================================================
# 4) Physik + Simulation: FCFS, Ladepunkt-Zuordnung, Reservierungs-Lastgang
# =============================================================================

def find_free_charger_fcfs(
    charger_occupied_until_step: list[int],
    arrival_step: int,
) -> int | None:
    """
    Gibt die erste freie Charger-ID nach FCFS zurück (kleinste ID zuerst).

    Ein Charger gilt als frei, wenn sein Eintrag in ``charger_occupied_until_step`` <= arrival_step ist.
    """
    for charger_id, occupied_until in enumerate(charger_occupied_until_step):
        if int(occupied_until) <= int(arrival_step):
            return int(charger_id)
    return None


def _group_sessions_by_arrival_day(
    sessions: list[SampledSession],
    timestamps: pd.DatetimeIndex,
) -> dict[pd.Timestamp, list[SampledSession]]:
    """Gruppiert Sessions nach Kalendertag des arrival_step (TZ-sicher über timestamps)."""
    by_day: dict[pd.Timestamp, list[SampledSession]] = {}
    for s in sessions:
        day_key = pd.Timestamp(timestamps[int(s.arrival_step)]).normalize()
        by_day.setdefault(day_key, []).append(s)
    return by_day


def _compute_debug_balance(
    *,
    timestamps: pd.DatetimeIndex,
    scenario: dict,
    pv_generation_kwh_per_step: np.ndarray,
    base_load_kwh_per_step: np.ndarray,
    reserved_total_ev_energy_kwh_per_step: np.ndarray,
    reserved_pv_ev_energy_kwh_per_step: np.ndarray,
) -> pd.DataFrame:
    """
    Baut eine einfache, konsistente Energiebilanz (PV->Base, PV->EV, Grid->Base, Grid->EV) als DataFrame.
    """
    time_resolution_min = int(scenario["time_resolution_min"])
    step_hours = float(time_resolution_min) / 60.0
    grid_limit_kwh = float(scenario["site"]["grid_limit_p_avb_kw"]) * step_hours

    pv = np.asarray(pv_generation_kwh_per_step, dtype=float)
    base = np.asarray(base_load_kwh_per_step, dtype=float)
    ev = np.asarray(reserved_total_ev_energy_kwh_per_step, dtype=float)
    pv_ev_tracked = np.asarray(reserved_pv_ev_energy_kwh_per_step, dtype=float)

    pv_to_base = np.minimum(pv, base)
    base_remaining = base - pv_to_base

    pv_after_base = np.maximum(pv - pv_to_base, 0.0)
    pv_to_ev = np.minimum(pv_ev_tracked, pv_after_base)
    ev_remaining = np.maximum(ev - pv_to_ev, 0.0)

    grid_to_base = np.minimum(base_remaining, grid_limit_kwh)
    grid_headroom_after_base = np.maximum(grid_limit_kwh - grid_to_base, 0.0)
    grid_to_ev = np.minimum(ev_remaining, grid_headroom_after_base)

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "pv_generation_kwh_per_step": pv,
            "base_load_kwh_per_step": base,
            "ev_load_kwh_per_step": ev,
            "pv_ev_tracked_kwh_per_step": pv_ev_tracked,
            "pv_to_base_kwh_per_step": pv_to_base,
            "pv_to_ev_kwh_per_step": pv_to_ev,
            "grid_to_base_kwh_per_step": grid_to_base,
            "grid_to_ev_kwh_per_step": grid_to_ev,
            "grid_limit_kwh_per_step": grid_limit_kwh,
        }
    )
    return df


def simulate_site_fcfs_with_planning(
    *,
    scenario: dict,
    timestamps: pd.DatetimeIndex,
    sessions: list[SampledSession],
    vehicle_curves_by_name: dict[str, VehicleChargingCurve],
    pv_generation_kwh_per_step: np.ndarray,
    base_load_kwh_per_step: np.ndarray,
    reserved_total_ev_energy_kwh_per_step: Optional[np.ndarray] = None,
    reserved_pv_ev_energy_kwh_per_step: Optional[np.ndarray] = None,
    market_price_eur_per_mwh: Optional[np.ndarray] = None,
    record_debug: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    """
    Kern-Simulation: Admission (FCFS) + Tagesplanung (Strategie) + EV-Lastgang.

    Ablauf
    ------
    1) FCFS-Admission: Charger-Vergabe, Sessions ohne freien Charger -> "drive_off"
    2) Tagesplanung: pro Arrival-Tag ``plan_charging_for_day(...)`` (nutzt Strategien + Reservierungen)
    3) Lastgang: ``ev_load_kw`` aus reservierter EV-Energie (kWh/Step) / step_hours
    4) Optional: Debug-Bilanz als DataFrame

    Returns
    -------
    (ev_load_kw, sessions_out, reserved_total_kwh, reserved_pv_kwh, debug_df)
    """
    n_total = int(len(timestamps))
    time_resolution_min = int(scenario["time_resolution_min"])
    step_hours = float(time_resolution_min) / 60.0

    # Reservierungs-Arrays (global, in-place durch Strategien gefüllt)
    if reserved_total_ev_energy_kwh_per_step is None:
        reserved_total_ev_energy_kwh_per_step = np.zeros(n_total, dtype=float)
    if reserved_pv_ev_energy_kwh_per_step is None:
        reserved_pv_ev_energy_kwh_per_step = np.zeros(n_total, dtype=float)

    reserved_total = reserved_total_ev_energy_kwh_per_step
    reserved_pv = reserved_pv_ev_energy_kwh_per_step

    number_chargers = int(scenario["site"]["number_chargers"])
    charger_occupied_until: list[int] = [0] * number_chargers

    # --- Admission: FCFS (nach Ankunft) ---
    sessions_sorted = sorted(sessions, key=lambda s: (int(s.arrival_step), str(s.session_id)))

    charger_id_by_session_id: dict[str, int] = {}
    plugged: list[SampledSession] = []
    sessions_out: list[dict[str, Any]] = []

    for s in sessions_sorted:
        a = int(np.clip(int(s.arrival_step), 0, n_total - 1))
        d = int(np.clip(int(s.departure_step), 0, n_total))
        if d <= a:
            d = min(a + 1, n_total)

        s.arrival_step = a
        s.departure_step = d

        cid = find_free_charger_fcfs(charger_occupied_until, a)
        if cid is None:
            sessions_out.append(
                {
                    "session_id": s.session_id,
                    "vehicle_name": s.vehicle_name,
                    "vehicle_class": s.vehicle_class,
                    "arrival_step": a,
                    "departure_step": d,
                    "status": "drive_off",
                    "charger_id": None,
                    "charged_site_kwh": 0.0,
                    "charged_pv_site_kwh": 0.0,
                    "remaining_site_kwh": np.nan,
                    "final_soc": float(np.clip(s.state_of_charge_at_arrival, 0.0, 1.0)),
                }
            )
            continue

        charger_occupied_until[cid] = d
        charger_id_by_session_id[str(s.session_id)] = cid
        plugged.append(s)

    # --- Planung: pro Arrival-Tag ---
    plugged_by_day = _group_sessions_by_arrival_day(plugged, timestamps)

    for day_key in sorted(plugged_by_day.keys()):
        day_sessions = plugged_by_day[day_key]

        day_results = plan_charging_for_day(
            sessions=day_sessions,
            scenario=scenario,
            vehicle_curves_by_name=vehicle_curves_by_name,
            pv_generation_kwh_per_step=np.asarray(pv_generation_kwh_per_step, dtype=float),
            base_load_kwh_per_step=np.asarray(base_load_kwh_per_step, dtype=float),
            reserved_total_ev_energy_kwh_per_step=reserved_total,
            reserved_pv_ev_energy_kwh_per_step=reserved_pv,
            market_price_eur_per_mwh=np.asarray(market_price_eur_per_mwh, dtype=float)
            if market_price_eur_per_mwh is not None
            else None,
        )

        # Charger-ID + Status ergänzen
        for r in day_results:
            sid = str(r["session_id"])
            r["status"] = "plugged"
            r["charger_id"] = int(charger_id_by_session_id[sid])
            sessions_out.append(r)

    ev_load_kw = np.asarray(reserved_total, dtype=float) / max(step_hours, 1e-12)

    debug_df = None
    if record_debug:
        debug_df = _compute_debug_balance(
            timestamps=timestamps,
            scenario=scenario,
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_total_ev_energy_kwh_per_step=reserved_total,
            reserved_pv_ev_energy_kwh_per_step=reserved_pv,
        )

    return ev_load_kw, sessions_out, reserved_total, reserved_pv, debug_df


# =============================================================================
# 5) Analyse / Validierung / Notebook-Helper
# =============================================================================


# -----------------------------------------------------------------------------
# A) Scenario / Time helpers
# -----------------------------------------------------------------------------

def get_time_resolution_min_from_scenario(scenario: dict) -> int:
    """Liest time_resolution_min robust aus dem Szenario."""
    return int(scenario.get("time_resolution_min", 15))


def get_charging_strategy_from_scenario(scenario: dict) -> str:
    """
    Normalisiert scenario["charging_strategy"] auf:
    - "immediate"
    - "market"
    - "pv"
    """
    raw = str(scenario.get("charging_strategy", "immediate")).strip().lower()
    if raw in {"generation", "pv", "pv_generation", "pv_first", "pv-first"}:
        return "pv"
    if raw in {"market"}:
        return "market"
    return "immediate"


def get_holiday_dates_from_scenario(scenario: dict, timestamps: pd.DatetimeIndex) -> set[date]:
    """
    Liefert Feiertage als set[date]. Minimalistisch:

    Unterstützt:
    - scenario["holidays"]["dates"] = ["2026-01-01", ...]
    - scenario["holidays"]["country"] (optional) über python 'holidays' falls installiert
    """
    holidays_cfg = scenario.get("holidays") or {}
    out: set[date] = set()

    explicit = holidays_cfg.get("dates")
    if isinstance(explicit, list) and explicit:
        for x in explicit:
            try:
                out.add(pd.to_datetime(x).date())
            except Exception:
                continue
        return out

    country = str(holidays_cfg.get("country", "")).strip()
    if not country:
        return out

    try:
        import holidays as _holidays  # type: ignore
    except Exception:
        warnings.warn(
            "python-package 'holidays' nicht installiert -> holiday_dates bleibt leer.",
            UserWarning,
        )
        return out

    years = sorted({pd.Timestamp(t).year for t in timestamps})
    subdiv = holidays_cfg.get("subdivision") or holidays_cfg.get("state") or None

    try:
        cal = _holidays.country_holidays(country, subdiv=subdiv, years=years)
        out = {d for d in cal.keys()}
    except Exception:
        return set()

    return out


def get_daytype_calendar(
    *,
    start_datetime: pd.Timestamp,
    horizon_days: int,
    holiday_dates: set[date],
) -> dict[str, list[date]]:
    """Gibt einen Kalender {day_type: [dates]} über den Horizont zurück."""
    out = {"working_day": [], "saturday": [], "sunday_holiday": []}
    start_date = pd.Timestamp(start_datetime).date()

    for i in range(int(horizon_days)):
        d = (pd.Timestamp(start_date) + pd.Timedelta(days=i)).date()
        wd = int(pd.Timestamp(d).weekday())  # 0=Mo..6=So
        if d in holiday_dates or wd == 6:
            out["sunday_holiday"].append(d)
        elif wd == 5:
            out["saturday"].append(d)
        else:
            out["working_day"].append(d)
    return out


def decorate_title_with_status(title: str, charging_strategy: str, strategy_status: str | None = None) -> str:
    """Kleiner Helper für Plot-Titel."""
    cs = str(charging_strategy)
    ss = str(strategy_status) if strategy_status is not None else ""
    if ss and ss != cs:
        return f"{title} — {cs} ({ss})"
    return f"{title} — {cs}"


def initialize_time_window(
    *,
    timestamps: pd.DatetimeIndex,
    scenario: dict,
    days: int = 1,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Liefert (window_start, window_end) für Zoom-Plots.

    Konvention:
    - window_start = erster Timestamp
    - window_end   = letzter Timestamp innerhalb der ersten `days` Tage (inklusive)
    """
    if len(timestamps) == 0:
        t = pd.Timestamp("1970-01-01")
        return t, t

    time_resolution_min = get_time_resolution_min_from_scenario(scenario)
    steps_per_day = int(round(24 * 60 / time_resolution_min))
    n = int(min(len(timestamps), max(1, int(days)) * steps_per_day))

    start = pd.Timestamp(timestamps[0])
    end = pd.Timestamp(timestamps[n - 1])
    return start, end


# -----------------------------------------------------------------------------
# B) Sessions: Grouping / KPI / Preview
# -----------------------------------------------------------------------------

def group_sessions_by_day(
    sessions_out: list[dict[str, Any]],
    *,
    only_plugged: bool = False,
) -> dict[date, list[dict[str, Any]]]:
    """
    Gruppiert sessions_out nach Kalendertag.

    Bevorzugt arrival_time (falls vorhanden), sonst None-Day.
    """
    out: dict[date, list[dict[str, Any]]] = {}

    for s in sessions_out:
        if only_plugged and str(s.get("status")) != "plugged":
            continue

        arrival_time = s.get("arrival_time")
        if arrival_time is None:
            # fallback: pack into a synthetic bucket
            key = date(1970, 1, 1)
        else:
            key = pd.to_datetime(arrival_time).date()

        out.setdefault(key, []).append(s)

    return out


def summarize_sessions(sessions_out: list[dict[str, Any]], *, tol_kwh: float = 1e-9) -> dict[str, Any]:
    """
    KPI-Summary im Stil deines Notebooks.

    Erwartet in sessions_out möglichst:
    - status ("plugged" / "drive_off")
    - remaining_site_kwh
    - final_soc (oder state_of_charge_end)
    - charger_id, arrival_time, departure_time
    """
    total = int(len(sessions_out))
    plugged = [s for s in sessions_out if str(s.get("status")) == "plugged"]
    rejected = [s for s in sessions_out if str(s.get("status")) == "drive_off"]

    not_reached_rows: list[dict[str, Any]] = []
    for s in plugged:
        remaining = float(s.get("remaining_site_kwh", 0.0) if s.get("remaining_site_kwh") is not None else 0.0)
        if remaining > tol_kwh:
            arrival_time = s.get("arrival_time")
            departure_time = s.get("departure_time")
            parking_min = s.get("parking_duration_min")

            if parking_min is None and arrival_time is not None and departure_time is not None:
                try:
                    parking_min = float(
                        (pd.to_datetime(departure_time) - pd.to_datetime(arrival_time)).total_seconds() / 60.0
                    )
                except Exception:
                    parking_min = np.nan

            not_reached_rows.append(
                {
                    "session_id": s.get("session_id"),
                    "charger_id": s.get("charger_id"),
                    "arrival_time": arrival_time,
                    "parking_duration_min": parking_min,
                    "soc_arrival": s.get("state_of_charge_at_arrival", s.get("soc_arrival")),
                    "soc_end": s.get("final_soc", s.get("state_of_charge_end")),
                    "remaining_energy_kwh": remaining,  # notebook-kompatibel
                }
            )

    return {
        "num_sessions_total": total,
        "num_sessions_plugged": int(len(plugged)),
        "num_sessions_rejected": int(len(rejected)),
        "not_reached_rows": not_reached_rows,
    }


def build_plugged_sessions_preview_table(sessions_out: list[dict[str, Any]], *, n: int = 10) -> pd.DataFrame:
    """Kleine Preview-Tabelle (nur plugged Sessions)."""
    rows = []
    for s in sessions_out:
        if str(s.get("status")) != "plugged":
            continue
        rows.append(
            {
                "session_id": s.get("session_id"),
                "charger_id": s.get("charger_id"),
                "arrival_time": s.get("arrival_time"),
                "departure_time": s.get("departure_time"),
                "parking_duration_min": s.get("parking_duration_min"),
                "soc_arrival": s.get("state_of_charge_at_arrival", s.get("soc_arrival")),
                "soc_end": s.get("final_soc", s.get("state_of_charge_end")),
                "charged_site_kwh": s.get("charged_site_kwh"),
                "charged_pv_site_kwh": s.get("charged_pv_site_kwh"),
                "remaining_site_kwh": s.get("remaining_site_kwh"),
                "vehicle_name": s.get("vehicle_name"),
            }
        )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df = df.sort_values(["arrival_time", "session_id"], na_position="last").reset_index(drop=True)
    return df.head(int(max(0, n)))


# -----------------------------------------------------------------------------
# C) Timeseries DataFrames
# -----------------------------------------------------------------------------

def build_timeseries_dataframe(
    *,
    timestamps: pd.DatetimeIndex,
    scenario: dict,
    base_load_kwh_per_step: np.ndarray,
    pv_generation_kwh_per_step: np.ndarray,
    ev_load_kw: np.ndarray,
    market_price_eur_per_mwh: Optional[np.ndarray] = None,
    debug_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Baut timeseries_dataframe für Notebook-Plots.

    Garantierte Spalten:
    - timestamp
    - base_load_kw
    - pv_generation_kw
    - ev_load_kw
    - market_price_eur_per_mwh (falls vorhanden)
    Zusätzlich (falls debug_df vorhanden): pv_to_ev_kwh_per_step, grid_to_ev_kwh_per_step etc.
    """
    time_resolution_min = get_time_resolution_min_from_scenario(scenario)
    step_hours = float(time_resolution_min) / 60.0

    base_kw = np.asarray(base_load_kwh_per_step, dtype=float) / max(step_hours, 1e-12)
    pv_kw = np.asarray(pv_generation_kwh_per_step, dtype=float) / max(step_hours, 1e-12)
    ev_kw = np.asarray(ev_load_kw, dtype=float)

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "base_load_kw": base_kw,
            "pv_generation_kw": pv_kw,
            "ev_load_kw": ev_kw,
        }
    )

    if market_price_eur_per_mwh is not None:
        df["market_price_eur_per_mwh"] = np.asarray(market_price_eur_per_mwh, dtype=float)

    if debug_df is not None and len(debug_df) == len(df):
        # merge by position (safe if built from same timestamps)
        for c in debug_df.columns:
            if c == "timestamp":
                continue
            if c not in df.columns:
                df[c] = debug_df[c].to_numpy()

    return df


def build_site_overview_plot_data(
    *,
    timeseries_dataframe: pd.DataFrame,
    scenario: dict,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> dict[str, Any]:
    """
    Liefert Plot-Daten für Standort-Übersicht:
    - dataframe (gefiltert)
    - total_load_kw
    - pv_generation_kw
    - grid_limit_kw
    """
    df = timeseries_dataframe.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end)]

    base = df.get("base_load_kw", 0.0)
    ev = df.get("ev_load_kw", 0.0)
    total = base.astype(float).fillna(0.0) + ev.astype(float).fillna(0.0)

    pv = df.get("pv_generation_kw")
    pv_out = pv.astype(float).fillna(0.0) if pv is not None else None

    grid_limit_kw = float(scenario["site"].get("grid_limit_p_avb_kw", 0.0))

    return {
        "dataframe": df.reset_index(drop=True),
        "total_load_kw": total.to_numpy(dtype=float),
        "pv_generation_kw": None if pv_out is None else pv_out.to_numpy(dtype=float),
        "grid_limit_kw": grid_limit_kw,
    }


def build_ev_power_by_source_timeseries(timeseries_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    EV-Leistung nach Quelle (PV vs Grid).
    Erwartet idealerweise debug-Spalten (kWh/step) aus _compute_debug_balance:
      - pv_to_ev_kwh_per_step, grid_to_ev_kwh_per_step
    Fallback: nutzt pv_ev_tracked_kwh_per_step oder approximiert über ev_load_kw.
    """
    df = timeseries_dataframe.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # step_hours aus Timestamp-Diff schätzen
    if len(df) >= 2:
        dt_h = float((df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds()) / 3600.0
        step_hours = max(dt_h, 1e-12)
    else:
        step_hours = 0.25  # default 15min

    ev_kw = df.get("ev_load_kw", pd.Series(0.0, index=df.index)).astype(float).fillna(0.0).to_numpy()

    if "pv_to_ev_kwh_per_step" in df.columns:
        pv_kw = df["pv_to_ev_kwh_per_step"].astype(float).fillna(0.0).to_numpy() / step_hours
        grid_kw = df.get("grid_to_ev_kwh_per_step", 0.0)
        if isinstance(grid_kw, pd.Series):
            grid_kw = grid_kw.astype(float).fillna(0.0).to_numpy() / step_hours
        else:
            grid_kw = np.maximum(ev_kw - pv_kw, 0.0)
    elif "pv_ev_tracked_kwh_per_step" in df.columns:
        pv_kw = df["pv_ev_tracked_kwh_per_step"].astype(float).fillna(0.0).to_numpy() / step_hours
        pv_kw = np.clip(pv_kw, 0.0, ev_kw)
        grid_kw = np.maximum(ev_kw - pv_kw, 0.0)
    else:
        pv_kw = np.zeros_like(ev_kw)
        grid_kw = ev_kw

    return pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "ev_from_pv_kw": pv_kw,
            "ev_from_grid_kw": grid_kw,
        }
    )


# -----------------------------------------------------------------------------
# D) Charger-Traces + per-charger plots + heatmap
# -----------------------------------------------------------------------------

def enrich_sessions_out_with_metadata(
    *,
    sessions_out: list[dict[str, Any]],
    sampled_sessions: list[SampledSession],
) -> list[dict[str, Any]]:
    """
    Fügt arrival_time/departure_time/duration + soc_arrival etc. aus sampled_sessions hinzu.
    (Sehr hilfreich, damit dein Notebook-Histogramm etc. stabil laufen.)
    """
    by_id: dict[str, SampledSession] = {str(s.session_id): s for s in sampled_sessions}

    out: list[dict[str, Any]] = []
    for row in sessions_out:
        r = dict(row)
        sid = str(r.get("session_id"))
        meta = by_id.get(sid)
        if meta is not None:
            r.setdefault("arrival_time", meta.arrival_time)
            r.setdefault("departure_time", meta.departure_time)
            r.setdefault("parking_duration_min", float(meta.duration_minutes))
            r.setdefault("state_of_charge_at_arrival", float(meta.state_of_charge_at_arrival))
        out.append(r)
    return out


def _get_plan_array(row: dict[str, Any], candidates: list[str], length: int) -> np.ndarray:
    """Pick first existing plan array, then pad/clip to `length`."""
    arr = None
    for k in candidates:
        if k in row:
            arr = row.get(k)
            break
    a = np.asarray(arr if arr is not None else [], dtype=float).reshape(-1)
    if len(a) == length:
        return a
    if len(a) > length:
        return a[:length]
    out = np.zeros(length, dtype=float)
    out[: len(a)] = a
    return out


def build_charger_traces_dataframe(
    *,
    sessions_out: list[dict[str, Any]],
    scenario: dict,
    vehicle_curves_by_name: dict[str, VehicleChargingCurve],
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Baut charger_traces_dataframe aus sessions_out + Plänen.

    Enthält bewusst auch 0-kW Steps während 'plugged' (wie in deinem alten Code).
    Spalten (Notebook-kompatibel):
    - timestamp, charger_id, session_id, vehicle_name
    - power_kw, pv_power_kw
    - soc
    - is_plugged, is_charging
    """
    time_resolution_min = get_time_resolution_min_from_scenario(scenario)
    step_hours = float(time_resolution_min) / 60.0
    eff = float(scenario["site"].get("charger_efficiency", 1.0))

    rows: list[dict[str, Any]] = []

    for s in sessions_out:
        if str(s.get("status")) != "plugged":
            continue

        a = int(s.get("arrival_step", 0))
        d = int(s.get("departure_step", a))
        a = int(np.clip(a, 0, max(0, len(timestamps) - 1)))
        d = int(np.clip(d, 0, len(timestamps)))
        if d <= a:
            continue

        window_len = int(d - a)

        plan_site = _get_plan_array(
            s,
            ["plan_site_kwh_per_step", "plan_site_kwh", "plan_site"],
            window_len,
        )
        plan_pv = _get_plan_array(
            s,
            ["plan_pv_site_kwh_per_step", "plan_pv_kwh_per_step", "plan_pv"],
            window_len,
        )

        charger_id = s.get("charger_id")
        session_id = s.get("session_id")
        vehicle_name = s.get("vehicle_name")

        soc0 = s.get("state_of_charge_at_arrival", s.get("soc_arrival"))
        try:
            soc0_f = float(np.clip(float(soc0), 0.0, 1.0))
        except Exception:
            soc0_f = np.nan

        curve = vehicle_curves_by_name.get(str(vehicle_name)) if vehicle_name is not None else None
        battery_cap = float(max(getattr(curve, "battery_capacity_kwh", 1e-12), 1e-12))

        # soc trace: cum battery energy / cap
        batt_added = np.cumsum(plan_site) * eff
        soc_trace = soc0_f + batt_added / battery_cap if np.isfinite(soc0_f) else np.full(window_len, np.nan)
        soc_trace = np.clip(soc_trace, 0.0, 1.0)

        for i in range(window_len):
            abs_step = a + i
            if abs_step < 0 or abs_step >= len(timestamps):
                continue

            site_kwh = float(plan_site[i])
            pv_kwh = float(np.clip(float(plan_pv[i]), 0.0, site_kwh))

            power_kw = site_kwh / max(step_hours, 1e-12)
            pv_power_kw = pv_kwh / max(step_hours, 1e-12)

            rows.append(
                {
                    "timestamp": pd.Timestamp(timestamps[abs_step]),
                    "charger_id": None if charger_id is None else int(charger_id),
                    "session_id": session_id,
                    "vehicle_name": vehicle_name,
                    "power_kw": float(power_kw),
                    "pv_power_kw": float(pv_power_kw),
                    "soc": float(soc_trace[i]) if np.isfinite(soc_trace[i]) else np.nan,
                    "is_plugged": True,
                    "is_charging": bool(site_kwh > 1e-12),
                }
            )

    return pd.DataFrame(rows)


def build_power_per_charger_timeseries(
    charger_traces_dataframe: pd.DataFrame,
    *,
    charger_id: int,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Timeseries power_kw für einen Charger (Zoom)."""
    df = charger_traces_dataframe.copy()
    if len(df) == 0:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["charger_id"] == int(charger_id)]
    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end)]
    return df.sort_values("timestamp").reset_index(drop=True)


def build_soc_timeseries_by_charger(
    *,
    charger_traces_dataframe: pd.DataFrame,
    charger_ids: list[int],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> dict[int, pd.DataFrame]:
    """SoC-Timeseries je Charger (Zoom)."""
    out: dict[int, pd.DataFrame] = {}
    for cid in charger_ids:
        df = charger_traces_dataframe.copy()
        if len(df) == 0:
            out[int(cid)] = pd.DataFrame()
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[df["charger_id"] == int(cid)]
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        out[int(cid)] = df.sort_values("timestamp").reset_index(drop=True)[
            [c for c in ["timestamp", "soc", "session_id"] if c in df.columns]
        ]
    return out


def build_charger_power_heatmap_matrix(
    charger_traces_dataframe: pd.DataFrame,
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> dict[str, Any]:
    """
    Baut eine Heatmap-Matrix (charger x time).
    Return:
      - matrix: np.ndarray
      - timestamps: list[pd.Timestamp]
      - charger_ids: list[int]
    """
    df = charger_traces_dataframe.copy()
    if len(df) == 0:
        return {"matrix": np.zeros((0, 0)), "timestamps": [], "charger_ids": []}

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end)]

    if len(df) == 0:
        return {"matrix": np.zeros((0, 0)), "timestamps": [], "charger_ids": []}

    charger_ids = sorted({int(x) for x in df["charger_id"].dropna().unique().tolist()})
    df_pivot = df.pivot_table(index="timestamp", columns="charger_id", values="power_kw", aggfunc="sum").fillna(0.0)

    # full time grid (best effort)
    ts = df_pivot.index.sort_values()
    if len(ts) >= 2:
        freq = ts[1] - ts[0]
        full_ts = pd.date_range(start=ts[0], end=ts[-1], freq=freq)
        df_pivot = df_pivot.reindex(full_ts).fillna(0.0)
        timestamps_out = list(full_ts)
    else:
        timestamps_out = list(ts)

    # ensure charger columns exist in order
    for cid in charger_ids:
        if cid not in df_pivot.columns:
            df_pivot[cid] = 0.0
    df_pivot = df_pivot[charger_ids]

    matrix = df_pivot.to_numpy(dtype=float).T  # charger x time
    return {"matrix": matrix, "timestamps": timestamps_out, "charger_ids": charger_ids}


# -----------------------------------------------------------------------------
# E) EV-Leistung nach Modus (Generation/Market/Immediate)
# -----------------------------------------------------------------------------

def build_ev_power_by_mode_timeseries_dataframe(
    *,
    timeseries_dataframe: pd.DataFrame,
    sessions_out: list[dict[str, Any]],
    scenario: dict,
) -> pd.DataFrame:
    """
    Baut eine Timeseries mit:
      - ev_generation_kw
      - ev_market_kw
      - ev_immediate_kw

    Logik:
    - immediate-strategy: alles -> immediate
    - market-strategy: alles -> market
    - pv/generation-strategy: PV-Anteil -> generation, Grid-Fallback (plan_market_*) -> market
    """
    df = timeseries_dataframe.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    n = len(df)

    time_resolution_min = get_time_resolution_min_from_scenario(scenario)
    step_hours = float(time_resolution_min) / 60.0

    ev_gen_kwh = np.zeros(n, dtype=float)
    ev_market_kwh = np.zeros(n, dtype=float)
    ev_imm_kwh = np.zeros(n, dtype=float)

    strategy = get_charging_strategy_from_scenario(scenario)

    for s in sessions_out:
        if str(s.get("status")) != "plugged":
            continue

        a = int(s.get("arrival_step", 0))
        d = int(s.get("departure_step", a))
        a = int(np.clip(a, 0, n))
        d = int(np.clip(d, 0, n))
        if d <= a:
            continue

        L = int(d - a)

        plan_site = _get_plan_array(s, ["plan_site_kwh_per_step"], L)
        plan_pv = _get_plan_array(s, ["plan_pv_site_kwh_per_step", "plan_pv_kwh_per_step"], L)
        plan_market = _get_plan_array(s, ["plan_market_kwh_per_step", "plan_market_site_kwh_per_step"], L)

        if strategy == "immediate":
            ev_imm_kwh[a:d] += plan_site
        elif strategy == "market":
            # market plan == site plan
            ev_market_kwh[a:d] += plan_site if np.sum(plan_market) <= 1e-12 else plan_market
        else:
            # pv strategy
            ev_gen_kwh[a:d] += np.clip(plan_pv, 0.0, plan_site)
            # grid fallback
            if np.sum(plan_market) > 1e-12:
                ev_market_kwh[a:d] += plan_market
            else:
                # fallback: everything not PV is market
                ev_market_kwh[a:d] += np.maximum(plan_site - np.clip(plan_pv, 0.0, plan_site), 0.0)

    return pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "ev_generation_kw": ev_gen_kwh / max(step_hours, 1e-12),
            "ev_market_kw": ev_market_kwh / max(step_hours, 1e-12),
            "ev_immediate_kw": ev_imm_kwh / max(step_hours, 1e-12),
        }
    )


# -----------------------------------------------------------------------------
# F) Curve Validation Helpers
# -----------------------------------------------------------------------------

def get_most_used_vehicle_name(
    *,
    sessions_out: list[dict[str, Any]],
    charger_traces_dataframe: pd.DataFrame,
    only_plugged_sessions: bool = True,
) -> str:
    """Wählt das am häufigsten geladene Fahrzeug."""
    if charger_traces_dataframe is not None and len(charger_traces_dataframe) > 0:
        df = charger_traces_dataframe
        if only_plugged_sessions:
            # traces sind ohnehin nur plugged
            pass
        counts = df["vehicle_name"].value_counts(dropna=True)
        if len(counts) > 0:
            return str(counts.index[0])

    # fallback: sessions_out
    rows = [s for s in sessions_out if (not only_plugged_sessions) or str(s.get("status")) == "plugged"]
    counts = Counter([str(s.get("vehicle_name")) for s in rows if s.get("vehicle_name") is not None])
    return counts.most_common(1)[0][0] if counts else ""


def build_master_curve_and_actual_points_for_vehicle(
    *,
    charger_traces_dataframe: pd.DataFrame,
    scenario: dict,
    curve: VehicleChargingCurve,           
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    power_tolerance_kw: float = 1e-6,
) -> dict[str, Any]:
    """
    Vergleich Master-Ladekurve (Batterieseite) vs. Ist-Punkte (Batterieseite).

    Erwartet in charger_traces_dataframe:
    - vehicle_name
    - timestamp
    - soc
    - power_kw   (Standortseite)

    Returns (Notebook-kompatibel):
    - vehicle_name
    - master_soc
    - master_power_battery_kw
    - actual_soc
    - actual_power_batt_kw
    - violation_mask
    - number_violations
    """
    df = charger_traces_dataframe.copy()
    if len(df) == 0:
        return {
            "vehicle_name": curve.vehicle_name,
            "master_soc": np.asarray(curve.state_of_charge_fraction, dtype=float),
            "master_power_battery_kw": np.asarray(curve.power_kw, dtype=float),
            "actual_soc": np.array([]),
            "actual_power_batt_kw": np.array([]),
            "violation_mask": np.array([], dtype=bool),
            "number_violations": 0,
        }

    # Filter: Fahrzeug + optional Zeitfenster
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df.get("vehicle_name", "").astype(str) == str(curve.vehicle_name)]

    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end)]

    # Ist-Punkte
    actual_soc = pd.to_numeric(df.get("soc"), errors="coerce").to_numpy(dtype=float)
    actual_power_site_kw = pd.to_numeric(df.get("power_kw"), errors="coerce").to_numpy(dtype=float)

    # Batterie-Seite
    eff = float(scenario["site"].get("charger_efficiency", 1.0))
    eff = float(np.clip(eff, 1e-12, 1.0))
    actual_power_batt_kw = actual_power_site_kw * eff

    # Master-Kurve (Batterieseite)
    master_soc = np.asarray(curve.state_of_charge_fraction, dtype=float).reshape(-1)
    master_power_batt_kw = np.asarray(curve.power_kw, dtype=float).reshape(-1)

    # Aufräumen / Clipping
    actual_soc = np.clip(actual_soc, 0.0, 1.0)
    actual_power_batt_kw = np.maximum(actual_power_batt_kw, 0.0)

    valid_mask = np.isfinite(actual_soc) & np.isfinite(actual_power_batt_kw)
    actual_soc_v = actual_soc[valid_mask]
    actual_p_v = actual_power_batt_kw[valid_mask]

    if actual_soc_v.size == 0 or master_soc.size < 2:
        return {
            "vehicle_name": curve.vehicle_name,
            "master_soc": master_soc,
            "master_power_battery_kw": master_power_batt_kw,
            "actual_soc": actual_soc_v,
            "actual_power_batt_kw": actual_p_v,
            "violation_mask": np.array([], dtype=bool),
            "number_violations": 0,
        }

    # erlaubte Leistung aus Masterkurve interpolieren
    allowed_kw = np.interp(actual_soc_v, master_soc, master_power_batt_kw)
    violation_mask = actual_p_v > (allowed_kw + float(power_tolerance_kw))
    number_violations = int(np.count_nonzero(violation_mask))

    return {
        "vehicle_name": curve.vehicle_name,
        "master_soc": master_soc,
        "master_power_battery_kw": master_power_batt_kw,
        "actual_soc": actual_soc_v,
        "actual_power_batt_kw": actual_p_v,
        "violation_mask": violation_mask,
        "number_violations": number_violations,
    }
