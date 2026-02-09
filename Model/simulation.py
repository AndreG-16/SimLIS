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

    Die Simulation arbeitet konsequent tz-aware ("Europe/Berlin").
    `simulation_horizon_days` wird als Kalendertage interpretiert, damit DST-Tage
    automatisch 23h bzw. 25h haben.

    Wichtiger Hinweis
    -----------------
    Wenn `start_datetime` tz-naiv ist und in eine *ambige* lokale Zeit fällt
    (DST-Ende, typischerweise 02:xx), wird mit `ambiguous="raise"` ein Fehler
    geworfen. In dem Fall `start_datetime` im YAML auf eine eindeutige Zeit legen
    (z. B. 00:00 oder 03:00) oder explizit einen Offset angeben.

    Ausnahmen
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
    """Lokalisiert einen tz-naiven Wandzeit-Index DST-robust in eine Zeitzone.
    
        Diese Hilfsfunktion geht davon aus, dass die Zeitstempel aus der CSV lokale Wandzeit
        ohne Zeitzoneninformation sind (z. B. deutsche Lokalzeit).
    
        DST-Behandlung
        --------------
        - Sommerzeitbeginn (nicht-existente Zeiten): nach vorn schieben.
        - Sommerzeitende (ambige Zeiten): bei doppelten Zeitstempeln wird das erste
          Vorkommen als Sommerzeit und das zweite als Standardzeit interpretiert.
    
        Parameter
        ---------
        datetime_index:
            tz-naiver DatetimeIndex (lokale Wandzeit).
        timezone:
            IANA-Zeitzone (z. B. "Europe/Berlin").
    
        Rückgabe
        --------
        pd.DatetimeIndex
            tz-aware Index in der gewünschten Zeitzone.
    
    """
    idx = pd.DatetimeIndex(idx)

    if idx.tz is not None:
        return idx.tz_convert(tz)

    if idx.has_duplicates:
        duplicate_mask = idx.duplicated(keep=False)

        # Count occurrences per timestamp: 0 for first, 1 for second, ...
        occurrence_index = pd.Series(np.arange(len(idx))).groupby(idx).cumcount().to_numpy()

        # For ambiguous duplicated wall times:
        # first occurrence -> DST=True, second -> DST=False
        ambiguous = np.where(duplicate_mask, occurrence_index == 0, False)

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

    Attribute
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
    Liest eine YAML-Szenario-Datei ein und validiert Pflichtfelder.

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

    Parameter
    ----------
    scenario_path:
        Pfad zur YAML-Datei.

    Rückgabe
    -------
    dict
        Szenario als Dictionary.

    Ausnahmen
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

    Parameter
    ----------
    scenario:
        Szenario-Dictionary, wie es aus der YAML eingelesen wurde. Das Dictionary wird
        in-place aktualisiert.
    scenario_path:
        Pfad zur YAML-Datei, die als Referenz für relative Pfade dient.

    Rückgabe
    -------
    dict
        Das aktualisierte Szenario-Dictionary mit absoluten Pfaden.

    Ausnahmen
    ------
    KeyError
        Wenn eines der erwarteten Pfadfelder im Szenario fehlt.
    ValueError
        Wenn ein Pfad leer ist oder nur aus Whitespaces besteht.
    """
    base_directory = Path(scenario_path).resolve().parent

    def to_absolute(path_value: Any) -> str:
        """Wandelt einen Pfad in einen absoluten Pfad um.
        
            Relative Pfade werden relativ zum YAML-Verzeichnis aufgelöst.
        """
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
    """
    Liest Grundlast-, PV- und Marktpreis-Zeitreihen aus CSV und richtet sie am Simulationsraster aus.

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

    Rückgabe
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        (base_load_kwh_per_step, pv_generation_kwh_per_step, market_price_eur_per_mwh)

    Ausnahmen
    ------
    ValueError
        - wenn `timestamps` tz-naiv ist (harte Prüfung)
        - wenn CSV < 4 Spalten hat
        - wenn Numerik-Parsen fehlschlägt
    """
    simulation_time_index = pd.DatetimeIndex(timestamps)

    # Harte Prüfung: konsequent tz-aware Simulation.
    if simulation_time_index.tz is None:
        raise ValueError(
            "Die Simulation ist nicht tz-aware: `timestamps.tz` ist None. "
            "Bitte den Simulations-Zeitindex z. B. mit 'Europe/Berlin' lokalisieren."
        )

    if len(simulation_time_index) < 2:
        raise ValueError("`timestamps` muss mindestens zwei Einträge enthalten.")

    # CSV einlesen und Grundstruktur prüfen.
    dataframe = pd.read_csv(csv_path, sep=separator, decimal=decimal)
    if dataframe.shape[1] < 4:
        raise ValueError(
            "CSV hat zu wenige Spalten. Erwartet: datetime | base load | PV | market price."
        )

    dt_col = dataframe.columns[0]
    base_col = dataframe.columns[1]
    pv_col = dataframe.columns[2]
    price_col = dataframe.columns[3]

    # Datums-/Zeitspalte parsen und als Index setzen.
    parsed_dt = pd.to_datetime(
        dataframe[dt_col].astype(str).str.strip(),
        format=datetime_format,
        errors="raise",
    )

    dataframe = dataframe.copy()
    dataframe[dt_col] = parsed_dt
    dataframe = dataframe.sort_values(dt_col).set_index(dt_col)

    # CSV-Index in Simulations-TZ lokalisieren (DST-robust).
    target_tz = str(timezone) if timezone else str(simulation_time_index.tz)
    dataframe.index = _localize_wall_time_index(dataframe.index, target_tz)

    # Numerische Spalten parsen (und Profile auf >= 0 clampen).
    dataframe[base_col] = pd.to_numeric(dataframe[base_col], errors="raise").astype(float).clip(lower=0.0)
    dataframe[pv_col] = pd.to_numeric(dataframe[pv_col], errors="raise").astype(float).clip(lower=0.0)
    dataframe[price_col] = pd.to_numeric(dataframe[price_col], errors="raise").astype(float)

    dataframe = dataframe[[base_col, pv_col, price_col]].sort_index()

    # Zeitschrittgrößen bestimmen.
    sim_step_min = int(round((simulation_time_index[1] - simulation_time_index[0]).total_seconds() / 60.0))
    sim_step_hours = float(sim_step_min) / 60.0
    input_step_hours = float(input_time_resolution_min) / 60.0

    # Profile interpretieren / skalieren (auf Input-Gitter).
    base_profile = dataframe[base_col]
    pv_profile = dataframe[pv_col]
    price_series = dataframe[price_col]

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
        base_kw_aligned = base_kw.reindex(simulation_time_index)
        pv_kw_aligned = pv_kw.reindex(simulation_time_index)
        price_aligned = price_series.reindex(simulation_time_index, method="ffill")

    elif sim_step_min < int(input_time_resolution_min):
        # Simulation feiner als CSV: stückweise konstante Leistung bis zur nächsten Messung.
        base_kw_aligned = base_kw.reindex(simulation_time_index, method="ffill")
        pv_kw_aligned = pv_kw.reindex(simulation_time_index, method="ffill")
        price_aligned = price_series.reindex(simulation_time_index, method="ffill")

    else:
        # Simulation gröber als CSV: mittlere Leistung über das größere Intervall.
        rule = f"{sim_step_min}min"
        base_kw_aligned = base_kw.resample(rule).mean().reindex(simulation_time_index)
        pv_kw_aligned = pv_kw.resample(rule).mean().reindex(simulation_time_index)
        price_aligned = price_series.resample(rule).mean().reindex(simulation_time_index, method="ffill")

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
    Liest fahrzeugspezifische Ladekurven aus einer CSV (ein Fahrzeug je Spalte).
    
        Erwartetes CSV-Format
        ---------------------
        In Spalte 0 stehen feste Labels in den Zeilen 0..3:
        0) Hersteller
        1) Modell
        2) Fahrzeugklasse
        3) max. Kapazität
    
        Ab Zeile 4 folgt die SoC-Stützstellen-Spalte (0..100 in 0,5er Schritten oder 0..1),
        und in den Fahrzeugspalten die maximale Ladeleistung in kW (Batterieseite) je SoC.
    
        Parameter
        ---------
        vehicle_curve_csv_path:
            Pfad zur CSV-Datei.
    
        Rückgabe
        --------
        dict[str, VehicleChargingCurve]
            Zuordnung von eindeutigem Fahrzeugnamen auf Ladekurve und Batteriekapazität.
    
        Ausnahmen
        ---------
        ValueError
            Wenn das CSV-Format nicht passt oder ungültige Werte enthält.
    """
    vehicle_curve_table = pd.read_csv(
        vehicle_curve_csv_path,
        sep=None,
        engine="python",
        header=None,
        dtype=str,
        encoding="utf-8-sig",
    )

    if vehicle_curve_table.shape[0] < 6 or vehicle_curve_table.shape[1] < 2:
        raise ValueError(
            "Vehicleloadprofile-CSV hat nicht das erwartete Format "
            "(benötigt mindestens 6 Zeilen und 2 Spalten)."
        )

    expected_labels = ["Hersteller", "Modell", "Fahrzeugklasse", "max. Kapazität"]

    def _clean_label(x: object) -> str:
        """Bereinigt ein Label aus der CSV (Whitespace/BOM entfernen).
        """
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

    Attribute
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

    Parameter
    ----------
    simulation_day_start:
        Startzeitpunkt des Tages (Datumsteil ist relevant).
    holiday_dates:
        Menge von Feiertagen als datetime.date.

    Rückgabe
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

    Rückgabe
    -------
    float
        Gesampelter Wert.

    Ausnahmen
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
        """Liest einen Pflichtparameter aus einer Verteilungsspezifikation.
        
            Wirft einen Fehler, wenn der Parameter fehlt.
        
        """
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

    Parameter
    ----------
    vehicle_curves_by_name:
        Zuordnung {vehicle_name: VehicleChargingCurve}.
    fleet_mix:
        Klassen-Gewichte, z. B. {"PKW": 0.98, "Transporter": 0.02}. Kann leer sein.
    random_generator:
        Numpy RNG.
    vehicle_type:
        Optionaler Override: String oder Liste[String], max. 5 Fahrzeugnamen.

    Rückgabe
    -------
    Tuple[str, str]
        (vehicle_name, vehicle_class)

    Ausnahmen
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

    Rückgabe
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

    Hinweise
    --------
    - `arrival_step` und `departure_step` sind *absolute* Indizes im Simulationsraster.
    - `departure_step` ist Ende-exklusiv.
    - Ankunft wird auf den nächstgelegenen Simulationszeitpunkt gerastert (nearest).
    - Abfahrt wird als `arrival_wall + parkdauer` berechnet und dann per **ceil** gerastert
    - Wenn `allow_cross_day_charging=False`, wird die Session am Tagesende abgeschnitten.

    Rückgabe
    -------
    list[SampledSession]
        Liste gesampelter Sessions (nach arrival_time sortiert).
    """
    simulation_time_index = pd.DatetimeIndex(timestamps)

    time_resolution_min = int(scenario["time_resolution_min"])
    step_minutes = float(time_resolution_min)
                                     
    day_start_ts = pd.Timestamp(simulation_day_start)
    day_key = day_start_ts.normalize()

    day_mask = simulation_time_index.normalize() == day_key
    if not np.any(day_mask):
        return []

    day_timestamps = simulation_time_index[day_mask]
    steps_per_day = int(len(day_timestamps))
    if steps_per_day <= 0:
        return []

    day_start_abs_step = int(simulation_time_index.get_loc(day_timestamps[0]))
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
        """Rundet einen Zeitpunkt auf den nächsten Simulationsschritt auf.
        
            Rückgabe ist der erste Index mit `timestamps[index] >= target`.
        
        """
        idx = int(simulation_time_index.searchsorted(target, side="left"))
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

        arrival_abs_step = int(simulation_time_index.get_indexer([arrival_wall], method="nearest")[0])

        # In den Tagesbereich clampen, damit die Session diesem Kalendertag zugeordnet bleibt
        arrival_abs_step = int(np.clip(arrival_abs_step, day_start_abs_step, day_end_excl - 1))
        arrival_time = pd.to_datetime(simulation_time_index[arrival_abs_step]).to_pydatetime()

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
        if departure_abs_step < len(simulation_time_index):
            departure_time = pd.to_datetime(simulation_time_index[departure_abs_step]).to_pydatetime()
        else:
            # Falls ganz am Ende des Horizonts: virtueller Endpunkt ein Schritt danach
            departure_time = (
                pd.to_datetime(simulation_time_index[-1]) + pd.Timedelta(minutes=time_resolution_min)
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

    Parameter
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

    Rückgabe
    -------
    Tuple[float, float]
        ``(allocated_site_kwh, pv_share_kwh)`` mit der allokierten Energie auf Standortseite und
        dem PV-Anteil daran, jeweils in kWh/Step.

    Ausnahmen
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
    state_of_charge = float(np.clip(state_of_charge_fraction, 0.0, 1.0))
    battery_power_kw = float(np.interp(state_of_charge, curve.state_of_charge_fraction, curve.power_kw))
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

    Rückgabe
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
    state_of_charge = float(np.clip(state_of_charge_at_arrival, 0.0, 1.0))

    battery_capacity_kwh = float(max(curve.battery_capacity_kwh, 1e-12))
    eff = float(charger_efficiency)

    charged_site_kwh = 0.0
    charged_pv_site_kwh = 0.0

    for abs_step in range(start, end_excl):
        if remaining_site_kwh <= 1e-12 or state_of_charge >= 1.0 - 1e-12:
            break

        alloc_site_kwh, alloc_pv_kwh = available_energy_for_session_step(
            step_index=int(abs_step),
            scenario=scenario,
            curve=curve,
            state_of_charge_fraction=float(state_of_charge),
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
        state_of_charge = float(min(1.0, state_of_charge + battery_energy_kwh / battery_capacity_kwh))

    return {
        "plan_site_kwh_per_step": plan_site_kwh_per_step,
        "plan_pv_site_kwh_per_step": plan_pv_site_kwh_per_step,
        "plan_market_kwh_per_step": np.zeros_like(plan_site_kwh_per_step),
        "charged_site_kwh": float(charged_site_kwh),
        "charged_pv_site_kwh": float(charged_pv_site_kwh),
        "remaining_site_kwh": float(max(remaining_site_kwh, 0.0)),
        "final_soc": float(state_of_charge),
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
    Vergibt Energie an eine Session nach der Market-Strategie (günstige Slots nutzen).

    Vorgehen
    --------
    1) Alle Zeitschritte der Session-Standzeit werden bestimmt.
    2) Diese Schritte werden nach Marktpreis aufsteigend sortiert.
    3) Es werden die k günstigsten Schritte ausgewählt, bis eine optimistische Kapazitätsschätzung
       (ohne SoC-Limit) theoretisch ausreichen würde. Falls das nicht möglich ist, werden alle
       Schritte zugelassen.
    4) Die tatsächliche Zuweisung erfolgt chronologisch und ausschließlich in den zugelassenen
       Schritten über ``available_energy_for_session_step(...)``.

    Wenn die Session innerhalb ihrer Standzeit nicht genug Energie bekommt (weil keine
    Standortenergie verfügbar ist), bleibt Restbedarf übrig.

    Rückgabe
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
    state_of_charge = float(np.clip(state_of_charge_at_arrival, 0.0, 1.0))

    if window_len == 0 or remaining_site_kwh <= 1e-12:
        return {
            "plan_site_kwh_per_step": plan_site_kwh_per_step,
            "plan_pv_site_kwh_per_step": plan_pv_site_kwh_per_step,
            "plan_market_kwh_per_step": plan_market_kwh_per_step,
            "charged_site_kwh": 0.0,
            "charged_pv_site_kwh": 0.0,
            "remaining_site_kwh": float(max(remaining_site_kwh, 0.0)),
            "final_soc": float(state_of_charge),
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
        if remaining_site_kwh <= 1e-12 or state_of_charge >= 1.0 - 1e-12:
            break
        if not bool(allowed[local_i]):
            continue

        alloc_site_kwh, alloc_pv_kwh = available_energy_for_session_step(
            step_index=int(abs_step),
            scenario=scenario,
            curve=curve,
            state_of_charge_fraction=float(state_of_charge),
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
        state_of_charge = float(min(1.0, state_of_charge + battery_energy_kwh / battery_capacity_kwh))

    return {
        "plan_site_kwh_per_step": plan_site_kwh_per_step,
        "plan_pv_site_kwh_per_step": plan_pv_site_kwh_per_step,
        "plan_market_kwh_per_step": plan_market_kwh_per_step,
        "charged_site_kwh": float(charged_site_kwh),
        "charged_pv_site_kwh": float(charged_pv_site_kwh),
        "remaining_site_kwh": float(max(remaining_site_kwh, 0.0)),
        "final_soc": float(state_of_charge),
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
    Plant und allokiert Ladeenergie nach dem Prinzip „PV zuerst, Netz nur als günstiger Fallback“.

    Grundidee
    ---------
    - Solange innerhalb der Standzeit PV-Überschuss (nach Grundlast) verfügbar ist, wird damit geladen.
    - Reicht PV voraussichtlich nicht aus, wird zusätzlich Netzstrom in den günstigsten Preis-Zeitschritten
    (Marktpreis) zugelassen.


    1) PV-Ladung (pv_only):
    Es wird versucht, ausschließlich mit PV-Überschuss zu laden.
    Dabei werden berücksichtigt:
    - PV nach Abzug der Grundlast,
    - bereits reservierte PV-EV-Anteile,
    - Ladepunktlimit,
    - Fahrzeug-Ladekurve (SoC-abhängige Leistung),
    - verbleibender Energiebedarf.

    2) Netz-Fallback (grid_only) nur in ausgewählten Schritten:
    Wenn vorher geschätzt wurde, dass PV insgesamt nicht reicht, werden einige Zeitschritte als
    „Netz erlaubt“ markiert. Diese Schritte sind die günstigsten (niedrigster Marktpreis) innerhalb
    der Standzeit, bis die geschätzte fehlende Energie gedeckt wäre.
    In genau diesen Schritten wird nach der PV-Ladung optional zusätzlich aus dem Netz geladen
    (ebenfalls begrenzt durch Netzlimit, Reservierungen, Ladepunktlimit, Ladekurve und Restbedarf).

    Wichtige Eigenschaften
    ----------------------
    - Chronologisch: Obwohl Netz-Slots nach Preis ausgewählt werden, wird die tatsächliche Ladung
    immer in Zeitreihen-Reihenfolge durchgeführt, damit SoC und Ladekurve konsistent bleiben.

    Rückgabe
    --------
    Gibt Zeitreihen (pro Session-Fenster) und Summen zurück:
    - plan_site_kwh_per_step: gesamte zugewiesene Energie (PV + Netz) je Schritt
    - plan_pv_site_kwh_per_step: PV-Anteil je Schritt
    - plan_market_kwh_per_step: Netz-Anteil je Schritt (Fallback)
    sowie charged_* Summen, remaining_site_kwh und final_soc.
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
    state_of_charge = float(np.clip(state_of_charge_at_arrival, 0.0, 1.0))

    if window_len == 0 or remaining_site_kwh <= 1e-12:
        return {
            "plan_site_kwh_per_step": plan_site_kwh_per_step,
            "plan_pv_site_kwh_per_step": plan_pv_site_kwh_per_step,
            "plan_market_kwh_per_step": plan_market_kwh_per_step,
            "charged_site_kwh": 0.0,
            "charged_pv_site_kwh": 0.0,
            "charged_market_kwh": 0.0,
            "remaining_site_kwh": float(remaining_site_kwh),
            "final_soc": float(state_of_charge),
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
        if remaining_site_kwh <= 1e-12 or state_of_charge >= 1.0 - 1e-12:
            break

        # (1) PV-only
        pv_alloc_kwh, pv_share_kwh = available_energy_for_session_step(
            step_index=int(abs_step),
            scenario=scenario,
            curve=curve,
            state_of_charge_fraction=float(state_of_charge),
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
            state_of_charge = float(min(1.0, state_of_charge + battery_energy_kwh / battery_capacity_kwh))

            if remaining_site_kwh <= 1e-12 or state_of_charge >= 1.0 - 1e-12:
                break

        # (2) Grid-only Fallback in ausgewählten Slots
        if not bool(preferred_grid[local_i]):
            continue

        grid_alloc_kwh, _grid_pv_share = available_energy_for_session_step(
            step_index=int(abs_step),
            scenario=scenario,
            curve=curve,
            state_of_charge_fraction=float(state_of_charge),
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
        state_of_charge = float(min(1.0, state_of_charge + battery_energy_kwh / battery_capacity_kwh))

    return {
        "plan_site_kwh_per_step": plan_site_kwh_per_step,
        "plan_pv_site_kwh_per_step": plan_pv_site_kwh_per_step,
        "plan_market_kwh_per_step": plan_market_kwh_per_step,
        "charged_site_kwh": float(charged_site_kwh),
        "charged_pv_site_kwh": float(charged_pv_site_kwh),
        "charged_market_kwh": float(charged_market_kwh),
        "remaining_site_kwh": float(max(remaining_site_kwh, 0.0)),
        "final_soc": float(state_of_charge),
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

    def duration_steps(session: SampledSession) -> int:
        """Berechnet die Sessiondauer in Simulationsschritten.
        """
        return int(max(0, int(session.departure_step) - int(session.arrival_step)))

    if strategy == "immediate":
        return sorted(
            sessions,
            key=lambda session: (int(session.arrival_step), int(session.departure_step), str(session.session_id)),
        )

    return sorted(
        sessions,
        key=lambda session: (duration_steps(session), int(session.arrival_step), str(session.session_id)),
    )


def required_site_energy_for_session(
    *,
    scenario: dict,
    curve: VehicleChargingCurve,
    soc_at_arrival: float,
) -> float:
    """
    Berechnet die benötigte Ladeenergie auf Standortseite bis zum Ziel-SoC.

    Die Zielgröße ist die Energie, die der Standort (über den Ladepunkt) liefern muss,
    damit die Fahrzeugbatterie von ``soc_at_arrival`` auf ``scenario["vehicles"]["soc_target"]``
    geladen werden kann. Dabei wird der Ladepunktwirkungsgrad berücksichtigt.

    Formel
    ------
    needed_battery_kwh = max(0, (soc_target - soc_at_arrival) * battery_capacity_kwh)
    needed_site_kwh    = needed_battery_kwh / charger_efficiency

    Parameter
    ---------
    scenario:
        Szenario-Konfiguration. Verwendet:
        - ``scenario["vehicles"]["soc_target"]`` als Ziel-SoC (0..1),
        - ``scenario["site"]["charger_efficiency"]`` als Wirkungsgrad (0..1].
    curve:
        Fahrzeugspezifische Ladekurve mit ``battery_capacity_kwh`` (Batteriekapazität).
    soc_at_arrival:
        Ladezustand bei Ankunft als Anteil (0..1).

    Rückgabe
    --------
    float
        Benötigte Energie auf Standortseite in kWh (>= 0).
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
    Plant für einen Kalendertag die Ladeenergie-Zuweisung für alle Ladesessions.

    Die Funktion wählt die Reihenfolge der Planung abhängig von
    ``scenario["charging_strategy"]`` und ruft anschließend pro Session die
    passende Strategie-Funktion auf:

    - ``"immediate"``: Ladeplanung nach First-Come-First-Served (Ankunft zuerst)
    - ``"market"``: Nutzung preisgünstiger Zeitschritte (Marktpreis-geführt)
    - ``"pv"`` / ``"generation"``: PV zuerst, Netz als preisgünstiger Fallback

    Die Strategie-Funktionen schreiben ihre Zuweisungen in die globalen
    Reservierungs-Arrays (``reserved_*``), sodass nachfolgende Sessions die bereits
    belegte Standortkapazität berücksichtigen.

    Parameter
    ----------
    sessions:
        Liste der Sessions, die an diesem Tag geplant werden sollen.
    scenario:
        Szenario-Konfiguration. Verwendet u. a. ``scenario["charging_strategy"]``
        sowie Parameter unter ``scenario["site"]`` und ``scenario["vehicles"]``.
    vehicle_curves_by_name:
        Zuordnung von ``vehicle_name`` zu ``VehicleChargingCurve``.
    pv_generation_kwh_per_step:
        PV-Erzeugung pro Simulationsschritt [kWh/Schritt].
    base_load_kwh_per_step:
        Grundlast pro Simulationsschritt [kWh/Schritt].
    reserved_total_ev_energy_kwh_per_step:
        Bereits reservierte EV-Ladeenergie (gesamt) pro Schritt [kWh/Schritt].
        Wird in-place um die geplanten Zuweisungen dieser Funktion erhöht.
    reserved_pv_ev_energy_kwh_per_step:
        Bereits reservierter PV-Anteil der EV-Ladeenergie pro Schritt [kWh/Schritt].
        Wird in-place um die geplanten PV-Zuweisungen dieser Funktion erhöht.
    market_price_eur_per_mwh:
        Marktpreis-Zeitreihe [€/MWh]. Erforderlich für ``"market"`` sowie als
        Fallback-Ranking für ``"pv"`` / ``"generation"``.

    Rückgabe
    -------
    list[dict[str, Any]]
        Liste von Ergebnis-Dictionaries, je Session ein Eintrag. Enthält u. a.
        Plan-Zeitreihen (z. B. ``plan_site_kwh_per_step``) sowie Summenwerte
        (z. B. ``charged_site_kwh``, ``remaining_site_kwh``) und Metadaten
        (z. B. ``session_id``, ``arrival_time``).

    Ausnahme
    ------
    ValueError
        Wenn für eine Session keine Ladekurve vorhanden ist, eine unbekannte
        Strategie konfiguriert ist oder ``market_price_eur_per_mwh`` für eine
        preisbasierte Strategie fehlt.
    """
    strategy = str(scenario["charging_strategy"]).strip().lower()
    ordered = order_sessions_for_planning(sessions, strategy)

    results: list[dict[str, Any]] = []

    for session in ordered:
        curve = vehicle_curves_by_name.get(session.vehicle_name)
        if curve is None:
            raise ValueError(f"Keine Ladekurve für vehicle_name='{session.vehicle_name}' gefunden.")

        required_site_energy_kwh = required_site_energy_for_session(
            scenario=scenario,
            curve=curve,
            soc_at_arrival=float(session.state_of_charge_at_arrival),
        )

        if strategy == "immediate":
            session_result = charging_strategy_immediate(
                scenario=scenario,
                session_arrival_step=int(session.arrival_step),
                session_departure_step=int(session.departure_step),
                required_site_energy_kwh=float(required_site_energy_kwh),
                pv_generation_kwh_per_step=pv_generation_kwh_per_step,
                base_load_kwh_per_step=base_load_kwh_per_step,
                reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
                reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
                curve=curve,
                state_of_charge_at_arrival=float(session.state_of_charge_at_arrival),
            )

        elif strategy == "market":
            if market_price_eur_per_mwh is None:
                raise ValueError("charging_strategy='market' benötigt market_price_eur_per_mwh.")

            session_result = charging_strategy_market(
                scenario=scenario,
                session_arrival_step=int(session.arrival_step),
                session_departure_step=int(session.departure_step),
                required_site_energy_kwh=float(required_site_energy_kwh),
                market_price_eur_per_mwh=market_price_eur_per_mwh,
                pv_generation_kwh_per_step=pv_generation_kwh_per_step,
                base_load_kwh_per_step=base_load_kwh_per_step,
                reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
                reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
                curve=curve,
                state_of_charge_at_arrival=float(session.state_of_charge_at_arrival),
            )

        elif strategy in ("pv", "generation"):
            if market_price_eur_per_mwh is None:
                raise ValueError(
                    "charging_strategy='pv' benötigt market_price_eur_per_mwh (Fallback-Ranking)."
                )

            session_result = charging_strategy_pv(
                scenario=scenario,
                session_arrival_step=int(session.arrival_step),
                session_departure_step=int(session.departure_step),
                required_site_energy_kwh=float(required_site_energy_kwh),
                market_price_eur_per_mwh=market_price_eur_per_mwh,
                pv_generation_kwh_per_step=pv_generation_kwh_per_step,
                base_load_kwh_per_step=base_load_kwh_per_step,
                reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
                reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
                curve=curve,
                state_of_charge_at_arrival=float(session.state_of_charge_at_arrival),
            )

        else:
            raise ValueError(f"Unbekannte charging_strategy='{strategy}'")

        # Meta-Infos ergänzen (fürs Notebook)
        session_result["session_id"] = session.session_id
        session_result["vehicle_name"] = session.vehicle_name
        session_result["vehicle_class"] = session.vehicle_class
        session_result["arrival_step"] = int(session.arrival_step)
        session_result["departure_step"] = int(session.departure_step)
        session_result["required_site_kwh"] = float(required_site_energy_kwh)
        session_result["arrival_time"] = session.arrival_time
        session_result["departure_time"] = session.departure_time
        session_result["parking_duration_min"] = float(session.duration_minutes)
        session_result["state_of_charge_at_arrival"] = float(session.state_of_charge_at_arrival)

        results.append(session_result)

    return results


# =============================================================================
# 4) Physik + Simulation: FCFS, Ladepunkt-Zuordnung, Reservierungs-Lastgang
# =============================================================================

def find_free_charger_fcfs(
    charger_occupied_until_step: list[int],
    arrival_step: int,
) -> int | None:
    """
    Ermittelt nach dem First-Come-First-Served-Prinzip (FCFS) den ersten freien Ladepunkt.

    Parameter
    ---------
    charger_occupied_until_step:
        Liste mit Länge = Anzahl der Ladepunkte. Jeder Eintrag enthält den Simulationsschritt,
        bis zu dem der jeweilige Ladepunkt belegt ist (Ende-exklusiv oder inklusiv – entscheidend
        ist die verwendete Vergleichslogik ``<= arrival_step``).
    arrival_step:
        Simulationsschritt der Ankunft der neuen Session.

    Rückgabe
    --------
    int | None
        ID des ersten freien Ladepunkts (0-basiert). Falls kein Ladepunkt frei ist, ``None``.
    """
    for charger_id, occupied_until in enumerate(charger_occupied_until_step):
        if int(occupied_until) <= int(arrival_step):
            return int(charger_id)
    return None


def _group_sessions_by_arrival_day(
    sessions: list[SampledSession],
    timestamps: pd.DatetimeIndex,
) -> dict[pd.Timestamp, list[SampledSession]]:
    """
    Gruppiert Sessions nach dem Kalendertag ihrer Ankunft.

    Diese Gruppierung ist für die Simulation notwendig, weil die Ladeplanung in
    ``simulate_site_fcfs_with_planning(...)`` tageweise erfolgt: Für jeden Ankunftstag
    werden alle an diesem Tag „plugged“ Sessions gemeinsam an ``plan_charging_for_day(...)``
    übergeben (inkl. Sortierung gemäß Strategie und gemeinsamer Nutzung der Reservierungen).

    Der Kalendertag wird aus dem Simulationszeitstempel am ``arrival_step`` abgeleitet und
    mit ``normalize()`` auf 00:00 Uhr des jeweiligen Tages gesetzt. Dadurch ist die
    Gruppierung zeitzonenrobust, solange ``timestamps`` tz-aware ist.

    Parameter
    ---------
    sessions:
        Liste von Sessions, die gruppiert werden sollen.
    timestamps:
        Simulations-Zeitindex. Der Eintrag an Position ``arrival_step`` bestimmt den
        zugehörigen Kalendertag der Session.

    Rückgabe
    --------
    dict[pd.Timestamp, list[SampledSession]]
        Dictionary mit Tages-Schlüssel (Timestamp auf Tagesstart) und den zugehörigen Sessions.
    """
    by_day: dict[pd.Timestamp, list[SampledSession]] = {}
    for session in sessions:
        day_key = pd.Timestamp(timestamps[int(session.arrival_step)]).normalize()
        by_day.setdefault(day_key, []).append(session)
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
    Erstellt eine Debug-Energiebilanz pro Simulationsschritt als DataFrame.

    Die Bilanz zerlegt PV-Erzeugung und Netzbezug in nachvollziehbare Flüsse:

    - PV → Grundlast: PV deckt zuerst die Grundlast bis zur Höhe der Grundlast.
    - PV → EV: verbleibender PV-Überschuss wird den EV-Ladevorgängen zugeordnet
    (begrenzt durch den in ``reserved_pv_ev_energy_kwh_per_step`` getrackten PV-Anteil).
    - Netz → Grundlast: Rest-Grundlast wird aus dem Netz gedeckt (begrenzt durch das Netzlimit).
    - Netz → EV: verbleibender Netz-Spielraum nach Grundlast wird den EV-Ladevorgängen zugeordnet.

    Diese Funktion dient ausschließlich der Plausibilisierung/Analyse (Debugging) und hat
    keinen Einfluss auf die eigentliche Ladeplanung.

    Parameter
    ---------
    timestamps:
        Simulations-Zeitindex (Länge = Anzahl Simulationsschritte).
    scenario:
        Szenario-Konfiguration. Verwendet:
        - ``scenario["time_resolution_min"]`` zur Umrechnung kW↔kWh,
        - ``scenario["site"]["grid_limit_p_avb_kw"]`` als Netzanschlussgrenze [kW].
    pv_generation_kwh_per_step:
        PV-Erzeugung pro Schritt [kWh/Schritt].
    base_load_kwh_per_step:
        Grundlast pro Schritt [kWh/Schritt].
    reserved_total_ev_energy_kwh_per_step:
        Gesamte EV-Ladeenergie pro Schritt (PV + Netz) [kWh/Schritt].
    reserved_pv_ev_energy_kwh_per_step:
        Getrackter PV-Anteil der EV-Ladeenergie pro Schritt [kWh/Schritt].

    Rückgabe
    --------
    pd.DataFrame
        DataFrame mit Zeitstempel und Bilanzspalten (alle in kWh/Schritt), u. a.:
        - ``pv_to_base_kwh_per_step``, ``pv_to_ev_kwh_per_step``
        - ``grid_to_base_kwh_per_step``, ``grid_to_ev_kwh_per_step``
        - ``grid_limit_kwh_per_step`` (konstant pro Schritt)
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

    dataframe = pd.DataFrame(
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
    return dataframe


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

    Rückgabe
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
    sessions_sorted = sorted(sessions, key=lambda session: (int(session.arrival_step), str(session.session_id)))

    charger_id_by_session_id: dict[str, int] = {}
    plugged: list[SampledSession] = []
    sessions_out: list[dict[str, Any]] = []

    for session in sessions_sorted:
        a = int(np.clip(int(session.arrival_step), 0, n_total - 1))
        d = int(np.clip(int(session.departure_step), 0, n_total))
        if d <= a:
            d = min(a + 1, n_total)

        session.arrival_step = a
        session.departure_step = d

        charger_id = find_free_charger_fcfs(charger_occupied_until, a)
        if charger_id is None:
            sessions_out.append(
                {
                    "session_id": session.session_id,
                    "vehicle_name": session.vehicle_name,
                    "vehicle_class": session.vehicle_class,
                    "arrival_step": a,
                    "departure_step": d,
                    "arrival_time": session.arrival_time,             
                    "departure_time": session.departure_time,         
                    "parking_duration_min": float(session.duration_minutes),  
                    "state_of_charge_at_arrival": float(session.state_of_charge_at_arrival), 
                    "status": "drive_off",
                    "charger_id": None,
                    "charged_site_kwh": 0.0,
                    "charged_pv_site_kwh": 0.0,
                    "remaining_site_kwh": np.nan,
                    "final_soc": float(np.clip(session.state_of_charge_at_arrival, 0.0, 1.0)),
                    "plan_site_kwh_per_step": np.zeros(0, dtype=float),
                    "plan_pv_site_kwh_per_step": np.zeros(0, dtype=float),
                    "plan_market_kwh_per_step": np.zeros(0, dtype=float),
                }
            )
            continue

        charger_occupied_until[charger_id] = d
        charger_id_by_session_id[str(session.session_id)] = charger_id
        plugged.append(session)

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
            session_identifier = str(r["session_id"])
            r["status"] = "plugged"
            r["charger_id"] = int(charger_id_by_session_id[session_identifier])
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

def get_time_resolution_min_from_scenario(scenario: dict) -> int:
    """
    Gibt die Zeitauflösung der Simulation in Minuten zurück.
    
    Parameter
    ---------
    scenario:
        Szenario-Dictionary.
    
    Rückgabe
    --------
    int
        Zeitauflösung in Minuten (Default: 15).
    """
    return int(scenario.get("time_resolution_min", 15))


def get_holiday_dates_from_scenario(scenario: dict, timestamps: pd.DatetimeIndex) -> set[date]:
    """
    Ermittelt Feiertage für den Simulationszeitraum.
    
    Es werden entweder explizit konfigurierte Datumswerte aus `scenario["holidays"]["dates"]`
    verwendet oder – falls vorhanden – das Paket `holidays` zur Berechnung genutzt.
    
    Parameter
    ---------
    scenario:
        Szenario-Dictionary (Schlüssel: holidays).
    timestamps:
        Simulationszeitindex zur Ableitung der relevanten Jahre.
    
    Rückgabe
    --------
    set[date]
        Menge der Feiertage als `datetime.date`.
    """
    configuration = scenario.get("holidays") or {}
    explicit = configuration.get("dates")
    if isinstance(explicit, list) and explicit:
        output: set[date] = set()
        for x in explicit:
            try:
                output.add(pd.to_datetime(x).date())
            except Exception:
                pass
        return output

    country = str(configuration.get("country", "")).strip()
    if not country:
        return set()

    try:
        import holidays as _holidays
    except Exception:
        warnings.warn("python-package 'holidays' nicht installiert -> holiday_dates bleibt leer.", UserWarning)
        return set()

    years = sorted({pd.Timestamp(t).year for t in timestamps})
    subdiv = configuration.get("subdivision") or configuration.get("state") or None
    try:
        cal = _holidays.country_holidays(country, subdiv=subdiv, years=years)
        return set(cal.keys())
    except Exception:
        return set()


def get_daytype_calendar(*, start_datetime: pd.Timestamp, horizon_days: int, holiday_dates: set[date]) -> dict[str, list[date]]:
    """
    Erstellt einen Kalender der Tagtypen im Simulationshorizont.
    
    Parameter
    ---------
    start_datetime:
        Startdatum der Simulation (Datumsteil wird verwendet).
    horizon_days:
        Anzahl Kalendertage im Horizont.
    holiday_dates:
        Feiertage als Menge von `date`.
    
    Rückgabe
    --------
    dict[str, list[date]]
        Zuordnung mit Schlüsseln "working_day", "saturday", "sunday_holiday".
    """
    output = {"working_day": [], "saturday": [], "sunday_holiday": []}
    start_date = pd.Timestamp(start_datetime).date()
    for i in range(int(horizon_days)):
        d = (pd.Timestamp(start_date) + pd.Timedelta(days=i)).date()
        weekday_index = int(pd.Timestamp(d).weekday())
        if d in holiday_dates or weekday_index == 6:
            output["sunday_holiday"].append(d)
        elif weekday_index == 5:
            output["saturday"].append(d)
        else:
            output["working_day"].append(d)
    return output


def decorate_title_with_status(title: str, charging_strategy: str, strategy_status: str | None = None) -> str:
    """
    Erweitert einen Plot-Titel um Strategieinformationen.
    
    Parameter
    ---------
    title:
        Basistitel.
    charging_strategy:
        Gewählte Strategie (z. B. "pv").
    strategy_status:
        Optionaler Status/alias, der zusätzlich angezeigt wird.
    
    Rückgabe
    --------
    str
        Formatierter Titelstring.

    """
    if strategy_status and str(strategy_status) != str(charging_strategy):
        return f"{title} — {charging_strategy} ({strategy_status})"
    return f"{title} — {charging_strategy}"


def initialize_time_window(*, timestamps: pd.DatetimeIndex, scenario: dict, days: int = 1) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Erzeugt ein Zoom-Zeitfenster ab Simulationsstart.
    
    Parameter
    ---------
    timestamps:
        Simulationszeitindex.
    scenario:
        Szenario-Dictionary (enthält Zeitauflösung).
    days:
        Länge des Fensters in Tagen (Default: 1).
    
    Rückgabe
    --------
    tuple[pd.Timestamp, pd.Timestamp]
        (Start, Ende) des Fensters.
    """
    if len(timestamps) == 0:
        t = pd.Timestamp("1970-01-01")
        return t, t
    step_min = get_time_resolution_min_from_scenario(scenario)
    steps = int(max(1, int(days)) * round(24 * 60 / step_min))
    end_i = int(min(len(timestamps) - 1, steps - 1))
    return pd.Timestamp(timestamps[0]), pd.Timestamp(timestamps[end_i])


def group_sessions_by_day(sessions_out: list[dict[str, Any]], *, only_plugged: bool = False) -> dict[date, list[dict[str, Any]]]:
    """
    Gruppiert Sessions nach Ankunftsdatum.
    
    Parameter
    ---------
    sessions_out:
        Session-Ergebnisliste aus der Simulation.
    only_plugged:
        Wenn True, werden nur Sessions mit Status "plugged" berücksichtigt.
    
    Rückgabe
    --------
    dict[date, list[dict[str, Any]]]
        Zuordnung von Datum auf Sessions.
    """
    output: dict[date, list[dict[str, Any]]] = {}
    for session in sessions_out:
        if only_plugged and str(session.get("status")) != "plugged":
            continue
        at = session.get("arrival_time")
        key = pd.to_datetime(at).date() if at is not None else date(1970, 1, 1)
        output.setdefault(key, []).append(session)
    return output


def summarize_sessions(sessions_out: list[dict[str, Any]], *, tol_kwh: float = 1e-9) -> dict[str, Any]:
    """
    Berechnet eine kompakte KPI-Zusammenfassung über alle Sessions.

    Parameter
    ---------
    sessions_out:
        Session-Ergebnisliste aus der Simulation.
    tol_kwh:
        Toleranz für Restenergie zur Bewertung "Ziel-SoC erreicht".
    
    Rückgabe
    --------
    dict[str, Any]
        Enthält u. a. Anzahl Sessions, rejected/plugged und eine Liste von Sessions,
        die den Ziel-SoC nicht erreicht haben. 
    """
    total = len(sessions_out)
    plugged = [session for session in sessions_out if str(session.get("status")) == "plugged"]
    rejected = [session for session in sessions_out if str(session.get("status")) == "drive_off"]

    not_reached_rows: list[dict[str, Any]] = []
    for session in plugged:
        rem = float(session.get("remaining_site_kwh", 0.0) or 0.0)
        if rem > tol_kwh:
            not_reached_rows.append(
                {
                    "session_id": session.get("session_id"),
                    "charger_id": session.get("charger_id"),
                    "arrival_time": session.get("arrival_time"),
                    "parking_duration_min": session.get("parking_duration_min"),
                    "soc_arrival": session.get("state_of_charge_at_arrival"),
                    "soc_end": session.get("final_soc"),
                    "remaining_energy_kwh": rem,
                }
            )

    return {
        "num_sessions_total": int(total),
        "num_sessions_plugged": int(len(plugged)),
        "num_sessions_rejected": int(len(rejected)),
        "not_reached_rows": not_reached_rows,
    }


def build_plugged_sessions_preview_table(sessions_out: list[dict[str, Any]], *, n: int = 10) -> pd.DataFrame:
    """
    Erzeugt eine Vorschau-Tabelle für die ersten n erfolgreichen Ladesessions.
    
    Parameter
    ---------
    sessions_out:
        Session-Ergebnisliste aus der Simulation.
    n:
        Maximale Anzahl Zeilen.
    
    Rückgabe
    --------
    pd.DataFrame
        Vorschau mit wichtigen Session-Attributen.
    """
    rows = []
    for session in sessions_out:
        if str(session.get("status")) != "plugged":
            continue
        rows.append(
            {
                "session_id": session.get("session_id"),
                "charger_id": session.get("charger_id"),
                "arrival_time": session.get("arrival_time"),
                "departure_time": session.get("departure_time"),
                "parking_duration_min": session.get("parking_duration_min"),
                "soc_arrival": session.get("state_of_charge_at_arrival"),
                "soc_end": session.get("final_soc"),
                "charged_site_kwh": session.get("charged_site_kwh"),
                "charged_pv_site_kwh": session.get("charged_pv_site_kwh"),
                "remaining_site_kwh": session.get("remaining_site_kwh"),
                "vehicle_name": session.get("vehicle_name"),
            }
        )
    dataframe = pd.DataFrame(rows)
    if len(dataframe) == 0:
        return dataframe
    return dataframe.sort_values(["arrival_time", "session_id"], na_position="last").head(int(n)).reset_index(drop=True)


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
    Baut eine Zeitreihen-Tabelle für Auswertung und Plots.
    
    Parameter
    ---------
    timestamps:
        Simulationszeitindex.
    scenario:
        Szenario-Dictionary (Zeitauflösung).
        base_load_kwh_per_step:
            Grundlast als Energie pro Schritt.
        pv_generation_kwh_per_step:
            PV-Erzeugung als Energie pro Schritt.
        ev_load_kw:
            EV-Ladeleistung als Leistung pro Schritt.
        market_price_eur_per_mwh:
            Optional: Marktpreis pro Schritt.
        debug_df:
            Optional: Debug-Bilanzdaten; Spalten werden übernommen, falls kompatibel.
    
    Rückgabe
    --------
    pd.DataFrame
        Einheitliche Tabelle (timestamp, base_load_kw, pv_generation_kw, ev_load_kw, ...).
    """
    step_h = get_time_resolution_min_from_scenario(scenario) / 60.0
    dataframe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "base_load_kw": np.asarray(base_load_kwh_per_step, float) / max(step_h, 1e-12),
            "pv_generation_kw": np.asarray(pv_generation_kwh_per_step, float) / max(step_h, 1e-12),
            "ev_load_kw": np.asarray(ev_load_kw, float),
        }
    )
    if market_price_eur_per_mwh is not None:
        dataframe["market_price_eur_per_mwh"] = np.asarray(market_price_eur_per_mwh, float)
    if debug_df is not None and len(debug_df) == len(dataframe):
        for c in debug_df.columns:
            if c != "timestamp" and c not in dataframe.columns:
                dataframe[c] = debug_df[c].to_numpy()
    return dataframe


def build_site_overview_plot_data(*, timeseries_dataframe: pd.DataFrame, scenario: dict, start=None, end=None) -> dict[str, Any]:
    """
    Bereitet Daten für eine Standort-Übersichtsgrafik vor.
    
    Parameter
    ---------
    timeseries_dataframe:
        Zeitreihen-DataFrame der Simulation.
    scenario:
        Szenario-Dictionary (für Grid-Limit).
    start, end:
        Optionales Zeitfenster (inklusive Grenzen).
    
    Rückgabe
    --------
    dict[str, Any]
        Enthält gefiltertes DataFrame, Gesamtlast, PV (optional) und Grid-Limit.
    """
    dataframe = timeseries_dataframe.copy()
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
    if start is not None:
        dataframe = dataframe[dataframe["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        dataframe = dataframe[dataframe["timestamp"] <= pd.Timestamp(end)]

    total = dataframe["base_load_kw"].astype(float).fillna(0.0) + dataframe["ev_load_kw"].astype(float).fillna(0.0)
    pv = dataframe["pv_generation_kw"].astype(float).fillna(0.0) if "pv_generation_kw" in dataframe.columns else None

    return {
        "dataframe": dataframe.reset_index(drop=True),
        "total_load_kw": total.to_numpy(),
        "pv_generation_kw": None if pv is None else pv.to_numpy(),
        "grid_limit_kw": float(scenario["site"].get("grid_limit_p_avb_kw", 0.0)),
    }


def build_ev_power_by_source_timeseries(timeseries_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Leitet EV-Leistung nach Quelle (PV vs Netz) als Zeitreihe ab.
    
    Parameter
    ---------
    timeseries_dataframe:
        Zeitreihen-DataFrame der Simulation (ggf. mit Debug-Spalten).
    
    Rückgabe
    --------
    pd.DataFrame
        Spalten: timestamp, ev_from_pv_kw, ev_from_grid_kw
    """
    dataframe = timeseries_dataframe.copy()
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
    step_h = (
        float((dataframe["timestamp"].iloc[1] - dataframe["timestamp"].iloc[0]).total_seconds()) / 3600.0
        if len(dataframe) >= 2 else 0.25
    )
    step_h = max(step_h, 1e-12)

    ev_kw = dataframe["ev_load_kw"].astype(float).fillna(0.0).to_numpy()

    if "pv_to_ev_kwh_per_step" in dataframe.columns:
        pv_kw = dataframe["pv_to_ev_kwh_per_step"].astype(float).fillna(0.0).to_numpy() / step_h
        grid_kw = dataframe.get("grid_to_ev_kwh_per_step")
        if isinstance(grid_kw, pd.Series):
            grid_kw = grid_kw.astype(float).fillna(0.0).to_numpy() / step_h
        else:
            grid_kw = np.maximum(ev_kw - pv_kw, 0.0)
    elif "pv_ev_tracked_kwh_per_step" in dataframe.columns:
        pv_kw = dataframe["pv_ev_tracked_kwh_per_step"].astype(float).fillna(0.0).to_numpy() / step_h
        pv_kw = np.clip(pv_kw, 0.0, ev_kw)
        grid_kw = np.maximum(ev_kw - pv_kw, 0.0)
    else:
        pv_kw = np.zeros_like(ev_kw)
        grid_kw = ev_kw

    return pd.DataFrame({"timestamp": dataframe["timestamp"], "ev_from_pv_kw": pv_kw, "ev_from_grid_kw": grid_kw})


def build_charger_traces_dataframe(*, sessions_out: list[dict[str, Any]], scenario: dict, vehicle_curves_by_name: dict[str, VehicleChargingCurve], timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Erzeugt einen zeitaufgelösten Trace pro Ladepunkt und Session.
    
    Parameter
    ---------
    sessions_out:
        Session-Ergebnisse inkl. Plan-Arrays.
    scenario:
        Szenario-Dictionary (Zeitauflösung, Effizienz).
    vehicle_curves_by_name:
        Ladekurven inkl. Batteriekapazitäten.
    timestamps:
        Simulationszeitindex.
    
    Rückgabe
    --------
    pd.DataFrame
        Zeilen pro (Zeit, Ladepunkt) mit Leistung, PV-Anteil und SoC.
    
    """
    step_h = get_time_resolution_min_from_scenario(scenario) / 60.0
    eff = float(scenario["site"].get("charger_efficiency", 1.0))
    rows: list[dict[str, Any]] = []

    for session in sessions_out:
        if str(session.get("status")) != "plugged":
            continue

        a = int(session.get("arrival_step", 0))
        d = int(session.get("departure_step", a))
        a = int(np.clip(a, 0, max(0, len(timestamps) - 1)))
        d = int(np.clip(d, 0, len(timestamps)))
        if d <= a:
            continue
        window_length_steps = d - a

        plan_site = np.asarray(session.get("plan_site_kwh_per_step", np.zeros(0)), float).reshape(-1)[:window_length_steps]
        plan_pv = np.asarray(session.get("plan_pv_site_kwh_per_step", np.zeros(0)), float).reshape(-1)[:window_length_steps]
        if len(plan_site) < window_length_steps:
            temporary_dataframe = np.zeros(window_length_steps, float); temporary_dataframe[:len(plan_site)] = plan_site; plan_site = temporary_dataframe
        if len(plan_pv) < window_length_steps:
            temporary_dataframe = np.zeros(window_length_steps, float); temporary_dataframe[:len(plan_pv)] = plan_pv; plan_pv = temporary_dataframe

        soc0 = session.get("state_of_charge_at_arrival")
        soc0 = float(np.clip(float(soc0), 0.0, 1.0)) if soc0 is not None else np.nan

        curve = vehicle_curves_by_name.get(str(session.get("vehicle_name", "")))
        cap = float(max(getattr(curve, "battery_capacity_kwh", 1e-12), 1e-12))

        batt_added_before = np.concatenate(([0.0], np.cumsum(plan_site[:-1]) * eff))
        soc_trace = (
            np.clip(soc0 + batt_added_before / cap, 0.0, 1.0)
            if np.isfinite(soc0)
            else np.full(window_length_steps, np.nan)
        )
        for i in range(window_length_steps):
            abs_step = a + i
            site_kwh = float(plan_site[i])
            pv_kwh = float(np.clip(plan_pv[i], 0.0, site_kwh))
            rows.append(
                {
                    "timestamp": pd.Timestamp(timestamps[abs_step]),
                    "charger_id": None if session.get("charger_id") is None else int(session["charger_id"]),
                    "session_id": session.get("session_id"),
                    "vehicle_name": session.get("vehicle_name"),
                    "power_kw": site_kwh / max(step_h, 1e-12),
                    "pv_power_kw": pv_kwh / max(step_h, 1e-12),
                    "soc": float(soc_trace[i]) if np.isfinite(soc_trace[i]) else np.nan,
                    "is_plugged": True,
                    "is_charging": bool(site_kwh > 1e-12),
                }
            )

    return pd.DataFrame(rows)


def build_power_per_charger_timeseries(charger_traces_dataframe: pd.DataFrame, *, charger_id: int, start=None, end=None) -> pd.DataFrame:
    """
    Filtert den Ladepunkt-Trace auf einen Ladepunkt und optionales Zeitfenster.
    
    Parameter
    ---------
    charger_traces_dataframe:
        Trace-DataFrame aus `build_charger_traces_dataframe`.
    charger_id:
        Ladepunkt-ID (0-basiert).
    start, end:
        Optionales Zeitfenster.
    
    Rückgabe
    --------
    pd.DataFrame
        Gefilterte Zeitreihe für den Ladepunkt.
    """
    dataframe = charger_traces_dataframe.copy()
    if len(dataframe) == 0:
        return dataframe
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
    dataframe = dataframe[dataframe["charger_id"] == int(charger_id)]
    if start is not None:
        dataframe = dataframe[dataframe["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        dataframe = dataframe[dataframe["timestamp"] <= pd.Timestamp(end)]
    return dataframe.sort_values("timestamp").reset_index(drop=True)


def build_soc_timeseries_by_charger(*, charger_traces_dataframe: pd.DataFrame, charger_ids: list[int], start=None, end=None) -> dict[int, pd.DataFrame]:
    """
    Erzeugt SoC-Zeitreihen je Ladepunkt.
    
    Parameter
    ---------
    charger_traces_dataframe:
        Trace-DataFrame aus `build_charger_traces_dataframe`.
    charger_ids:
        Liste der gewünschten Ladepunkt-IDs.
    start, end:
        Optionales Zeitfenster.
    
    Rückgabe
    --------
    dict[int, pd.DataFrame]
        Zuordnung charger_id -> DataFrame(timestamp, soc, session_id).
    """
    output: dict[int, pd.DataFrame] = {}
    charger_traces_dataframe_copy = charger_traces_dataframe.copy()
    if len(charger_traces_dataframe_copy) == 0:
        return {int(charger_id): pd.DataFrame() for charger_id in charger_ids}
    charger_traces_dataframe_copy["timestamp"] = pd.to_datetime(charger_traces_dataframe_copy["timestamp"])
    if start is not None:
        charger_traces_dataframe_copy = charger_traces_dataframe_copy[charger_traces_dataframe_copy["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        charger_traces_dataframe_copy = charger_traces_dataframe_copy[charger_traces_dataframe_copy["timestamp"] <= pd.Timestamp(end)]
    for charger_id in charger_ids:
        dataframe = charger_traces_dataframe_copy[charger_traces_dataframe_copy["charger_id"] == int(charger_id)].sort_values("timestamp")
        output[int(charger_id)] = dataframe[[c for c in ["timestamp", "soc", "session_id"] if c in dataframe.columns]].reset_index(drop=True)
    return output


def build_charger_power_heatmap_matrix(charger_traces_dataframe: pd.DataFrame, *, start=None, end=None) -> dict[str, Any]:
    """
    Bereitet eine Heatmap-Matrix (Ladepunkt x Zeit) der Ladeleistungen vor.
    
    Parameter
    ---------
    charger_traces_dataframe:
        Trace-DataFrame aus `build_charger_traces_dataframe`.
    start, end:
        Optionales Zeitfenster.
    
    Rückgabe
    --------
    dict[str, Any]
        Enthält Matrix (charger x time), timestamps und charger_ids.
    """
    dataframe = charger_traces_dataframe.copy()
    if len(dataframe) == 0:
        return {"matrix": np.zeros((0, 0)), "timestamps": [], "charger_ids": []}

    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
    if start is not None:
        dataframe = dataframe[dataframe["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        dataframe = dataframe[dataframe["timestamp"] <= pd.Timestamp(end)]
    if len(dataframe) == 0:
        return {"matrix": np.zeros((0, 0)), "timestamps": [], "charger_ids": []}

    charger_ids = sorted({int(x) for x in dataframe["charger_id"].dropna().unique().tolist()})
    pivot = dataframe.pivot_table(index="timestamp", columns="charger_id", values="power_kw", aggfunc="sum").fillna(0.0)

    timestamp_index = pivot.index.sort_values()
    if len(timestamp_index) >= 2:
        freq = timestamp_index[1] - timestamp_index[0]
        full_timestamp_index = pd.date_range(timestamp_index[0], timestamp_index[-1], freq=freq)
        pivot = pivot.reindex(full_timestamp_index).fillna(0.0)
        timestamp_list = list(full_timestamp_index)
    else:
        timestamp_list = list(timestamp_index)

    for charger_id in charger_ids:
        if charger_id not in pivot.columns:
            pivot[charger_id] = 0.0
    pivot = pivot[charger_ids]

    return {"matrix": pivot.to_numpy(float).T, "timestamps": timestamp_list, "charger_ids": charger_ids}


def build_ev_power_by_mode_timeseries_dataframe(*, timeseries_dataframe: pd.DataFrame, sessions_out: list[dict[str, Any]], scenario: dict) -> pd.DataFrame:
    """
    Aggregiert EV-Leistung nach Lademodus (z. B. PV/Market/Immediate).
    
    Parameter
    ---------
    timeseries_dataframe:
        Zeitreihen-DataFrame der Simulation.
    sessions_out:
        Session-Ergebnisse inkl. Plan-Arrays.
    scenario:
        Szenario-Dictionary (Strategie und Zeitauflösung).

    Rückgabe
    --------
    pd.DataFrame
        Spalten: timestamp, ev_generation_kw, ev_market_kw, ev_immediate_kw
    """
    dataframe = timeseries_dataframe.copy()
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
    n = len(dataframe)

    step_h = get_time_resolution_min_from_scenario(scenario) / 60.0
    strategy = str(scenario.get("charging_strategy", "immediate")).strip().lower()

    ev_gen = np.zeros(n, float)
    ev_market = np.zeros(n, float)
    ev_imm = np.zeros(n, float)

    for session in sessions_out:
        if str(session.get("status")) != "plugged":
            continue
        a = int(np.clip(int(session.get("arrival_step", 0)), 0, n))
        d = int(np.clip(int(session.get("departure_step", a)), 0, n))
        if d <= a:
            continue
        window_length_steps = d - a

        site = np.asarray(session.get("plan_site_kwh_per_step", np.zeros(0)), float).reshape(-1)[:window_length_steps]
        pv = np.asarray(session.get("plan_pv_site_kwh_per_step", np.zeros(0)), float).reshape(-1)[:window_length_steps]
        mkt = np.asarray(session.get("plan_market_kwh_per_step", np.zeros(0)), float).reshape(-1)[:window_length_steps]

        # pad falls nötig
        if len(site) < window_length_steps:
            temporary_dataframe = np.zeros(window_length_steps, float); temporary_dataframe[:len(site)] = site; site = temporary_dataframe
        if len(pv) < window_length_steps:
            temporary_dataframe = np.zeros(window_length_steps, float); temporary_dataframe[:len(pv)] = pv; pv = temporary_dataframe
        if len(mkt) < window_length_steps:
            temporary_dataframe = np.zeros(window_length_steps, float); temporary_dataframe[:len(mkt)] = mkt; mkt = temporary_dataframe

        if strategy == "immediate":
            ev_imm[a:d] += site
        elif strategy == "market":
            ev_market[a:d] += site  # market == site
        else:
            pv = np.clip(pv, 0.0, site)
            ev_gen[a:d] += pv
            # grid-fallback ist plan_market; wenn du in pv-strategy nur Grid dort schreibst, passt das.
            ev_market[a:d] += mkt if np.sum(mkt) > 1e-12 else np.maximum(site - pv, 0.0)

    return pd.DataFrame(
        {
            "timestamp": dataframe["timestamp"],
            "ev_generation_kw": ev_gen / max(step_h, 1e-12),
            "ev_market_kw": ev_market / max(step_h, 1e-12),
            "ev_immediate_kw": ev_imm / max(step_h, 1e-12),
        }
    )


def get_most_used_vehicle_name(
    *,
    sessions_out: list[dict[str, object]],
    charger_traces_dataframe: pd.DataFrame,
    only_plugged_sessions: bool = True,
) -> str:
    """
    Ermittelt den am häufigsten geladenen Fahrzeugnamen (vehicle_name).

    Die Funktion nutzt primär ``charger_traces_dataframe`` (sofern vorhanden), da dieses
    DataFrame die tatsächlich geplanten/geladenen Zeitschritte enthält und damit die
    zuverlässigste Quelle für die Häufigkeit eines Fahrzeugtyps ist. Falls kein
    Trace-DataFrame verfügbar ist oder die Spalte ``vehicle_name`` fehlt, wird als
    Fallback ``sessions_out`` ausgewertet.

    Parameter
    ---------
    sessions_out:
        Liste von Session-Ergebnis-Dictionaries aus der Simulation (z. B. aus
        ``simulate_site_fcfs_with_planning``). Erwartet u. a. Keys wie
        ``"vehicle_name"`` und optional ``"status"``.
    charger_traces_dataframe:
        Zeitreihen-DataFrame mit Charger- und Session-Traces. Erwartet eine Spalte
        ``"vehicle_name"``. Wenn das DataFrame nicht leer ist, wird diese Quelle
        bevorzugt.
    only_plugged_sessions:
        Wenn ``True``, werden beim Fallback über ``sessions_out`` nur Sessions mit
        ``status == "plugged"`` gezählt. Wenn ``False``, werden alle Sessions
        berücksichtigt.

    Rückgabe
    --------
    str
        Der häufigste ``vehicle_name``. Wenn keine Fahrzeugnamen gefunden werden,
        wird ein leerer String ``""`` zurückgegeben.
    """
    if charger_traces_dataframe is not None and len(charger_traces_dataframe) > 0 and "vehicle_name" in charger_traces_dataframe.columns:
        vehicle_name_value_counts = charger_traces_dataframe["vehicle_name"].dropna().astype(str).value_counts()
        if len(vehicle_name_value_counts) > 0:
            return str(vehicle_name_value_counts.index[0])

    rows = sessions_out
    if only_plugged_sessions:
        rows = [session for session in sessions_out if str(session.get("status")) == "plugged"]

    c = Counter(str(session.get("vehicle_name")) for session in rows if session.get("vehicle_name") is not None)
    return c.most_common(1)[0][0] if c else ""


def build_master_curve_and_actual_points_for_vehicle(
    *,
    charger_traces_dataframe: pd.DataFrame,
    scenario: dict,
    curve,  # VehicleChargingCurve
    power_tolerance_kw: float = 1e-6,
) -> dict[str, object]:
    """
    Bereitet Daten für den Vergleich von Master-Ladekurve und Ist-Ladepunkten auf.

    Die Funktion extrahiert aus ``curve`` die fahrzeugspezifische Master-Ladekurve
    (maximal zulässige Ladeleistung in Abhängigkeit vom SoC) und stellt dieser die
    tatsächlich in der Simulation auftretenden Ladepunkte (SoC und Leistung) aus
    ``charger_traces_dataframe`` gegenüber.

    Wichtiger Hinweis zur Leistungsdefinition
    -----------------------------------------
    - Die Masterkurve ``curve.power_kw`` ist auf **Batterieseite** definiert.
    - ``charger_traces_dataframe["power_kw"]`` ist die Leistung auf **Standort-/Ladepunktseite**.
      Für den Vergleich wird daher mit der Ladepunkt-Effizienz
      ``scenario["site"]["charger_efficiency"]`` auf Batterieseite umgerechnet:
      ``power_batt_kw = power_site_kw * charger_efficiency``.

    Parameter
    ---------
    charger_traces_dataframe:
        DataFrame mit Zeitschritt-Traces der Simulation. Erwartete Spalten:
        - ``"vehicle_name"`` (Fahrzeugname),
        - ``"timestamp"`` (Zeitpunkt),
        - ``"soc"`` (SoC als Anteil 0..1),
        - ``"power_kw"`` (Leistung auf Standort-/Ladepunktseite in kW).
    scenario:
        Szenario-Dictionary. Verwendet wird insbesondere ``scenario["site"]["charger_efficiency"]``.
    curve:
        Fahrzeug-Ladekurve (z. B. ``VehicleChargingCurve``) mit Attributen
        ``vehicle_name``, ``state_of_charge_fraction`` und ``power_kw``.
    power_tolerance_kw:
        Toleranz in kW für die Überschreitungsprüfung. Ein Ist-Punkt gilt als Überschreitung,
        wenn ``actual_power_batt_kw > allowed_power_batt_kw + power_tolerance_kw``.

    Rückgabe
    --------
    dict[str, object]
        Dictionary mit folgenden Einträgen:
        - ``"vehicle_name"``: Name des Fahrzeugs (String).
        - ``"master_soc"``: SoC-Stützstellen der Masterkurve (np.ndarray, 0..1).
        - ``"master_power_battery_kw"``: Master-Leistung auf Batterieseite (np.ndarray, kW).
        - ``"actual_soc"``: Ist-SoC-Werte aus den Traces (np.ndarray, 0..1).
        - ``"actual_power_batt_kw"``: Ist-Leistung auf Batterieseite (np.ndarray, kW).
        - ``"violation_mask"``: Bool-Maske gleicher Länge wie ``actual_soc`` mit True bei Überschreitung.
        - ``"number_violations"``: Anzahl der Überschreitungen (int).

        Wenn keine Ist-Daten vorhanden sind (z. B. leerer DataFrame oder keine passenden Zeilen),
        werden ``actual_*`` als leere Arrays zurückgegeben und ``number_violations`` ist 0.
    """
    master_soc = np.asarray(curve.state_of_charge_fraction, dtype=float).reshape(-1)
    master_power_batt_kw = np.asarray(curve.power_kw, dtype=float).reshape(-1)

    empty_result = {
        "vehicle_name": curve.vehicle_name,
        "master_soc": master_soc,
        "master_power_battery_kw": master_power_batt_kw,
        "actual_soc": np.array([]),
        "actual_power_batt_kw": np.array([]),
        "violation_mask": np.array([], dtype=bool),
        "number_violations": 0,
    }

    if charger_traces_dataframe is None or len(charger_traces_dataframe) == 0:
        return empty_result

    dataframe = charger_traces_dataframe.copy()

    # Spalten-Check (robuster als .get(...).astype(...))
    required_cols = {"vehicle_name", "timestamp", "soc", "power_kw"}
    if not required_cols.issubset(set(dataframe.columns)):
        return empty_result

    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], errors="coerce")
    dataframe = dataframe[dataframe["vehicle_name"].astype(str) == str(curve.vehicle_name)]

    actual_soc = pd.to_numeric(dataframe["soc"], errors="coerce").to_numpy(dtype=float)
    actual_power_site_kw = pd.to_numeric(dataframe["power_kw"], errors="coerce").to_numpy(dtype=float)

    eff = float(np.clip(float(scenario["site"].get("charger_efficiency", 1.0)), 1e-12, 1.0))
    actual_power_batt_kw = np.maximum(actual_power_site_kw * eff, 0.0)

    # Clean
    actual_soc = np.clip(actual_soc, 0.0, 1.0)
    valid = np.isfinite(actual_soc) & np.isfinite(actual_power_batt_kw)
    actual_soc = actual_soc[valid]
    actual_power_batt_kw = actual_power_batt_kw[valid]

    if actual_soc.size == 0 or master_soc.size < 2:
        return {
            "vehicle_name": curve.vehicle_name,
            "master_soc": master_soc,
            "master_power_battery_kw": master_power_batt_kw,
            "actual_soc": actual_soc,
            "actual_power_batt_kw": actual_power_batt_kw,
            "violation_mask": np.array([], dtype=bool),
            "number_violations": 0,
        }

    allowed_kw = np.interp(actual_soc, master_soc, master_power_batt_kw)
    violation_mask = actual_power_batt_kw > (allowed_kw + float(power_tolerance_kw))

    return {
        "vehicle_name": curve.vehicle_name,
        "master_soc": master_soc,
        "master_power_battery_kw": master_power_batt_kw,
        "actual_soc": actual_soc,
        "actual_power_batt_kw": actual_power_batt_kw,
        "violation_mask": violation_mask,
        "number_violations": int(np.count_nonzero(violation_mask)),
    }
