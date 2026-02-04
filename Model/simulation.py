from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Dict, Optional, List, Tuple, Any, Callable, Set, Literal
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
#   - Zeitprofil der EV-Ladeleistung (kW)
#   - Session-Details für KPI-Analysen
#   - optional Debug-Zeitreihen für Notebook-Auswertungen

# =============================================================================
# 0) Hilfsfunktionen: Zeit / Einheiten
# =============================================================================

def normalize_to_timestamps_timezone(dt, timestamps: pd.DatetimeIndex) -> pd.Timestamp:
    """
    Normalisiert einen beliebigen Zeitwert auf die Zeitzone eines Simulations-Zeitrasters.

    In der Simulation treten Zeitwerte in verschiedenen Formen auf (naive/aware `datetime`,
    Strings, `pd.Timestamp`). Diese Funktion sorgt dafür, dass solche Zeitwerte konsistent zur
    Zeitzone des Simulationsrasters `timestamps` sind, damit Vergleiche, Reindexing und Mapping
    auf Simulationsschritte zuverlässig funktionieren – auch über Sommer-/Winterzeit (DST) hinweg.
    """
    ## timezone von Zeitreihen prüfen & angleichen!
    tz = timestamps.tz
    ts = pd.Timestamp(dt)

    if tz is None:
        # falls ts tz-aware: tz entfernen, lokale Zeit beibehalten
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

    # tz-aware Ziel: ts in tz bringen
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
        if pd.isna(ts):
            raise ValueError(f"Timestamp ist in Zeitzone {tz} nicht eindeutig/ungültig: {dt}")
        return ts

    return ts.tz_convert(tz)


def normalize_and_floor_to_grid(
    dt,
    timestamps: pd.DatetimeIndex,
    time_resolution_min: int,
) -> pd.Timestamp:
    """
    Normalisiert `dt` auf die Zeitzone von `timestamps` und floort (Abrunden nach unten) auf das Simulationsraster.
    - Sekunden/Mikrosekunden werden entfernt
    """
    ## An welcher Stelle wird die Funktion benötigt?
    ts = normalize_to_timestamps_timezone(dt, timestamps)
    ts = ts.replace(second=0, microsecond=0)
    ts = ts.floor(f"{int(time_resolution_min)}min")
    return ts


def _infer_step_hours_from_index(datetime_index: pd.DatetimeIndex, fallback_minutes: int) -> float:
    """
    Bestimmt die zeitliche Schrittweite eines Zeitrasters (in Stunden) aus einem DatetimeIndex.

    In den CSV-Dateien stehen Gebäudelast und PV-Erzeugung als *Energie pro Zeitschritt*
    (z. B. kWh pro 15 Minuten). Wenn diese Werte auf ein anderes Simulationsraster
    (z. B. 5 Minuten oder 30 Minuten) überführen willst, musst du wissen, wie lang ein
    Input-Zeitschritt tatsächlich ist. Sonst würdest du beim Reindexing (z. B. per ffill)
    Energiewerte fälschlich mehrfach zählen oder zu wenig zählen.

    Beispiel:
    - CSV: 3.0 kWh pro 15 Minuten um 08:00
    - Simulation: 5-Minuten-Raster
    Wenn man 3.0 kWh einfach auf 08:00, 08:05, 08:10 "vorwärts füllt", hätte man
    9.0 kWh in 15 Minuten (falsch). Korrekt ist:
    1) Aus 15 Minuten Schrittweite ableiten
    2) Energie -> Leistung umrechnen (kW)
    3) Leistung auf Simulationsraster bringen
    4) zurück zu kWh pro Simulationsschritt

    Fallback-Verhalten
    ------------------
    Wenn die Schrittweite nicht zuverlässig aus dem Index bestimmt werden kann
    (z. B. zu wenige Zeitstempel oder ungültige Differenzen), wird stattdessen
    ``fallback_minutes`` verwendet.

    Parameters
    ----------
    datetime_index:
        Zeitindex der Eingabedaten (z. B. aus der CSV).
    fallback_minutes:
        Ersatz-Schrittweite in Minuten, falls die Bestimmung aus dem Index nicht möglich ist.

    Returns
    -------
    float
        Schrittweite in Stunden (z. B. 0.25 für 15 Minuten).
    """
    if len(datetime_index) < 2:
        return float(fallback_minutes) / 60.0

    time_differences = pd.Series(datetime_index).diff().dropna()
    if time_differences.empty:
        return float(fallback_minutes) / 60.0

    median_seconds = float(time_differences.dt.total_seconds().median())
    if not np.isfinite(median_seconds) or median_seconds <= 0:
        return float(fallback_minutes) / 60.0

    return median_seconds / 3600.0


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


def resolve_paths_relative_to_yaml(scenario: Dict[str, Any], scenario_path: str) -> Dict[str, Any]:
    """
    Löst relative Dateipfade im Szenario relativ zum Speicherort der YAML-Datei auf.

    Zweck
    -----
    In der YAML werden Dateipfade (z. B. zu CSV-Dateien) häufig relativ angegeben.
    Diese Funktion erzeugt eine tiefe Kopie des eingelesenen Szenarios und ersetzt
    relevante Pfadfelder durch absolute Pfade, sodass das Programm unabhängig vom
    aktuellen Working Directory zuverlässig läuft.

    Aufgelöste Pfadfelder (aktueller YAML-Stand)
    --------------------------------------------
    - scenario["localload_pv_market_csv"]
        Pfad zur CSV mit Grundlast, PV-Erzeugung und Marktpreisen.
    - scenario["vehicles"]["vehicle_curve_csv"]
        Pfad zur CSV mit Fahrzeug-Ladekurven.

    Verhalten
    ---------
    - Relative Pfade werden relativ zum Ordner der YAML-Datei aufgelöst.
    - Absolute Pfade bleiben unverändert.
    - "~" wird expandiert (Home-Verzeichnis).
    - Leere Strings bleiben unverändert.
    - Wenn ein Pfadfeld fehlt oder None ist, wird es nicht verändert.

    Parameters
    ----------
    scenario:
        Szenario-Dictionary (zuvor aus YAML eingelesen).
    scenario_path:
        Pfad zur YAML-Datei (dient als Basis für relative Pfade).

    Returns
    -------
    Dict[str, Any]
        Neue (deep-copied) Szenario-Struktur mit aufgelösten absoluten Pfaden.
    """
    base_directory = Path(scenario_path).resolve().parent
    scenario_copy: Dict[str, Any] = copy.deepcopy(scenario)

    def resolve_path_value(path_value: Any) -> Any:
        if path_value is None:
            return None
        path_str = str(path_value).strip()
        if path_str == "":
            return path_value

        path_object = Path(path_str).expanduser()
        if path_object.is_absolute():
            return str(path_object)
        return str((base_directory / path_object).resolve())

    # top-level: localload_pv_market_csv
    if "localload_pv_market_csv" in scenario_copy:
        scenario_copy["localload_pv_market_csv"] = resolve_path_value(
            scenario_copy["localload_pv_market_csv"]
        )

    # vehicles: vehicle_curve_csv
    vehicles_section = scenario_copy.get("vehicles")
    if isinstance(vehicles_section, dict) and "vehicle_curve_csv" in vehicles_section:
        vehicles_section["vehicle_curve_csv"] = resolve_path_value(
            vehicles_section.get("vehicle_curve_csv")
        )

    return scenario_copy


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


def read_local_load_pv_market_from_csv(
    csv_path: str,
    timestamps: pd.DatetimeIndex,
    timezone: Optional[str],
    base_load_annual_kwh: float,
    pv_system_size_kwp: float,
    profiles_are_normalized: bool = True,
    datetime_format: str = "%d.%m.%Y %H:%M",
    separator: str = ";",
    decimal: str = ",",
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Liest Grundlastprofil, PV-Profil und Marktpreis aus CSV und richtet sie auf das Simulationsraster aus.

    CSV-Format (mind. 4 Spalten)
    ----------------------------
    0) Datum/Zeit (Text, strikt nach ``datetime_format``)
    1) Grundlast:
       - wenn profiles_are_normalized=True: normiertes Profil (einheitenlos, >=0)
       - sonst: Gebäudelast [kWh] pro Input-Zeitschritt
    2) PV:
       - wenn profiles_are_normalized=True: Kapazitätsfaktor 0..1 (einheitenlos, >=0)
       - sonst: PV-Erzeugung [kWh] pro Input-Zeitschritt
    3) Marktpreis [€/MWh] pro Input-Zeitschritt

    Skalierung (nur wenn profiles_are_normalized=True)
    --------------------------------------------------
    - Grundlast wird so skaliert, dass die Summe über die CSV = base_load_annual_kwh ergibt.
    - PV wird als Kapazitätsfaktor interpretiert und mit pv_system_size_kwp skaliert (kW -> kWh/step).

    Returns
    -------
    (building_load_kwh_per_step, pv_generation_kwh_per_step, market_price_eur_per_mwh)
    auf ``timestamps`` indiziert.
    """
    dataframe = pd.read_csv(csv_path, sep=separator, decimal=decimal)

    if dataframe.shape[1] < 4:
        raise ValueError(
            "CSV hat nicht genug Spalten. Erwartet: "
            "Datum | Grundlast | PV | Marktpreis."
        )

    datetime_column = dataframe.columns[0]
    datetime_text = dataframe[datetime_column].astype(str).str.strip()

    parsed_datetime = pd.to_datetime(datetime_text, format=datetime_format, errors="coerce")
    if parsed_datetime.isna().any():
        invalid_examples = datetime_text[parsed_datetime.isna()].head(5).to_list()
        raise ValueError(
            f"CSV-Zeitformat stimmt nicht. Erwartet '{datetime_format}'. "
            f"Beispiele ungültiger Werte: {invalid_examples}"
        )

    dataframe = dataframe.copy()
    dataframe[datetime_column] = parsed_datetime
    dataframe = dataframe.sort_values(datetime_column).set_index(datetime_column)

    if timezone:
        if dataframe.index.tz is None:
            dataframe.index = dataframe.index.tz_localize(
                timezone, ambiguous="infer", nonexistent="shift_forward"
            )
        else:
            dataframe.index = dataframe.index.tz_convert(timezone)

    if not dataframe.index.is_unique:
        dataframe = dataframe.groupby(level=0).mean(numeric_only=True)

    # Input-Stepweite bestimmen
    simulation_step_hours = (
        float((timestamps[1] - timestamps[0]).total_seconds()) / 3600.0
        if len(timestamps) > 1 else 0.25
    )
    input_step_hours = _infer_step_hours_from_index(
        dataframe.index, fallback_minutes=int(simulation_step_hours * 60)
    )

    col_base = pd.to_numeric(dataframe.iloc[:, 1], errors="coerce").fillna(0.0).clip(lower=0.0)
    col_pv = pd.to_numeric(dataframe.iloc[:, 2], errors="coerce").fillna(0.0).clip(lower=0.0)
    market_price_eur_per_mwh = pd.to_numeric(dataframe.iloc[:, 3], errors="coerce").fillna(0.0)

    if profiles_are_normalized:
        # (A) Grundlast: Jahresenergie proportional zum Profil verteilen
        base_load_annual_kwh = float(max(base_load_annual_kwh, 0.0))
        w_sum = float(col_base.sum())
        if w_sum <= 0.0 or base_load_annual_kwh <= 0.0:
            building_load_kwh_per_input_step = col_base * 0.0
        else:
            building_load_kwh_per_input_step = (col_base / w_sum) * base_load_annual_kwh

        # (B) PV: Kapazitätsfaktor * kWp => kW, dann * Δt_input => kWh/step
        pv_system_size_kwp = float(max(pv_system_size_kwp, 0.0))
        pv_power_kw = col_pv * pv_system_size_kwp
        pv_generation_kwh_per_input_step = pv_power_kw * float(input_step_hours)
    else:
        # CSV liefert bereits kWh pro Input-Zeitschritt
        building_load_kwh_per_input_step = col_base
        pv_generation_kwh_per_input_step = col_pv

    # Energie (kWh/step) -> Leistung (kW) -> auf Sim-Raster -> zurück zu kWh/Sim-Step
    building_load_kw = building_load_kwh_per_input_step / max(float(input_step_hours), 1e-12)
    pv_generation_kw = pv_generation_kwh_per_input_step / max(float(input_step_hours), 1e-12)

    building_load_kw = building_load_kw.reindex(timestamps, method="nearest")
    pv_generation_kw = pv_generation_kw.reindex(timestamps, method="nearest")

    building_load_kwh_per_step = building_load_kw * float(simulation_step_hours)
    pv_generation_kwh_per_step = pv_generation_kw * float(simulation_step_hours)

    market_price_eur_per_mwh = market_price_eur_per_mwh.reindex(timestamps, method="ffill")

    return building_load_kwh_per_step, pv_generation_kwh_per_step, market_price_eur_per_mwh



def read_vehicle_load_profiles_from_csv(vehicle_curve_csv_path: str) -> Dict[str, VehicleChargingCurve]:
    """
    Liest Fahrzeug-Ladekurven aus einer CSV im festen Wide-Format (ein Fahrzeug pro Spalte).

    Erwarteter Aufbau
    -----------------
    Spalte 0 enthält in den ersten vier Zeilen feste Labels:
    - Zeile 0: "Hersteller"
    - Zeile 1: "Modell"
    - Zeile 2: "Fahrzeugklasse"
    - Zeile 3: "max. Kapazität"

    Ab Zeile 4 enthält Spalte 0 die SoC-Stützstellen (typisch in Prozent, Dezimalkomma erlaubt).
    Ab Spalte 1 stehen pro Fahrzeug die Ladeleistungen in kW (Batterieseite).

    Parameters
    ----------
    vehicle_curve_csv_path:
        Pfad zur CSV-Datei.

    Returns
    -------
    Dict[str, VehicleChargingCurve]
        Dictionary mit Schlüssel = eindeutiger Fahrzeugname und Wert = Ladekurve.

    Raises
    ------
    ValueError
        Wenn das CSV nicht dem erwarteten Format entspricht oder keine Kurven erzeugt werden können.
    """
    vehicle_curve_table = pd.read_csv(
        vehicle_curve_csv_path,
        sep=None,
        engine="python",
        header=None,
        dtype=str,
    )

    if vehicle_curve_table.shape[0] < 6 or vehicle_curve_table.shape[1] < 2:
        raise ValueError(
            "Vehicleloadprofile-CSV hat nicht das erwartete Format "
            "(benötigt mindestens 6 Zeilen und 2 Spalten)."
        )

    expected_labels = ["Hersteller", "Modell", "Fahrzeugklasse", "max. Kapazität"]
    found_labels = [str(vehicle_curve_table.iat[row_index, 0]).strip() for row_index in range(4)]
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
    - Ankunft und Abfahrt,
    - welche Zeitschritt-Indizes diese Zeiten im Simulationsraster haben,
    - Standdauer,
    - Ladezustand bei Ankunft(SoC),
    - Fahrzeugmodell und Fahrzeugklasse,
    - Tagtyp (z.B. Werktag/Samstag/Sonn- bzw. Feiertag)
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


def _get_day_type(simulation_day_start: datetime, holiday_dates: Set[date]) -> str:
    """
    Klassifiziert einen Simulationstag als "working_day", "saturday" oder "sunday_holiday".

    Regeln
    ------
    1) Wenn das Datum in ``holiday_dates`` enthalten ist, wird immer "sunday_holiday" zurückgegeben.
    2) Sonst gilt anhand des Wochentags:
       - Samstag  -> "saturday"
       - Sonntag  -> "sunday_holiday"
       - Mo–Fr    -> "working_day"

    Parameters
    ----------
    simulation_day_start:
        Startzeitpunkt des Tages (Datumsteil ist relevant).
    holiday_dates:
        Menge von Feiertagen als ``datetime.date`` (z. B. {date(2026, 1, 1), ...}).

    Returns
    -------
    str
        Einer der Werte: "working_day", "saturday", "sunday_holiday".
    """
    date_only = simulation_day_start.date()

    # Feiertag hat Priorität (wird wie Sonntag behandelt)
    if date_only in holiday_dates:
        return "sunday_holiday"

    weekday_index = int(simulation_day_start.weekday())  # 0=Mo .. 6=So
    if weekday_index == 5:
        return "saturday"
    if weekday_index == 6:
        return "sunday_holiday"
    return "working_day"


def _sample_vehicle_by_class(
    vehicle_curves_by_name: Dict[str, "VehicleChargingCurve"],
    fleet_mix: dict,
    random_generator: np.random.Generator,
    vehicle_type: Optional[Any] = None,
) -> Tuple[str, str]:
    """
    Wählt ein Fahrzeug für eine Ladesession.

    Standardverhalten
    -----------------
    - Klasse wird gemäß ``fleet_mix`` gewichtet gezogen (z. B. PKW/Transporter).
    - Innerhalb der gezogenen Klasse wird ein Fahrzeug gleichverteilt gewählt.

    Override über YAML: ``vehicles.vehicle_type``
    --------------------------------------------
    Wenn in der YAML unter ``vehicles`` der Parameter ``vehicle_type`` gesetzt ist,
    wird die Auswahl auf diese Fahrzeuge begrenzt.

    - Erlaubt sind bis zu 5 Fahrzeuge.
    - ``vehicle_type`` darf entweder ein String (ein Fahrzeug) oder eine Liste von Strings sein.
    - Die Namen müssen exakt den Schlüsseln in ``vehicle_curves_by_name`` entsprechen
      (also z. B. "Audi Q4 e-tron", so wie sie aus der Ladekurven-CSV erzeugt werden).

    Beispiel YAML
    -------------
    vehicles:
      fleet_mix:
        PKW: 0.98
        Transporter: 0.02
      vehicle_type:
        - "Audi Q4 e-tron"
        - "Fiat 500e Hatchback"

    Verhalten mit ``vehicle_type``
    ------------------------------
    - Es werden nur die angegebenen Fahrzeuge betrachtet.
    - Falls ``fleet_mix`` Klassen enthält, die innerhalb dieser Auswahl vorkommen, wird weiterhin
      nach Klassen gewichtet gezogen (aber nur innerhalb der eingeschränkten Fahrzeugliste).
    - Wenn ``fleet_mix`` nicht passt (keine Klassenüberschneidung), wird gleichverteilt aus den
      angegebenen Fahrzeugen gewählt.

    Parameters
    ----------
    vehicle_curves_by_name:
        Dict aus Ladekurven-Reader: {vehicle_name: VehicleChargingCurve}.
    fleet_mix:
        Dict mit Klassen-Gewichten, z. B. {"PKW": 0.98, "Transporter": 0.02}.
    random_generator:
        Numpy RNG.
    vehicle_type:
        Optionaler Override (aus YAML), String oder Liste[String]. Maximal 5 Fahrzeuge.

    Returns
    -------
    Tuple[str, str]
        (vehicle_name, vehicle_class)

    Raises
    ------
    ValueError
        Wenn keine Fahrzeuge verfügbar sind oder ``vehicle_type`` ungültige Fahrzeugnamen enthält.
    """

    def _parse_vehicle_type(value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = value
        else:
            raise ValueError("vehicles.vehicle_type muss ein String oder eine Liste von Strings sein.")

        cleaned: List[str] = []
        for item in items:
            name = str(item).strip()
            if name and name not in cleaned:
                cleaned.append(name)

        if len(cleaned) == 0:
            return None

        if len(cleaned) > 5:
            raise ValueError("vehicles.vehicle_type darf maximal 5 Fahrzeuge enthalten.")

        return cleaned

    # 1) Kandidatenmenge bestimmen (ggf. via vehicle_type einschränken)
    selected_vehicle_names = _parse_vehicle_type(vehicle_type)

    if selected_vehicle_names is not None:
        unknown = [name for name in selected_vehicle_names if name not in vehicle_curves_by_name]
        if unknown:
            available_preview = list(vehicle_curves_by_name.keys())[:15]
            raise ValueError(
                "vehicles.vehicle_type enthält Fahrzeugnamen, die nicht in den Ladekurven vorhanden sind: "
                f"{unknown}. Verfügbare Beispiele: {available_preview}"
            )

        candidate_curves = {name: vehicle_curves_by_name[name] for name in selected_vehicle_names}
    else:
        candidate_curves = vehicle_curves_by_name

    if len(candidate_curves) == 0:
        raise ValueError("Keine Fahrzeuge in vehicle_curves_by_name verfügbar.")

    # 2) Verfügbare Klassen aus Kandidaten ableiten
    available_classes = sorted({curve.vehicle_class for curve in candidate_curves.values()})
    if len(available_classes) == 0:
        raise ValueError("Keine Fahrzeugklassen in vehicle_curves_by_name gefunden.")

    # 3) Klassen nach fleet_mix gewichten (nur wenn fleet_mix zu verfügbaren Klassen passt)
    class_names_from_yaml = [class_name for class_name in fleet_mix.keys() if class_name in available_classes]
    if len(class_names_from_yaml) == 0:
        class_names = available_classes
        class_weights = np.ones(len(class_names), dtype=float)
    else:
        class_names = class_names_from_yaml
        class_weights = np.array([float(fleet_mix[class_name]) for class_name in class_names], dtype=float)

    class_weights = np.maximum(class_weights, 0.0)
    wsum = float(np.sum(class_weights))
    if wsum <= 0.0:
        class_weights = np.ones(len(class_names), dtype=float) / float(len(class_names))
    else:
        class_weights = class_weights / wsum

    chosen_class = str(random_generator.choice(class_names, p=class_weights))

    # 4) Fahrzeug innerhalb der Klasse wählen (gleichverteilt)
    vehicle_names_in_class = [
        curve.vehicle_name
        for curve in candidate_curves.values()
        if str(curve.vehicle_class).strip() == chosen_class
    ]
    if len(vehicle_names_in_class) == 0:
        # Sollte praktisch nicht passieren, aber als Sicherheitsnetz:
        vehicle_names_in_class = list(candidate_curves.keys())

    chosen_vehicle_name = str(random_generator.choice(vehicle_names_in_class))
    return chosen_vehicle_name, chosen_class


def sample_value_from_distribution(spec: Any, random_generator: np.random.Generator) -> float:
    """
    Zieht genau einen Sample-Wert aus einer Verteilungsspezifikation.

    Unterstützte Eingaben
    ---------------------
    0) Konstante:
       3.5  -> gibt 3.5 zurück

    1) Uniform-Range:
       [min, max]  -> uniform(min, max)

    2) Einzelkomponente (dict):
       {"distribution": "normal|beta|lognormal", ...}

    3) Mixture (dict):
       {"type": "mixture", "components": [ {distribution:..., weight:...}, ... ]}

    4) Mixture-Kurzform (list von dicts):
       [ {distribution:..., weight:...}, ... ]
    """
    # 0) Konstante
    if isinstance(spec, (int, float, np.number)):
        return float(spec)

    # 1) Uniform-Range [min, max] (nur wenn wirklich 2 Zahlen drinstehen!)
    if isinstance(spec, list) and len(spec) == 2 and all(isinstance(x, (int, float, np.number)) for x in spec):
        low, high = float(spec[0]), float(spec[1])
        if high < low:
            low, high = high, low
        return float(random_generator.uniform(low, high))

    # 4) Mixture-Kurzform: list von dicts
    if isinstance(spec, list):
        spec = {"type": "mixture", "components": spec}

    if not isinstance(spec, dict):
        raise ValueError(f"Distribution spec muss Zahl, list oder dict sein, bekommen: {type(spec)}")

    spec_type = str(spec.get("type", "")).strip().lower()

    # 3) Mixture
    if spec_type == "mixture":
        components = spec.get("components") or []
        if not isinstance(components, list) or len(components) == 0:
            raise ValueError("mixture benötigt eine nicht-leere Liste 'components'")

        if not all(isinstance(c, dict) for c in components):
            raise ValueError("mixture.components muss eine Liste von dicts sein")

        weights = np.array([float(c.get("weight", 1.0)) for c in components], dtype=float)
        weights = np.maximum(weights, 0.0)
        wsum = float(np.sum(weights))
        weights = (weights / wsum) if wsum > 0.0 else np.ones(len(components), dtype=float) / float(len(components))

        chosen_idx = int(random_generator.choice(len(components), p=weights))
        return sample_value_from_distribution(components[chosen_idx], random_generator)

    # 2) Einzelkomponente
    distribution_name = str(spec.get("distribution", "")).strip().lower()
    if distribution_name == "":
        raise ValueError("Einzelkomponente benötigt key 'distribution'")

    def _require(key: str) -> Any:
        if key not in spec:
            raise ValueError(f"Distribution '{distribution_name}' benötigt Parameter '{key}'")
        return spec[key]

    if distribution_name == "normal":
        mean_value = sample_value_from_distribution(_require("mu"), random_generator)
        standard_deviation = max(sample_value_from_distribution(_require("sigma"), random_generator), 1e-9)
        return float(random_generator.normal(loc=float(mean_value), scale=float(standard_deviation)))

    if distribution_name == "beta":
        alpha_value = max(sample_value_from_distribution(_require("alpha"), random_generator), 1e-9)
        beta_value = max(sample_value_from_distribution(_require("beta"), random_generator), 1e-9)
        return float(random_generator.beta(a=float(alpha_value), b=float(beta_value)))

    if distribution_name == "lognormal":
        mean_value = sample_value_from_distribution(_require("mu"), random_generator)
        standard_deviation = max(sample_value_from_distribution(_require("sigma"), random_generator), 1e-9)
        return float(random_generator.lognormal(mean=float(mean_value), sigma=float(standard_deviation)))

    raise ValueError(f"Unbekannte distribution in spec: '{distribution_name}'")


def sample_sessions_for_simulation_day(
    scenario: dict,
    simulation_day_start: datetime,
    timestamps: pd.DatetimeIndex,
    holiday_dates: Set[date],
    vehicle_curves_by_name: Dict[str, VehicleChargingCurve],
    random_generator: np.random.Generator,
    day_index: int,
) -> List[SampledSession]:
    """
    Erzeugt zufällige Ladesessions für genau einen Kalendertag der Simulation.

    Wichtige Konventionen
    ---------------------
    - `arrival_step` ist ein Index innerhalb des Tagesrasters (0 .. steps_per_day-1).
    - `departure_step` ist als *Ende exklusiv* gedacht (>= arrival_step+1).
      Beispiel: arrival_step=10, departure_step=12 -> Session belegt die Schritte 10 und 11.

    Der Session-Endzeitpunkt wird auf das Simulationsraster "gesnappt". Dadurch kann die
    ursprünglich gesampelte `duration_minutes` leicht angepasst werden; die gespeicherte
    `duration_minutes` wird deshalb aus (departure_time - arrival_time) neu berechnet.
    """

    def _sample_scalar_or_distribution(spec: Any, default: float) -> float:
        """
        Akzeptiert:
        - Zahl (int/float) -> direkt
        - dict/list -> sample_value_from_distribution
        - None -> default
        """
        if spec is None:
            return float(default)
        if isinstance(spec, (int, float, np.number)):
            return float(spec)
        if isinstance(spec, (dict, list)):
            return float(sample_value_from_distribution(spec, random_generator))
        # Optional: Strings, die Zahlen enthalten
        if isinstance(spec, str):
            try:
                return float(spec.strip())
            except Exception as exc:
                raise ValueError(f"Ungültige numerische Spezifikation: {spec}") from exc
        raise ValueError(f"Ungültige Spezifikation (erwartet Zahl|dict|list|None): {type(spec)}")

    time_resolution_min = int(scenario["time_resolution_min"])
    step_minutes = float(time_resolution_min)

    # Tag auf TZ der Simulation normalisieren
    day_start = normalize_to_timestamps_timezone(simulation_day_start, timestamps)
    day_key = day_start.normalize()

    day_mask = timestamps.normalize() == day_key
    if not np.any(day_mask):
        return []

    day_timestamps = timestamps[day_mask]
    steps_per_day = int(len(day_timestamps))
    if steps_per_day <= 0:
        return []

    day_start_abs_step = int(timestamps.get_loc(day_timestamps[0]))

    site_configuration = scenario["site"]
    number_chargers = int(site_configuration["number_chargers"])

    # --- Erwartete Sessions pro Charger ---
    expected_sessions_spec = site_configuration.get("expected_sessions_per_charger_per_day", 1.0)
    expected_sessions_per_charger = _sample_scalar_or_distribution(expected_sessions_spec, default=1.0)
    expected_sessions_per_charger = max(float(expected_sessions_per_charger), 0.0)

    # --- Tagtyp bestimmen ---
    day_type = _get_day_type(simulation_day_start, holiday_dates)

    # --- weekday_weight lesen (Zahl oder Verteilung) ---
    arrival_time_distribution = scenario.get("arrival_time_distribution", {}) or {}
    weekday_weight_table = arrival_time_distribution.get("weekday_weight", {}) or {}
    weekday_weight_spec = weekday_weight_table.get(day_type, 1.0)
    weekday_weight = _sample_scalar_or_distribution(weekday_weight_spec, default=1.0)
    weekday_weight = max(float(weekday_weight), 0.0)

    expected_total_sessions = expected_sessions_per_charger * float(number_chargers) * float(weekday_weight)
    expected_total_sessions = max(float(expected_total_sessions), 0.0)
    number_sessions = int(random_generator.poisson(lam=expected_total_sessions))

    # --- Arrival-Spec für den Tagtyp ---
    components_per_weekday = arrival_time_distribution.get("components_per_weekday", {}) or {}
    arrival_spec = components_per_weekday.get(day_type)

    if number_sessions > 0 and not isinstance(arrival_spec, (list, dict)):
        raise ValueError(
            f"arrival_time_distribution.components_per_weekday['{day_type}'] muss list oder dict sein."
        )

    # --- Parkdauer-Spec ---
    parking_duration_distribution = scenario.get("parking_duration_distribution", {}) or {}
    min_duration_minutes = float(parking_duration_distribution.get("min_duration_minutes", 0.0))
    max_duration_minutes = float(parking_duration_distribution.get("max_duration_minutes", 24 * 60))
    if max_duration_minutes < min_duration_minutes:
        max_duration_minutes = min_duration_minutes

    parking_spec = parking_duration_distribution.get("components", parking_duration_distribution.get("spec"))
    if number_sessions > 0 and not isinstance(parking_spec, (list, dict)):
        raise ValueError("parking_duration_distribution benötigt 'components' (list) oder 'spec' (dict).")

    # --- SoC-at-arrival-Spec ---
    state_of_charge_distribution = scenario.get("soc_at_arrival_distribution", {}) or {}
    max_state_of_charge = float(state_of_charge_distribution.get("max_soc", 1.0))
    max_state_of_charge = float(np.clip(max_state_of_charge, 0.0, 1.0))

    soc_spec = state_of_charge_distribution.get("components", state_of_charge_distribution.get("spec"))
    if number_sessions > 0 and not isinstance(soc_spec, (list, dict)):
        raise ValueError("soc_at_arrival_distribution benötigt 'components' (list) oder 'spec' (dict).")

    vehicles_section = scenario.get("vehicles", {}) or {}
    fleet_mix = vehicles_section.get("fleet_mix", {}) or {}

    allow_cross_day_charging = bool(site_configuration.get("allow_cross_day_charging", False))

    sampled_sessions: List[SampledSession] = []

    for session_number in range(number_sessions):
        # ---------- Arrival ----------
        arrival_hours = float(sample_value_from_distribution(arrival_spec, random_generator))
        arrival_minutes = float(np.clip(arrival_hours * 60.0, 0.0, 24.0 * 60.0 - 1e-9))

        arrival_step = int(np.floor(arrival_minutes / step_minutes))
        arrival_step = int(np.clip(arrival_step, 0, steps_per_day - 1))
        arrival_abs_step = day_start_abs_step + arrival_step
        arrival_time = pd.to_datetime(timestamps[arrival_abs_step]).to_pydatetime()

        # ---------- Duration / Departure (step-basiert) ----------
        raw_duration_minutes = float(sample_value_from_distribution(parking_spec, random_generator))
        raw_duration_minutes = float(np.clip(raw_duration_minutes, min_duration_minutes, max_duration_minutes))

        duration_steps = int(np.ceil(raw_duration_minutes / step_minutes))
        duration_steps = max(duration_steps, 1)

        departure_abs_step = arrival_abs_step + duration_steps
        departure_step = arrival_step + duration_steps  # kann > steps_per_day sein, wenn cross-day erlaubt

        if not allow_cross_day_charging:
            # Ende exklusiv: maximal bis steps_per_day
            departure_step = int(min(departure_step, steps_per_day))
            departure_abs_step = day_start_abs_step + departure_step

        # Horizon-Clip
        if departure_abs_step < len(timestamps):
            departure_time = pd.to_datetime(timestamps[departure_abs_step]).to_pydatetime()
        else:
            departure_time = (pd.to_datetime(timestamps[-1]) + pd.Timedelta(minutes=time_resolution_min)).to_pydatetime()

        # Jetzt Dauer konsistent aus gerasterten Zeiten ableiten
        duration_minutes = float((departure_time - arrival_time).total_seconds() / 60.0)

        # ---------- SoC ----------
        sampled_soc = float(sample_value_from_distribution(soc_spec, random_generator))
        state_of_charge_at_arrival = float(np.clip(sampled_soc, 0.0, max_state_of_charge))

        # ---------- Vehicle ----------
        vehicle_name, vehicle_class = _sample_vehicle_by_class(
            vehicle_curves_by_name=vehicle_curves_by_name,
            fleet_mix=fleet_mix,
            random_generator=random_generator,
            vehicle_type=vehicles_section.get("vehicle_type"),
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

    sampled_sessions.sort(key=lambda s: s.arrival_time)
    return sampled_sessions


# =============================================================================
# 3) Strategie: Reservierungsbasierte Session-Planung (immediate / market / generation)
# =============================================================================

def max_available_power_for_session(
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
    Maximale allokierbare Energie (kWh/step) für eine Session in einem Zeitschritt, inkl. PV-Anteil.

    Parameter
    ---------
    step_index:
        Absoluter Simulationsschritt-Index.
    scenario:
        Szenario-Dict. Erwartet: site.rated_power_kw, site.grid_limit_p_avb_kw, site.charger_efficiency, time_resolution_min.
    curve:
        Fahrzeug-Ladekurve (power_kw auf Batterieseite).
    state_of_charge_fraction:
        SoC zu Beginn des Schritts [0..1] (zeitkonsistent berechnet).
    remaining_site_energy_kwh:
        Noch benötigte Site-Energie bis Ziel-SoC (kWh). Begrenzt die Allokation.
    pv_generation_kwh_per_step, base_load_kwh_per_step:
        Zeitreihen (kWh/step).
    reserved_total_ev_energy_kwh_per_step:
        Bereits reservierte/zugewiesene EV-Energie (Site-seitig) über alle Sessions (kWh/step).
    reserved_pv_ev_energy_kwh_per_step:
        Bereits als PV→EV getrackte Energie (kWh/step). Optional: wenn None, wird PV-Share = 0 angenommen.
    already_allocated_on_this_charger_kwh:
        Bereits für diese Session/Ladepunkt im selben Schritt allokierte Energie (z.B. PV-Pass1 -> Pass2).
    supply_mode:
        - "site": PV + Grid (NAP) verfügbar
        - "pv_only": nur PV nach Grundlast und nach PV-Reservierungen (zusätzlich durch Gesamtsite-Headroom begrenzt)
        - "grid_only": nur Grid (NAP) verfügbar, physikalisch unter Berücksichtigung des bereits getrackten PV-Anteils

    Returns
    -------
    (allocated_site_kwh, pv_share_kwh)
        allocated_site_kwh: maximal allokierbar in diesem Schritt (kWh/step), >= 0
        pv_share_kwh: PV-Anteil dieser Allokation gemäß "PV physikalisch zuerst", >= 0
    """
    if step_index < 0 or step_index >= len(base_load_kwh_per_step):
        raise IndexError(f"step_index={step_index} außerhalb der Zeitreihenlänge.")

    site_cfg = scenario["site"]
    time_resolution_min = int(scenario["time_resolution_min"])
    step_hours = float(time_resolution_min) / 60.0

    charger_efficiency = float(site_cfg.get("charger_efficiency", 1.0))  # kommt aus read_scenario_from_yaml
    if not (0.0 < charger_efficiency <= 1.0):
        raise ValueError("site.charger_efficiency muss im Bereich (0, 1] liegen.")

    # --- Basiswerte ---
    pv_kwh = float(pv_generation_kwh_per_step[step_index])
    nap_kwh = float(site_cfg["grid_limit_p_avb_kw"]) * step_hours  # NAP(Grid-Limit) in kWh/step
    base_kwh = float(base_load_kwh_per_step[step_index])

    reserved_total_kwh = float(reserved_total_ev_energy_kwh_per_step[step_index])
    reserved_pv_kwh = float(reserved_pv_ev_energy_kwh_per_step[step_index]) if reserved_pv_ev_energy_kwh_per_step is not None else 0.0
    reserved_pv_kwh = float(max(reserved_pv_kwh, 0.0))

    # --- (A) Gesamtsite-Headroom: PV + Grid - Grundlast - reservierte EV-Energie ---
    site_headroom_kwh = pv_kwh + nap_kwh - base_kwh - reserved_total_kwh
    site_headroom_kwh = float(max(site_headroom_kwh, 0.0))

    # --- (B) PV-Remaining für EV (PV nach Grundlast, nach bereits getrackter PV->EV-Reservierung) ---
    pv_after_base_kwh = float(max(pv_kwh - base_kwh, 0.0))
    pv_remaining_for_ev_kwh = float(max(pv_after_base_kwh - reserved_pv_kwh, 0.0))

    # --- (C) Grid-Headroom (nur Grid), physikalisch: Grid deckt Base nach PV + EV-Anteil, der nicht PV ist ---
    # Base-Grid-Anteil:
    base_grid_kwh = float(max(base_kwh - pv_kwh, 0.0))
    # EV-Grid-Anteil (Site-EV minus getrackter PV->EV-Anteil):
    ev_grid_kwh = float(max(reserved_total_kwh - reserved_pv_kwh, 0.0))
    grid_headroom_kwh = nap_kwh - (base_grid_kwh + ev_grid_kwh)
    grid_headroom_kwh = float(max(grid_headroom_kwh, 0.0))

    # --- Angebot je nach Modus ---
    if supply_mode == "site":
        supply_headroom_kwh = site_headroom_kwh
    elif supply_mode == "pv_only":
        # PV-only darf nicht mehr als PV-Remaining liefern und darf das Gesamtsite-Limit nicht verletzen
        supply_headroom_kwh = float(min(pv_remaining_for_ev_kwh, site_headroom_kwh))
    elif supply_mode == "grid_only":
        supply_headroom_kwh = grid_headroom_kwh
    else:
        raise ValueError(f"Unbekannter supply_mode='{supply_mode}' (erlaubt: site|pv_only|grid_only)")

    # --- (D) Ladepunktlimit pro Session/Ladepunkt ---
    charger_limit_kwh = float(site_cfg["rated_power_kw"]) * step_hours
    charger_headroom_kwh = charger_limit_kwh - float(already_allocated_on_this_charger_kwh)
    charger_headroom_kwh = float(max(charger_headroom_kwh, 0.0))

    # --- (E) Fahrzeuglimit aus SoC-Kurve (Batterieseite -> Site-Seite) ---
    soc = float(np.clip(state_of_charge_fraction, 0.0, 1.0))
    battery_power_kw_allowed = float(np.interp(soc, curve.state_of_charge_fraction, curve.power_kw))
    battery_power_kw_allowed = float(max(battery_power_kw_allowed, 0.0))

    site_power_kw_allowed = battery_power_kw_allowed / max(charger_efficiency, 1e-12)
    vehicle_site_limit_kwh = float(max(site_power_kw_allowed * step_hours, 0.0))

    # --- (F) Restenergiebedarf ---
    remaining_site_energy_kwh = float(max(remaining_site_energy_kwh, 0.0))

    allocated_site_kwh = float(
        np.minimum.reduce(
            np.array(
                [
                    supply_headroom_kwh,
                    charger_headroom_kwh,
                    vehicle_site_limit_kwh,
                    remaining_site_energy_kwh,
                ],
                dtype=float,
            )
        )
    )
    allocated_site_kwh = float(max(allocated_site_kwh, 0.0))

    # --- PV-Share der Allokation ---
    if allocated_site_kwh <= 0.0:
        return 0.0, 0.0

    if supply_mode == "grid_only":
        pv_share_kwh = 0.0
    elif supply_mode == "pv_only":
        pv_share_kwh = allocated_site_kwh
    else:
        # "site": PV physikalisch zuerst
        pv_share_kwh = float(min(allocated_site_kwh, pv_remaining_for_ev_kwh))

    pv_share_kwh = float(max(pv_share_kwh, 0.0))
    return allocated_site_kwh, pv_share_kwh


def plan_charging_immediate(
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
    Immediate-Strategie (sofortiges Laden innerhalb des Session-Fensters).

    Es wird chronologisch von Ankunft bis Abfahrt (Ende exklusiv) iteriert. Pro Schritt wird
    über ``max_available_power_for_session(...)`` die maximal allokierbare Energie (kWh/step)
    bestimmt und die globalen Reservierungen werden IN-PLACE aktualisiert.

    Einheitliche Rückgabe
    ---------------------
    Die Rückgabe ist strukturell kompatibel zu den Tag-Planern (market/pv) und enthält immer:
    - KPI-Felder:
        - ``required_site_kwh``, ``required_battery_kwh``
        - ``charged_site_kwh``, ``charged_pv_site_kwh``
        - ``charged_market_site_kwh`` (bei immediate = 0)
        - ``charged_immediate_site_kwh`` (bei immediate = charged_site_kwh)
        - ``remaining_site_kwh``, ``final_soc``
    - Plan-Arrays:
        - ``plan_site_kwh_per_step`` (gesamt)
        - ``plan_pv_site_kwh_per_step`` (PV-Anteil)
        - ``plan_market_site_kwh_per_step`` (bei immediate = 0)
        - ``plan_immediate_site_kwh_per_step`` (bei immediate = plan_site)

    Parameters
    ----------
    session_arrival_step:
        Absoluter Ankunftsindex im Simulationsraster.
    session_departure_step:
        Absoluter Abfahrtsindex im Simulationsraster (Ende exklusiv).
    required_site_energy_kwh:
        Angeforderte Energie auf Standortseite (kWh), die in dieser Session geladen werden soll.
    pv_generation_kwh_per_step, base_load_kwh_per_step:
        Zeitreihen in kWh pro Simulationsschritt.
    reserved_total_ev_energy_kwh_per_step, reserved_pv_ev_energy_kwh_per_step:
        Globale Reservierungs-Arrays (kWh/step), werden IN-PLACE aktualisiert.
    curve:
        Fahrzeug-Ladekurve.
    state_of_charge_at_arrival:
        SoC bei Ankunft als Anteil [0..1].
    scenario:
        Szenario-Dictionary.

    Returns
    -------
    Dict[str, Any]
        Einheitliche Struktur mit Plänen, KPI-Werten und finalem SoC.
    """
    number_steps_total = int(len(reserved_total_ev_energy_kwh_per_step))
    for name, arr in [
        ("pv_generation_kwh_per_step", pv_generation_kwh_per_step),
        ("base_load_kwh_per_step", base_load_kwh_per_step),
        ("reserved_pv_ev_energy_kwh_per_step", reserved_pv_ev_energy_kwh_per_step),
    ]:
        if len(arr) != number_steps_total:
            raise ValueError(
                f"{name} muss die gleiche Länge haben wie "
                "reserved_total_ev_energy_kwh_per_step."
            )

    start_step_index = int(max(0, session_arrival_step))
    end_step_index_exclusive = int(min(int(session_departure_step), number_steps_total))
    window_length = int(max(0, end_step_index_exclusive - start_step_index))

    site_configuration = scenario["site"]
    charger_efficiency = float(site_configuration.get("charger_efficiency", 1.0))
    if not (0.0 < charger_efficiency <= 1.0):
        raise ValueError("site.charger_efficiency muss im Bereich (0, 1] liegen.")

    required_site_energy_kwh = float(max(required_site_energy_kwh, 0.0))
    required_battery_energy_kwh = float(required_site_energy_kwh * charger_efficiency)

    plan_site_kwh_per_step = np.zeros(window_length, dtype=float)
    plan_pv_site_kwh_per_step = np.zeros(window_length, dtype=float)
    plan_market_site_kwh_per_step = np.zeros(window_length, dtype=float)
    plan_immediate_site_kwh_per_step = np.zeros(window_length, dtype=float)

    if window_length == 0 or required_site_energy_kwh <= 1e-12:
        final_soc = float(np.clip(state_of_charge_at_arrival, 0.0, 1.0))
        return {
            "required_site_kwh": required_site_energy_kwh,
            "required_battery_kwh": required_battery_energy_kwh,
            "charged_site_kwh": 0.0,
            "charged_pv_site_kwh": 0.0,
            "charged_market_site_kwh": 0.0,
            "charged_immediate_site_kwh": 0.0,
            "remaining_site_kwh": required_site_energy_kwh,
            "final_soc": final_soc,
            "plan_site_kwh_per_step": plan_site_kwh_per_step,
            "plan_pv_site_kwh_per_step": plan_pv_site_kwh_per_step,
            "plan_market_site_kwh_per_step": plan_market_site_kwh_per_step,
            "plan_immediate_site_kwh_per_step": plan_immediate_site_kwh_per_step,
        }

    battery_capacity_kwh = float(max(curve.battery_capacity_kwh, 1e-12))

    state_of_charge_fraction = float(np.clip(state_of_charge_at_arrival, 0.0, 1.0))
    remaining_site_energy_kwh = float(required_site_energy_kwh)

    charged_site_energy_kwh = 0.0
    charged_pv_site_energy_kwh = 0.0

    for absolute_step_index in range(start_step_index, end_step_index_exclusive):
        if remaining_site_energy_kwh <= 1e-12 or state_of_charge_fraction >= 1.0 - 1e-12:
            break

        allocated_site_kwh, pv_share_kwh = max_available_power_for_session(
            step_index=int(absolute_step_index),
            scenario=scenario,
            curve=curve,
            state_of_charge_fraction=float(state_of_charge_fraction),
            remaining_site_energy_kwh=float(remaining_site_energy_kwh),
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
            reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
            already_allocated_on_this_charger_kwh=0.0,
            supply_mode="site",
        )

        allocated_site_kwh = float(max(allocated_site_kwh, 0.0))
        if allocated_site_kwh <= 1e-12:
            continue

        pv_share_kwh = float(np.clip(float(pv_share_kwh), 0.0, allocated_site_kwh))

        local_index = int(absolute_step_index - start_step_index)
        plan_site_kwh_per_step[local_index] = allocated_site_kwh
        plan_pv_site_kwh_per_step[local_index] = pv_share_kwh

        reserved_total_ev_energy_kwh_per_step[int(absolute_step_index)] += allocated_site_kwh
        reserved_pv_ev_energy_kwh_per_step[int(absolute_step_index)] += pv_share_kwh

        charged_site_energy_kwh += allocated_site_kwh
        charged_pv_site_energy_kwh += pv_share_kwh
        remaining_site_energy_kwh -= allocated_site_kwh

        battery_energy_kwh = allocated_site_kwh * charger_efficiency
        state_of_charge_fraction = float(
            min(1.0, state_of_charge_fraction + battery_energy_kwh / battery_capacity_kwh)
        )

    plan_immediate_site_kwh_per_step[:] = plan_site_kwh_per_step

    charged_site_energy_kwh = float(charged_site_energy_kwh)
    charged_pv_site_energy_kwh = float(charged_pv_site_energy_kwh)
    remaining_site_energy_kwh = float(max(remaining_site_energy_kwh, 0.0))

    return {
        "required_site_kwh": required_site_energy_kwh,
        "required_battery_kwh": required_battery_energy_kwh,
        "charged_site_kwh": charged_site_energy_kwh,
        "charged_pv_site_kwh": charged_pv_site_energy_kwh,
        "charged_market_site_kwh": 0.0,
        "charged_immediate_site_kwh": charged_site_energy_kwh,
        "remaining_site_kwh": remaining_site_energy_kwh,
        "final_soc": float(state_of_charge_fraction),
        "plan_site_kwh_per_step": plan_site_kwh_per_step,
        "plan_pv_site_kwh_per_step": plan_pv_site_kwh_per_step,
        "plan_market_site_kwh_per_step": plan_market_site_kwh_per_step,
        "plan_immediate_site_kwh_per_step": plan_immediate_site_kwh_per_step,
    }


def plan_charging_market_for_day(
    sessions: List["SampledSession"],
    vehicle_curves_by_name: Dict[str, "VehicleChargingCurve"],
    scenario: dict,
    market_price_eur_per_mwh: np.ndarray,
    pv_generation_kwh_per_step: np.ndarray,
    base_load_kwh_per_step: np.ndarray,
    reserved_total_ev_energy_kwh_per_step: np.ndarray,
    reserved_pv_ev_energy_kwh_per_step: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Plant Ladevorgänge für alle Sessions eines Tages nach Marktpreis-Ranking und liefert
    die gleiche Ergebnis- und Plan-Struktur wie die Planner `pv` und `immediate`.

    Strategie
    ---------
    1) Sessions werden nach aufsteigender Standdauer geplant (kurz zuerst).
    2) Innerhalb des Session-Fensters werden Steps nach Marktpreis (€/MWh) gerankt.
    3) Der Ladeprozess läuft chronologisch (SoC-konsistent), es wird aber nur in den
       "ausgewählten" günstigen Steps geladen – außer wenn "must-charge" greift.

    Preise
    ------
    - `market_price_eur_per_mwh` wird ausschließlich zur Reihenfolge/Slot-Auswahl genutzt
      (keine Umrechnung nach €/kWh).

    Ergebnisstruktur (vereinheitlicht)
    ----------------------------------
    Pro Session werden die vereinheitlichten Felder zurückgegeben:
    - required_site_kwh, required_battery_kwh
    - charged_site_kwh, charged_pv_site_kwh, charged_market_site_kwh, charged_immediate_site_kwh
    - remaining_site_kwh, final_soc
    - Plan-Arrays:
        plan_site_kwh_per_step           : Gesamtenergie am Standort (PV + Grid) in kWh/Step
        plan_pv_site_kwh_per_step        : PV-Anteil der Standortenergie in kWh/Step
        plan_market_site_kwh_per_step    : in dieser Strategie identisch zu plan_site (Market-Plan)
        plan_immediate_site_kwh_per_step : in dieser Strategie immer 0

    Hinweise
    --------
    - `reserved_total_ev_energy_kwh_per_step` und `reserved_pv_ev_energy_kwh_per_step` werden
      IN-PLACE aktualisiert (globale Kopplung über Sessions).
    - Indizes sind absolute Steps im Simulationsraster und müssen zu den Eingabe-Arrays passen.
    """
    if market_price_eur_per_mwh is None:
        raise ValueError(
            "Für charging_strategy='market' muss market_price_eur_per_mwh vorhanden sein."
        )

    n_total = int(len(reserved_total_ev_energy_kwh_per_step))
    for arr_name, arr in [
        ("pv_generation_kwh_per_step", pv_generation_kwh_per_step),
        ("base_load_kwh_per_step", base_load_kwh_per_step),
        ("reserved_pv_ev_energy_kwh_per_step", reserved_pv_ev_energy_kwh_per_step),
        ("market_price_eur_per_mwh", market_price_eur_per_mwh),
    ]:
        if len(arr) != n_total:
            raise ValueError(
                f"{arr_name} muss die gleiche Länge haben wie "
                "reserved_total_ev_energy_kwh_per_step."
            )

    time_resolution_min = int(scenario["time_resolution_min"])
    step_hours = float(time_resolution_min) / 60.0

    site_cfg = scenario["site"]
    vehicles_cfg = scenario["vehicles"]

    charger_efficiency = float(site_cfg.get("charger_efficiency", 1.0))
    if not (0.0 < charger_efficiency <= 1.0):
        raise ValueError("site.charger_efficiency muss im Bereich (0, 1] liegen.")

    target_soc = float(np.clip(float(vehicles_cfg.get("soc_target", 1.0)), 0.0, 1.0))

    rated_power_kw = float(site_cfg["rated_power_kw"])
    charger_cap_kwh_per_step = max(rated_power_kw * step_hours, 0.0)

    grid_limit_kw = float(site_cfg.get("grid_limit_p_avb_kw", 0.0))
    grid_limit_kwh_per_step = max(grid_limit_kw * step_hours, 0.0)

    def _duration_steps(session: "SampledSession") -> int:
        """Gibt die Standdauer einer Session in Simulationsschritten zurück (Ende exklusiv)."""
        return int(max(0, int(session.departure_step) - int(session.arrival_step)))

    sessions_sorted = sorted(
        sessions,
        key=lambda s: (_duration_steps(s), int(s.arrival_step), str(s.session_id)),
    )

    results: List[Dict[str, Any]] = []

    for session in sessions_sorted:
        start = int(max(0, session.arrival_step))
        end = int(min(int(session.departure_step), n_total))  # Ende exklusiv
        window_len = int(max(0, end - start))

        curve = vehicle_curves_by_name.get(session.vehicle_name)
        if curve is None:
            raise ValueError(
                f"Keine Ladekurve für vehicle_name='{session.vehicle_name}' gefunden."
            )

        battery_capacity_kwh = max(float(curve.battery_capacity_kwh), 1e-12)

        soc0 = float(np.clip(session.state_of_charge_at_arrival, 0.0, 1.0))
        needed_battery_kwh = max(0.0, (target_soc - soc0) * battery_capacity_kwh)
        needed_site_kwh = needed_battery_kwh / max(charger_efficiency, 1e-12)

        remaining_site_kwh = float(max(needed_site_kwh, 0.0))
        soc = float(soc0)

        plan_site_kwh_per_step = np.zeros(window_len, dtype=float)
        plan_pv_site_kwh_per_step = np.zeros(window_len, dtype=float)
        plan_market_site_kwh_per_step = np.zeros(window_len, dtype=float)
        plan_immediate_site_kwh_per_step = np.zeros(window_len, dtype=float)

        charged_site_kwh = 0.0
        charged_pv_site_kwh = 0.0

        if window_len <= 0 or remaining_site_kwh <= 1e-12:
            results.append(
                {
                    "session_id": session.session_id,
                    "vehicle_name": session.vehicle_name,
                    "vehicle_class": session.vehicle_class,
                    "arrival_step": int(session.arrival_step),
                    "departure_step": int(session.departure_step),
                    "required_site_kwh": float(needed_site_kwh),
                    "required_battery_kwh": float(needed_battery_kwh),
                    "charged_site_kwh": 0.0,
                    "charged_pv_site_kwh": 0.0,
                    "charged_market_site_kwh": 0.0,
                    "charged_immediate_site_kwh": 0.0,
                    "remaining_site_kwh": float(remaining_site_kwh),
                    "final_soc": float(soc),
                    "plan_site_kwh_per_step": plan_site_kwh_per_step,
                    "plan_pv_site_kwh_per_step": plan_pv_site_kwh_per_step,
                    "plan_market_site_kwh_per_step": plan_market_site_kwh_per_step,
                    "plan_immediate_site_kwh_per_step": plan_immediate_site_kwh_per_step,
                }
            )
            continue

        session_steps = np.arange(start, end, dtype=int)

        # Optimistische Cap (ohne SoC), nur für Preferred-Step-Auswahl + must-charge.
        supply_headroom = (
            pv_generation_kwh_per_step[session_steps].astype(float)
            + float(grid_limit_kwh_per_step)
            - base_load_kwh_per_step[session_steps].astype(float)
            - reserved_total_ev_energy_kwh_per_step[session_steps].astype(float)
        )
        supply_headroom = np.maximum(supply_headroom, 0.0)

        cap_est = np.minimum(supply_headroom, float(charger_cap_kwh_per_step))
        cap_est = np.maximum(cap_est, 0.0)

        prices_window = market_price_eur_per_mwh[session_steps].astype(float)
        order_by_price = np.argsort(prices_window)  # cheapest first

        preferred = np.zeros(window_len, dtype=bool)
        cap_acc = 0.0
        for idx in order_by_price:
            preferred[idx] = True
            cap_acc += float(cap_est[idx])
            if cap_acc >= remaining_site_kwh - 1e-9:
                break

        if float(np.sum(cap_est)) < remaining_site_kwh - 1e-9:
            preferred[:] = True

        suffix_cap = np.zeros(window_len + 1, dtype=float)
        for i in range(window_len - 1, -1, -1):
            suffix_cap[i] = suffix_cap[i + 1] + float(cap_est[i])

        for local_i, abs_step in enumerate(range(start, end)):
            if remaining_site_kwh <= 1e-12 or soc >= 1.0 - 1e-12:
                break

            future_possible = float(suffix_cap[local_i + 1])
            must_charge = remaining_site_kwh > future_possible + 1e-9

            if not must_charge and not bool(preferred[local_i]):
                continue

            allocated_site_kwh, pv_share_kwh = max_available_power_for_session(
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

            allocated_site_kwh = float(max(allocated_site_kwh, 0.0))
            if allocated_site_kwh <= 1e-12:
                continue

            pv_share_kwh = float(np.clip(float(pv_share_kwh), 0.0, allocated_site_kwh))

            plan_site_kwh_per_step[local_i] += allocated_site_kwh
            plan_pv_site_kwh_per_step[local_i] += pv_share_kwh
            plan_market_site_kwh_per_step[local_i] += allocated_site_kwh

            reserved_total_ev_energy_kwh_per_step[int(abs_step)] += allocated_site_kwh
            reserved_pv_ev_energy_kwh_per_step[int(abs_step)] += pv_share_kwh

            charged_site_kwh += allocated_site_kwh
            charged_pv_site_kwh += pv_share_kwh
            remaining_site_kwh -= allocated_site_kwh

            battery_energy_kwh = allocated_site_kwh * charger_efficiency
            soc = float(min(1.0, soc + battery_energy_kwh / battery_capacity_kwh))

        results.append(
            {
                "session_id": session.session_id,
                "vehicle_name": session.vehicle_name,
                "vehicle_class": session.vehicle_class,
                "arrival_step": int(session.arrival_step),
                "departure_step": int(session.departure_step),
                "required_site_kwh": float(needed_site_kwh),
                "required_battery_kwh": float(needed_battery_kwh),
                "charged_site_kwh": float(charged_site_kwh),
                "charged_pv_site_kwh": float(charged_pv_site_kwh),
                "charged_market_site_kwh": float(charged_site_kwh),
                "charged_immediate_site_kwh": 0.0,
                "remaining_site_kwh": float(max(remaining_site_kwh, 0.0)),
                "final_soc": float(soc),
                "plan_site_kwh_per_step": plan_site_kwh_per_step,
                "plan_pv_site_kwh_per_step": plan_pv_site_kwh_per_step,
                "plan_market_site_kwh_per_step": plan_market_site_kwh_per_step,
                "plan_immediate_site_kwh_per_step": plan_immediate_site_kwh_per_step,
            }
        )

    return results


def plan_charging_pv_for_day(
    sessions: List["SampledSession"],
    vehicle_curves_by_name: Dict[str, "VehicleChargingCurve"],
    scenario: dict,
    market_price_eur_per_mwh: np.ndarray,
    pv_generation_kwh_per_step: np.ndarray,
    base_load_kwh_per_step: np.ndarray,
    reserved_total_ev_energy_kwh_per_step: np.ndarray,
    reserved_pv_ev_energy_kwh_per_step: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Plant PV-first-Ladevorgänge für alle Sessions eines Tages (Standdauer-Priorität) und liefert
    die gleiche Ergebnis- und Plan-Struktur wie die Planner `market` und `immediate`.

    Strategie
    ---------
    1) Sessions werden nach aufsteigender Standdauer geplant (kurz zuerst).
    2) Innerhalb des Session-Fensters (chronologisch, SoC-konsistent):
       - Zuerst PV-only laden (`supply_mode="pv_only"`), solange PV verfügbar ist.
       - Reicht PV nicht aus, erfolgt ein Grid-/NAP-Fallback nur in bevorzugten günstigen Slots,
         die über `market_price_eur_per_mwh` (reines Ranking) bestimmt werden, plus must-charge-
         Sicherheitslogik.

    Ergebnisstruktur (vereinheitlicht)
    ----------------------------------
    Pro Session werden die vereinheitlichten Felder zurückgegeben:
    - required_site_kwh, required_battery_kwh
    - charged_site_kwh, charged_pv_site_kwh, charged_market_site_kwh, charged_immediate_site_kwh
    - remaining_site_kwh, final_soc
    - Plan-Arrays:
        plan_site_kwh_per_step           : Gesamtenergie am Standort (PV + Grid) in kWh/Step
        plan_pv_site_kwh_per_step        : PV-Anteil der Standortenergie in kWh/Step
        plan_market_site_kwh_per_step    : Grid-Energie aus Market-Fallback in kWh/Step
        plan_immediate_site_kwh_per_step : in dieser Strategie immer 0

    Hinweise
    --------
    - `market_price_eur_per_mwh` wird ausschließlich zur Sortierung/Slot-Auswahl genutzt (keine Umrechnung).
    - `reserved_total_ev_energy_kwh_per_step` und `reserved_pv_ev_energy_kwh_per_step` werden IN-PLACE
      aktualisiert (globale Kopplung über Sessions).
    - Indizes sind absolute Steps im Simulationsraster und müssen zu den Eingabe-Arrays passen.

    Parameters
    ----------
    sessions:
        Liste der Sessions, die innerhalb eines Tages geplant werden sollen.
    vehicle_curves_by_name:
        Mapping von `vehicle_name` auf eine Ladekurve inkl. Batteriekapazität.
    scenario:
        Szenario-Konfiguration (Zeitauflösung, Standort- und Fahrzeugparameter).
    market_price_eur_per_mwh:
        Marktpreis in €/MWh pro Simulationsschritt (nur zur Rangbildung für Grid-Fallback).
    pv_generation_kwh_per_step:
        PV-Erzeugung in kWh pro Simulationsschritt.
    base_load_kwh_per_step:
        Grundlast in kWh pro Simulationsschritt.
    reserved_total_ev_energy_kwh_per_step:
        Bereits reservierte EV-Gesamtenergie (PV + Grid) in kWh pro Simulationsschritt (wird IN-PLACE
        aktualisiert).
    reserved_pv_ev_energy_kwh_per_step:
        Bereits reservierte PV->EV Energie in kWh pro Simulationsschritt (wird IN-PLACE aktualisiert).

    Returns
    -------
    List[Dict[str, Any]]
        Liste von Session-Ergebnisdicts mit vereinheitlichter Feld- und Plan-Struktur.
    """
    if market_price_eur_per_mwh is None:
        raise ValueError(
            "Für charging_strategy='pv' muss market_price_eur_per_mwh vorhanden sein "
            "(nur für Fallback-Ranking)."
        )

    n_total = int(len(reserved_total_ev_energy_kwh_per_step))
    for name, arr in [
        ("pv_generation_kwh_per_step", pv_generation_kwh_per_step),
        ("base_load_kwh_per_step", base_load_kwh_per_step),
        ("reserved_pv_ev_energy_kwh_per_step", reserved_pv_ev_energy_kwh_per_step),
        ("market_price_eur_per_mwh", market_price_eur_per_mwh),
    ]:
        if len(arr) != n_total:
            raise ValueError(
                f"{name} muss die gleiche Länge haben wie reserved_total_ev_energy_kwh_per_step."
            )

    time_resolution_min = int(scenario["time_resolution_min"])
    step_hours = float(time_resolution_min) / 60.0

    site_cfg = scenario["site"]
    vehicles_cfg = scenario["vehicles"]

    charger_efficiency = float(site_cfg.get("charger_efficiency", 1.0))
    if not (0.0 < charger_efficiency <= 1.0):
        raise ValueError("site.charger_efficiency muss im Bereich (0, 1] liegen.")

    target_soc = float(np.clip(float(vehicles_cfg.get("soc_target", 1.0)), 0.0, 1.0))

    rated_power_kw = float(site_cfg["rated_power_kw"])
    charger_cap_kwh_per_step = max(rated_power_kw * step_hours, 0.0)

    grid_limit_kw = float(site_cfg.get("grid_limit_p_avb_kw", 0.0))
    grid_limit_kwh_per_step = max(grid_limit_kw * step_hours, 0.0)

    def _duration_steps(session: "SampledSession") -> int:
        """Gibt die Standdauer einer Session in Simulationsschritten zurück (Ende exklusiv)."""
        return int(max(0, int(session.departure_step) - int(session.arrival_step)))

    sessions_sorted = sorted(
        sessions,
        key=lambda s: (_duration_steps(s), int(s.arrival_step), str(s.session_id)),
    )

    results: List[Dict[str, Any]] = []

    for session in sessions_sorted:
        start = int(max(0, int(session.arrival_step)))
        end = int(min(int(session.departure_step), n_total))  # end exclusive
        window_len = int(max(0, end - start))

        curve = vehicle_curves_by_name.get(session.vehicle_name)
        if curve is None:
            raise ValueError(f"Keine Ladekurve für vehicle_name='{session.vehicle_name}' gefunden.")

        battery_capacity_kwh = max(float(curve.battery_capacity_kwh), 1e-12)

        soc0 = float(np.clip(session.state_of_charge_at_arrival, 0.0, 1.0))
        needed_battery_kwh = max(0.0, (target_soc - soc0) * battery_capacity_kwh)
        needed_site_kwh = needed_battery_kwh / max(charger_efficiency, 1e-12)

        remaining_site_kwh = float(max(needed_site_kwh, 0.0))
        soc = float(soc0)

        plan_site_kwh_per_step = np.zeros(window_len, dtype=float)
        plan_pv_site_kwh_per_step = np.zeros(window_len, dtype=float)
        plan_market_site_kwh_per_step = np.zeros(window_len, dtype=float)
        plan_immediate_site_kwh_per_step = np.zeros(window_len, dtype=float)

        charged_site_kwh = 0.0
        charged_pv_site_kwh = 0.0
        charged_market_site_kwh = 0.0

        if window_len <= 0 or remaining_site_kwh <= 1e-12:
            results.append(
                {
                    "session_id": session.session_id,
                    "vehicle_name": session.vehicle_name,
                    "vehicle_class": session.vehicle_class,
                    "arrival_step": int(session.arrival_step),
                    "departure_step": int(session.departure_step),
                    "required_site_kwh": float(needed_site_kwh),
                    "required_battery_kwh": float(needed_battery_kwh),
                    "charged_site_kwh": 0.0,
                    "charged_pv_site_kwh": 0.0,
                    "charged_market_site_kwh": 0.0,
                    "charged_immediate_site_kwh": 0.0,
                    "remaining_site_kwh": float(remaining_site_kwh),
                    "final_soc": float(soc),
                    "plan_site_kwh_per_step": plan_site_kwh_per_step,
                    "plan_pv_site_kwh_per_step": plan_pv_site_kwh_per_step,
                    "plan_market_site_kwh_per_step": plan_market_site_kwh_per_step,
                    "plan_immediate_site_kwh_per_step": plan_immediate_site_kwh_per_step,
                }
            )
            continue

        session_steps = np.arange(start, end, dtype=int)

        pv_after_base = np.maximum(
            pv_generation_kwh_per_step[session_steps].astype(float)
            - base_load_kwh_per_step[session_steps].astype(float),
            0.0,
        )
        pv_remaining = np.maximum(
            pv_after_base
            - reserved_pv_ev_energy_kwh_per_step[session_steps].astype(float),
            0.0,
        )
        pv_cap_est = np.minimum(pv_remaining, float(charger_cap_kwh_per_step))

        residual_base_on_grid = np.maximum(
            base_load_kwh_per_step[session_steps].astype(float)
            - pv_generation_kwh_per_step[session_steps].astype(float),
            0.0,
        )
        grid_ev_already = np.maximum(
            reserved_total_ev_energy_kwh_per_step[session_steps].astype(float)
            - reserved_pv_ev_energy_kwh_per_step[session_steps].astype(float),
            0.0,
        )
        nap_headroom = np.maximum(
            float(grid_limit_kwh_per_step) - residual_base_on_grid - grid_ev_already,
            0.0,
        )
        nap_cap_est = np.minimum(nap_headroom, float(charger_cap_kwh_per_step))

        total_pv_possible_est = float(np.sum(pv_cap_est))
        pv_only_plausible = total_pv_possible_est >= remaining_site_kwh - 1e-9

        nap_needed_est = max(0.0, remaining_site_kwh - total_pv_possible_est)

        preferred_nap = np.zeros(window_len, dtype=bool)
        if nap_needed_est > 1e-9:
            prices = market_price_eur_per_mwh[session_steps].astype(float)
            order = np.argsort(prices)
            cap_sorted = nap_cap_est[order]
            cumsum = np.cumsum(cap_sorted)
            k = int(np.searchsorted(cumsum, nap_needed_est, side="left")) + 1
            k = int(np.clip(k, 0, window_len))
            if k > 0:
                preferred_nap[order[:k]] = True

        if pv_only_plausible:
            preferred_nap[:] = False

        suffix_pv = np.zeros(window_len + 1, dtype=float)
        suffix_nap_pref = np.zeros(window_len + 1, dtype=float)
        for i in range(window_len - 1, -1, -1):
            suffix_pv[i] = suffix_pv[i + 1] + float(pv_cap_est[i])
            suffix_nap_pref[i] = suffix_nap_pref[i + 1] + (
                float(nap_cap_est[i]) if preferred_nap[i] else 0.0
            )

        for local_i, abs_step in enumerate(range(start, end)):
            if remaining_site_kwh <= 1e-12 or soc >= 1.0 - 1e-12:
                break

            pv_alloc_kwh, pv_share_kwh = max_available_power_for_session(
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
            pv_share_kwh = float(np.clip(float(pv_share_kwh), 0.0, pv_alloc_kwh))

            if pv_alloc_kwh > 1e-12:
                plan_site_kwh_per_step[local_i] += pv_alloc_kwh
                plan_pv_site_kwh_per_step[local_i] += pv_share_kwh

                reserved_total_ev_energy_kwh_per_step[int(abs_step)] += pv_alloc_kwh
                reserved_pv_ev_energy_kwh_per_step[int(abs_step)] += pv_share_kwh

                charged_site_kwh += pv_alloc_kwh
                charged_pv_site_kwh += pv_share_kwh
                remaining_site_kwh -= pv_alloc_kwh

                battery_energy_kwh = pv_alloc_kwh * charger_efficiency
                soc = float(min(1.0, soc + battery_energy_kwh / battery_capacity_kwh))

                if remaining_site_kwh <= 1e-12 or soc >= 1.0 - 1e-12:
                    break

            future_possible = float(suffix_pv[local_i + 1] + suffix_nap_pref[local_i + 1])
            must_charge = remaining_site_kwh > future_possible + 1e-9

            allow_nap_now = must_charge or bool(preferred_nap[local_i])
            if not allow_nap_now:
                continue

            nap_alloc_kwh, _nap_pv_share = max_available_power_for_session(
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

            nap_alloc_kwh = float(max(nap_alloc_kwh, 0.0))
            if nap_alloc_kwh <= 1e-12:
                continue

            plan_site_kwh_per_step[local_i] += nap_alloc_kwh
            plan_market_site_kwh_per_step[local_i] += nap_alloc_kwh

            reserved_total_ev_energy_kwh_per_step[int(abs_step)] += nap_alloc_kwh

            charged_site_kwh += nap_alloc_kwh
            charged_market_site_kwh += nap_alloc_kwh
            remaining_site_kwh -= nap_alloc_kwh

            battery_energy_kwh = nap_alloc_kwh * charger_efficiency
            soc = float(min(1.0, soc + battery_energy_kwh / battery_capacity_kwh))

        results.append(
            {
                "session_id": session.session_id,
                "vehicle_name": session.vehicle_name,
                "vehicle_class": session.vehicle_class,
                "arrival_step": int(session.arrival_step),
                "departure_step": int(session.departure_step),
                "required_site_kwh": float(needed_site_kwh),
                "required_battery_kwh": float(needed_battery_kwh),
                "charged_site_kwh": float(charged_site_kwh),
                "charged_pv_site_kwh": float(charged_pv_site_kwh),
                "charged_market_site_kwh": float(charged_market_site_kwh),
                "charged_immediate_site_kwh": 0.0,
                "remaining_site_kwh": float(max(remaining_site_kwh, 0.0)),
                "final_soc": float(soc),
                "plan_site_kwh_per_step": plan_site_kwh_per_step,
                "plan_pv_site_kwh_per_step": plan_pv_site_kwh_per_step,
                "plan_market_site_kwh_per_step": plan_market_site_kwh_per_step,
                "plan_immediate_site_kwh_per_step": plan_immediate_site_kwh_per_step,
            }
        )

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

    Ein Charger gilt als frei, wenn sein Eintrag in ``charger_occupied_until_step`` kleiner
    oder gleich ``arrival_step`` ist. In ``charger_occupied_until_step`` steht pro Charger
    der erste Simulationsschritt, ab dem der Charger wieder verfügbar ist.

    Parameters
    ----------
    charger_occupied_until_step:
        Liste, die pro Charger den ersten freien Simulationsschritt enthält.
    arrival_step:
        Absoluter Simulationsschritt der Ankunft.

    Returns
    -------
    int | None
        Charger-ID, falls frei, sonst None.
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
    market_price_eur_per_mwh: Optional[np.ndarray],
    record_debug: bool = False,
    record_charger_traces: bool = False,
    return_notebook_artifacts: bool = False,   
    zoom_days: int = 1,                        
) -> Tuple[np.ndarray, List[dict], List[dict], Optional[List[dict]]]:
    """
    Simuliert den Standort-Lastgang mit FCFS-Ladepunktvergabe (Admission) und konfigurierbarem
    Lademanagement (immediate/market/pv).

    Ablauf
    ------
    1) Zeitmapping: arrival_time / departure_time werden auf das Simulationsraster gemappt.
    2) FCFS-Admission: Charger-Vergabe erfolgt deterministisch nach First-Come-First-Served.
       Sessions ohne freien Charger werden als "drive_off" markiert.
    3) Lademanagement (tagweise, Admission bleibt FCFS):
       - "immediate": chronologisch pro Session innerhalb des Tages (FCFS-Reihenfolge)
       - "market": Tag-Planung (Standdauer-Priorität) via ``plan_charging_market_for_day``
       - "pv"/"generation": Tag-Planung (PV-first + Grid-Fallback) via ``plan_charging_pv_for_day``
    4) Outputs:
       - ev_load_kw: EV-Leistung am Standort (kW) pro Zeitschritt
       - sessions_out: Session-Details + vereinheitlichte Plan-/KPI-Struktur
       - debug_rows: optional Bilanzdaten pro Schritt (PV->Base, PV->EV, Grid->Base, Grid->EV)
       - charger_traces: optional pro Charger und Zeitschritt (inkl. 0 kW bei "plugged")

    Hinweise
    --------
    - Die Reservierungs-Arrays (reserved_*) werden IN-PLACE durch die Planer aktualisiert.
    - ``sessions_out`` enthält für "plugged" Sessions immer die vereinheitlichten Felder:
      required_*, charged_* und die vier Plan-Arrays (site/pv/market/immediate).
    - Charger-Traces enthalten bewusst auch Zeitschritte mit 0 kW, wenn eine Session angeschlossen ist.

    Returns
    -------
    Tuple[np.ndarray, List[dict], List[dict], Optional[List[dict]]]
        (ev_load_kw, sessions_out, debug_rows, charger_traces)
    """
    number_steps_total = int(len(timestamps))
    if len(pv_generation_kwh_per_step) != number_steps_total:
        raise ValueError(
            "pv_generation_kwh_per_step muss die gleiche Länge haben wie timestamps."
        )
    if len(base_load_kwh_per_step) != number_steps_total:
        raise ValueError("base_load_kwh_per_step muss die gleiche Länge haben wie timestamps.")
    if market_price_eur_per_mwh is not None and len(market_price_eur_per_mwh) != number_steps_total:
        raise ValueError(
            "market_price_eur_per_mwh muss die gleiche Länge haben wie timestamps (falls angegeben)."
        )

    site_configuration = scenario["site"]
    number_chargers = int(scenario["site"]["number_chargers"])
    time_resolution_min = int(scenario["time_resolution_min"])
    step_hours = float(time_resolution_min) / 60.0

    reserved_total_ev_energy_kwh_per_step = np.zeros(number_steps_total, dtype=float)
    reserved_pv_ev_energy_kwh_per_step = np.zeros(number_steps_total, dtype=float)

    charger_occupied_until_step: List[int] = [0 for _ in range(number_chargers)]
    charger_id_by_session_id: Dict[str, int] = {}

    sessions_out: List[dict] = []
    charger_traces: List[dict] = []
    debug_rows: List[dict] = []

    charging_strategy_raw = str(scenario.get("charging_strategy", "immediate")).strip().lower()

    if charging_strategy_raw in {"generation", "pv", "pv_generation", "pv_first", "pv-first"}:
        charging_strategy = "pv"
    elif charging_strategy_raw in {"market"}:
        charging_strategy = "market"
    else:
        charging_strategy = "immediate"

    # ------------------------------------------------------------------
    # Helper: Zeit -> Step Index
    # (nutzt deine normalize_and_floor_to_grid/normalize_to_timestamps_timezone)
    # ------------------------------------------------------------------
    step_delta = pd.Timedelta(minutes=time_resolution_min)
    horizon_start = pd.Timestamp(timestamps[0])
    horizon_end_exclusive = pd.Timestamp(timestamps[-1]) + step_delta  # [start, end)

    def map_datetime_to_step_index(datetime_value: datetime, allow_end: bool) -> int:
        """Mappt eine Zeitangabe auf einen gültigen Simulationsschritt-Index (DST-/TZ-robust)."""
        normalized_timestamp = normalize_and_floor_to_grid(
            datetime_value, timestamps, time_resolution_min
        )

        if normalized_timestamp <= horizon_start:
            return 0
        if normalized_timestamp >= horizon_end_exclusive:
            return number_steps_total if allow_end else (number_steps_total - 1)

        try:
            position = timestamps.get_loc(normalized_timestamp)
            if isinstance(position, slice):
                step_index = int(position.start)
            elif isinstance(position, (np.ndarray, list)):
                step_index = int(position[0])
            else:
                step_index = int(position)
        except KeyError:
            step_index = int(
                timestamps.get_indexer([normalized_timestamp], method="nearest")[0]
            )

        maximum_index = number_steps_total if allow_end else (number_steps_total - 1)
        return int(max(0, min(step_index, maximum_index)))

    # ------------------------------------------------------------------
    # Helper: Session-Output Basis (reduziert Dopplung)
    # ------------------------------------------------------------------
    def build_session_output_base(
        session: SampledSession,
        status: str,
        charger_id: Optional[int],
        arrival_step: Optional[int],
        departure_step: Optional[int],
    ) -> dict:
        """
        Baut das gemeinsame Grundgerüst für ``sessions_out``-Einträge.

        Die Keys sind so gewählt, dass sie zur vereinheitlichten Struktur der Planer
        (immediate/market/pv) passen. Für nicht-angeschlossene Sessions bleiben KPIs/Pläne
        auf Defaultwerten; für "plugged" werden sie später überschrieben.
        """
        soc0 = float(getattr(session, "state_of_charge_at_arrival", 0.0))
        soc0 = float(np.clip(soc0, 0.0, 1.0))

        return {
            "session_id": session.session_id,
            "vehicle_name": session.vehicle_name,
            "vehicle_class": session.vehicle_class,
            "arrival_time": getattr(session, "arrival_time", None),
            "departure_time": getattr(session, "departure_time", None),
            "arrival_step": int(arrival_step) if arrival_step is not None else None,
            "departure_step": int(departure_step) if departure_step is not None else None,
            "charger_id": int(charger_id) if charger_id is not None else None,
            "status": str(status),
            # Vereinheitlichte KPI-Felder (werden bei "plugged" überschrieben)
            "required_site_kwh": 0.0,
            "required_battery_kwh": 0.0,
            "charged_site_kwh": 0.0,
            "charged_pv_site_kwh": 0.0,
            "charged_market_site_kwh": 0.0,
            "charged_immediate_site_kwh": 0.0,
            "remaining_site_kwh": 0.0,
            "state_of_charge_at_arrival": soc0,
            "state_of_charge_end": soc0,
            # Vereinheitlichte Plan-Arrays (werden bei "plugged" überschrieben)
            "plan_site_kwh_per_step": np.array([], dtype=float),
            "plan_pv_site_kwh_per_step": np.array([], dtype=float),
            "plan_market_site_kwh_per_step": np.array([], dtype=float),
            "plan_immediate_site_kwh_per_step": np.array([], dtype=float),
        }

    # ------------------------------------------------------------------
    # Pass 1: Sessions sortieren, mappen, FCFS-Charger vergeben
    # ------------------------------------------------------------------
    def session_sort_key(session: SampledSession) -> pd.Timestamp:
        """Sort-Key für FCFS: Ankunftszeit in TZ des Rasters, robust gegen NaN."""
        try:
            return normalize_to_timestamps_timezone(session.arrival_time, timestamps)
        except Exception:
            return pd.Timestamp(timestamps[0])

    sessions_sorted = sorted(sessions, key=session_sort_key)

    plugged_sessions: List[SampledSession] = []

    for session in sessions_sorted:
        if (
            getattr(session, "arrival_time", None) is None
            or getattr(session, "departure_time", None) is None
        ):
            sessions_out.append(build_session_output_base(session, "invalid_time", None, None, None))
            continue

        mapped_arrival_step = map_datetime_to_step_index(session.arrival_time, allow_end=False)
        mapped_departure_step = map_datetime_to_step_index(session.departure_time, allow_end=True)

        if int(mapped_departure_step) <= int(mapped_arrival_step):
            mapped_departure_step = min(int(mapped_arrival_step) + 1, number_steps_total)

        session.arrival_step = int(mapped_arrival_step)
        session.departure_step = int(mapped_departure_step)

        chosen_charger_id = find_free_charger_fcfs(
            charger_occupied_until_step=charger_occupied_until_step,
            arrival_step=int(session.arrival_step),
        )

        if chosen_charger_id is None:
            sessions_out.append(
                build_session_output_base(
                    session=session,
                    status="drive_off",
                    charger_id=None,
                    arrival_step=int(session.arrival_step),
                    departure_step=int(session.departure_step),
                )
            )
            continue

        charger_occupied_until_step[int(chosen_charger_id)] = int(session.departure_step)
        charger_id_by_session_id[str(session.session_id)] = int(chosen_charger_id)
        plugged_sessions.append(session)

    # ------------------------------------------------------------------
    # Pass 2: Strategie anwenden (tagweise gruppiert, Admission bleibt FCFS)
    # ------------------------------------------------------------------
    sessions_out_by_id: Dict[str, dict] = {}

    def day_key_from_arrival_step(arrival_step: int) -> pd.Timestamp:
        """Kalendertag (TZ-sicher) aus einem absoluten arrival_step ableiten."""
        arrival_timestamp = pd.Timestamp(timestamps[int(arrival_step)])
        return arrival_timestamp.normalize()

    plugged_sessions_by_day: Dict[pd.Timestamp, List[SampledSession]] = {}
    for session in plugged_sessions:
        key = day_key_from_arrival_step(int(session.arrival_step))
        plugged_sessions_by_day.setdefault(key, []).append(session)

    for _, day_sessions in sorted(plugged_sessions_by_day.items(), key=lambda item: item[0]):
        if charging_strategy == "market":
            if market_price_eur_per_mwh is None:
                raise ValueError("charging_strategy='market' benötigt market_price_eur_per_mwh.")
            day_results = plan_charging_market_for_day(
                sessions=day_sessions,
                vehicle_curves_by_name=vehicle_curves_by_name,
                scenario=scenario,
                market_price_eur_per_mwh=np.asarray(market_price_eur_per_mwh, dtype=float),
                pv_generation_kwh_per_step=np.asarray(pv_generation_kwh_per_step, dtype=float),
                base_load_kwh_per_step=np.asarray(base_load_kwh_per_step, dtype=float),
                reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
                reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
            )
            for result in day_results:
                sessions_out_by_id[str(result["session_id"])] = result

        elif charging_strategy == "pv":
            if market_price_eur_per_mwh is None:
                raise ValueError(
                    "charging_strategy='pv' benötigt market_price_eur_per_mwh (Fallback-Ranking)."
                )
            day_results = plan_charging_pv_for_day(
                sessions=day_sessions,
                vehicle_curves_by_name=vehicle_curves_by_name,
                scenario=scenario,
                market_price_eur_per_mwh=np.asarray(market_price_eur_per_mwh, dtype=float),
                pv_generation_kwh_per_step=np.asarray(pv_generation_kwh_per_step, dtype=float),
                base_load_kwh_per_step=np.asarray(base_load_kwh_per_step, dtype=float),
                reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
                reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
            )
            for result in day_results:
                sessions_out_by_id[str(result["session_id"])] = result

        else:
            day_sessions_sorted = sorted(
                day_sessions, key=lambda s: (int(s.arrival_step), str(s.session_id))
            )
            for session in day_sessions_sorted:
                curve = vehicle_curves_by_name.get(session.vehicle_name)
                if curve is None:
                    raise ValueError(
                        f"Keine Ladekurve für vehicle_name='{session.vehicle_name}' gefunden."
                    )

                site_configuration = scenario["site"]
                vehicles_configuration = scenario["vehicles"]
                charger_efficiency = float(site_configuration.get("charger_efficiency", 1.0))
                target_soc = float(
                    np.clip(float(vehicles_configuration.get("soc_target", 1.0)), 0.0, 1.0)
                )

                battery_capacity_kwh = float(max(curve.battery_capacity_kwh, 1e-12))
                soc_start = float(np.clip(session.state_of_charge_at_arrival, 0.0, 1.0))

                required_battery_energy_kwh = float(
                    max(0.0, (target_soc - soc_start) * battery_capacity_kwh)
                )
                required_site_energy_kwh = float(
                    required_battery_energy_kwh / max(charger_efficiency, 1e-12)
                )

                result = plan_charging_immediate(
                    session_arrival_step=int(session.arrival_step),
                    session_departure_step=int(session.departure_step),
                    required_site_energy_kwh=float(required_site_energy_kwh),
                    pv_generation_kwh_per_step=np.asarray(pv_generation_kwh_per_step, dtype=float),
                    base_load_kwh_per_step=np.asarray(base_load_kwh_per_step, dtype=float),
                    reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
                    reserved_pv_ev_energy_kwh_per_step=reserved_pv_ev_energy_kwh_per_step,
                    curve=curve,
                    state_of_charge_at_arrival=float(soc_start),
                    scenario=scenario,
                )
                result["session_id"] = session.session_id
                result["vehicle_name"] = session.vehicle_name
                result["vehicle_class"] = session.vehicle_class
                result["arrival_step"] = int(session.arrival_step)
                result["departure_step"] = int(session.departure_step)
                sessions_out_by_id[str(session.session_id)] = result

    # ------------------------------------------------------------------
    # sessions_out finalisieren (plugged + nicht-plugged zusammenführen)
    # ------------------------------------------------------------------
    for session in plugged_sessions:
        charger_id = charger_id_by_session_id.get(str(session.session_id))
        result = sessions_out_by_id.get(str(session.session_id))
        if result is None:
            sessions_out.append(
                build_session_output_base(
                    session=session,
                    status="plugged",
                    charger_id=charger_id,
                    arrival_step=int(session.arrival_step),
                    departure_step=int(session.departure_step),
                )
            )
            continue

        window_start_step = int(max(0, session.arrival_step))
        window_end_step_exclusive = int(min(session.departure_step, number_steps_total))
        window_length = int(max(0, window_end_step_exclusive - window_start_step))

        def to_length(array_like: Any, length: int) -> np.ndarray:
            """Konvertiert array_like nach np.ndarray[float] und pad/clip auf Länge."""
            array = np.asarray(array_like, dtype=float).reshape(-1)
            if len(array) == length:
                return array
            if len(array) > length:
                return array[:length]
            padded = np.zeros(length, dtype=float)
            padded[: len(array)] = array
            return padded

        plan_site_kwh_per_step = to_length(result.get("plan_site_kwh_per_step", []), window_length)
        plan_pv_site_kwh_per_step = to_length(result.get("plan_pv_site_kwh_per_step", []), window_length)
        plan_market_site_kwh_per_step = to_length(result.get("plan_market_site_kwh_per_step", []), window_length)
        plan_immediate_site_kwh_per_step = to_length(result.get("plan_immediate_site_kwh_per_step", []), window_length)

        site_configuration = scenario["site"]
        charger_efficiency = float(site_configuration.get("charger_efficiency", 1.0))

        curve = vehicle_curves_by_name.get(session.vehicle_name)
        battery_capacity_kwh = (
            float(max(curve.battery_capacity_kwh, 1e-12)) if curve is not None else 1e-12
        )

        battery_added_cumulative_kwh = np.cumsum(plan_site_kwh_per_step) * charger_efficiency
        state_of_charge_trace = float(session.state_of_charge_at_arrival) + (
            battery_added_cumulative_kwh / battery_capacity_kwh
        )
        state_of_charge_trace = np.clip(state_of_charge_trace, 0.0, 1.0)

        default_final_soc = (
            float(state_of_charge_trace[-1])
            if window_length > 0
            else float(session.state_of_charge_at_arrival)
        )
        final_soc = float(np.clip(float(result.get("final_soc", default_final_soc)), 0.0, 1.0))

        session_row = build_session_output_base(
            session=session,
            status="plugged",
            charger_id=charger_id,
            arrival_step=int(session.arrival_step),
            departure_step=int(session.departure_step),
        )

        session_row.update(
            {
                "required_site_kwh": float(result.get("required_site_kwh", 0.0)),
                "required_battery_kwh": float(result.get("required_battery_kwh", 0.0)),
                "charged_site_kwh": float(result.get("charged_site_kwh", 0.0)),
                "charged_pv_site_kwh": float(result.get("charged_pv_site_kwh", 0.0)),
                "charged_market_site_kwh": float(result.get("charged_market_site_kwh", 0.0)),
                "charged_immediate_site_kwh": float(result.get("charged_immediate_site_kwh", 0.0)),
                "remaining_site_kwh": float(result.get("remaining_site_kwh", 0.0)),
                "state_of_charge_end": float(final_soc),
                "plan_site_kwh_per_step": plan_site_kwh_per_step,
                "plan_pv_site_kwh_per_step": plan_pv_site_kwh_per_step,
                "plan_market_site_kwh_per_step": plan_market_site_kwh_per_step,
                "plan_immediate_site_kwh_per_step": plan_immediate_site_kwh_per_step,
            }
        )

        sessions_out.append(session_row)

        # ------------------------------------------------------------------
        # Charger-Traces (inkl. 0 kW bei "plugged")
        # ------------------------------------------------------------------
        if record_charger_traces and charger_id is not None:
            for relative_step_index in range(window_length):
                absolute_step_index = window_start_step + relative_step_index
                if absolute_step_index < 0 or absolute_step_index >= number_steps_total:
                    continue

                timestamp_value = pd.Timestamp(timestamps[absolute_step_index]).to_pydatetime()
                site_kwh_this_step = float(plan_site_kwh_per_step[relative_step_index])
                pv_kwh_this_step = float(plan_pv_site_kwh_per_step[relative_step_index])

                power_kw_site = site_kwh_this_step / max(step_hours, 1e-12)
                pv_power_kw_site = pv_kwh_this_step / max(step_hours, 1e-12)

                charger_traces.append(
                    {
                        "timestamp": timestamp_value,
                        "charger_id": int(charger_id),
                        "session_id": session.session_id,
                        "vehicle_name": session.vehicle_name,
                        "power_kw": float(power_kw_site),
                        "pv_power_kw": float(pv_power_kw_site),
                        "mode": charging_strategy,
                        "state_of_charge": float(state_of_charge_trace[relative_step_index]),
                        "is_plugged": True,
                        "is_charging": bool(site_kwh_this_step > 1e-12),
                    }
                )

    # ------------------------------------------------------------------
    # Debug-Bilanz (vektorisiert)
    # ------------------------------------------------------------------
    if record_debug:
        grid_limit_kwh_per_step = float(scenario["site"]["grid_limit_p_avb_kw"]) * step_hours
        charger_rated_power_kw = float(scenario["site"]["rated_power_kw"])

        pv_generation = np.asarray(pv_generation_kwh_per_step, dtype=float)
        base_load = np.asarray(base_load_kwh_per_step, dtype=float)
        ev_load = np.asarray(reserved_total_ev_energy_kwh_per_step, dtype=float)
        pv_ev = np.asarray(reserved_pv_ev_energy_kwh_per_step, dtype=float)

        pv_to_base = np.minimum(pv_generation, base_load)
        base_remaining = base_load - pv_to_base

        pv_available_for_ev = np.maximum(pv_generation - pv_to_base, 0.0)
        pv_to_ev = np.minimum(pv_ev, pv_available_for_ev)
        ev_remaining = ev_load - pv_to_ev

        grid_to_base = np.minimum(base_remaining, grid_limit_kwh_per_step)
        grid_to_ev = np.minimum(
            ev_remaining, np.maximum(grid_limit_kwh_per_step - grid_to_base, 0.0)
        )

        for step_index in range(number_steps_total):
            debug_rows.append(
                {
                    "timestamp": pd.Timestamp(timestamps[step_index]),
                    "pv_generation_kwh_per_step": float(pv_generation[step_index]),
                    "base_load_kwh_per_step": float(base_load[step_index]),
                    "ev_load_kwh_per_step": float(ev_load[step_index]),
                    "pv_ev_kwh_per_step": float(pv_ev[step_index]),
                    "pv_to_base_kwh_per_step": float(pv_to_base[step_index]),
                    "pv_to_ev_kwh_per_step": float(pv_to_ev[step_index]),
                    "grid_to_base_kwh_per_step": float(grid_to_base[step_index]),
                    "grid_to_ev_kwh_per_step": float(grid_to_ev[step_index]),
                    "grid_limit_kwh_per_step": float(grid_limit_kwh_per_step),
                    "charger_rated_power_kw": float(charger_rated_power_kw),
                }
            )

    ev_load_kw = reserved_total_ev_energy_kwh_per_step / max(step_hours, 1e-12)
    charger_traces_out = (charger_traces if record_charger_traces else None)

    # Standard (wie früher): 4er-Tuple
    if not return_notebook_artifacts:
        return ev_load_kw, sessions_out, debug_rows, charger_traces_out

    # Notebook-Outputs direkt (ohne artifacts-Dict)
    charger_traces_dataframe = (
        pd.DataFrame(charger_traces_out) if charger_traces_out is not None else pd.DataFrame()
    )

    timeseries_dataframe = build_timeseries_dataframe(
        timestamps=timestamps,
        ev_load_kw=ev_load_kw,
        scenario=scenario,
        debug_rows=debug_rows if (record_debug and len(debug_rows) > 0) else None,
        generation_series=None,
        market_series=market_price_eur_per_mwh,
    )

    # Falls debug_rows nicht aktiv war: base/pv trotzdem befüllen
    if "base_load_kw" not in timeseries_dataframe.columns or timeseries_dataframe["base_load_kw"].isna().all():
        timeseries_dataframe["base_load_kw"] = np.asarray(base_load_kwh_per_step, dtype=float) / max(step_hours, 1e-12)
    if "pv_generation_kw" not in timeseries_dataframe.columns or timeseries_dataframe["pv_generation_kw"].isna().all():
        timeseries_dataframe["pv_generation_kw"] = np.asarray(pv_generation_kwh_per_step, dtype=float) / max(step_hours, 1e-12)

    if "market_price_eur_per_mwh" not in timeseries_dataframe.columns:
        timeseries_dataframe["market_price_eur_per_mwh"] = (
            np.asarray(market_price_eur_per_mwh, dtype=float)
            if market_price_eur_per_mwh is not None
            else np.nan
        )

    # Zoom-Fenster wie bisher
    _, window_start, window_end = initialize_time_window(
        timestamps=timestamps,
        scenario=scenario,
        days=int(max(1, zoom_days)),
    )

    return (
        ev_load_kw,
        sessions_out,
        debug_rows,
        charger_traces_dataframe,
        timeseries_dataframe,
        window_start,
        window_end,
    )




# =============================================================================
# 5) Analyse / Validierung / Notebook-Helper
# =============================================================================

def get_charging_strategy_from_scenario(scenario: dict) -> str:
    """
    Liefert charging_strategy direkt aus dem Scenario-Top-Level.
    """
    return str(scenario.get("charging_strategy", "immediate")).strip().lower()


def get_step_hours_from_scenario(scenario: dict) -> float:
    """
    Liefert die Schrittweite in Stunden.

    Unterstützt beide Varianten:
    - Legacy: scenario["time_resolution_min"]
    - Neu:    scenario["time_resolution"] + scenario["time_unit"] (z.B. hours)
    """
    if "time_resolution_min" in scenario:
        time_resolution_min = float(scenario["time_resolution_min"])
        return max(time_resolution_min / 60.0, 1e-12)

    time_resolution = float(scenario.get("time_resolution", 1.0))
    time_unit = str(scenario.get("time_unit", "minutes")).strip().lower()

    if time_unit in {"min", "mins", "minute", "minutes"}:
        return max(time_resolution / 60.0, 1e-12)
    if time_unit in {"h", "hr", "hour", "hours"}:
        return max(time_resolution, 1e-12)
    if time_unit in {"s", "sec", "second", "seconds"}:
        return max(time_resolution / 3600.0, 1e-12)

    raise ValueError(f"Unknown time_unit='{time_unit}'. Erwartet minutes/hours/seconds.")


def get_time_resolution_min_from_scenario(scenario: dict) -> int:
    """
    Liefert die Schrittweite in Minuten (int), z.B. für CSV-Reader, die Minuten erwarten.
    """
    if "time_resolution_min" in scenario:
        return int(scenario["time_resolution_min"])
    step_hours = get_step_hours_from_scenario(scenario)
    return int(round(step_hours * 60.0))


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

    Spalten:
    - EV-Ladeleistung (ev_load_kw) [kW]
    - Grundlast (base_load_kw) [kW]
    - PV-Erzeugung (pv_generation_kw) [kW]
    - optional: PV->EV und Grid->EV (pv_to_ev_kw / grid_to_ev_kw) [kW]
    - optional: Marktpreis (market_price_eur_per_mwh)

    Wenn `debug_rows` vorhanden ist, werden PV-/Netz-Aufteilungen direkt daraus berechnet (kWh/step -> kW).
    Wenn `debug_rows` fehlt, wird stattdessen eine konstante Grundlast aus dem Szenario verwendet und
    PV-Erzeugung optional aus `generation_series` übernommen (hier wird kWh/step angenommen). In diesem Fall
    bleiben pv_to_ev_kw/grid_to_ev_kw leer.
    """
    step_hours = get_step_hours_from_scenario(scenario)

    dataframe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "ev_load_kw": np.array(ev_load_kw, dtype=float),
        }
    )

    if debug_rows is not None and len(debug_rows) > 0:
        debug_dataframe = pd.DataFrame(debug_rows)
        debug_dataframe = debug_dataframe.sort_values("timestamp").reset_index(drop=True)

        dataframe["base_load_kw"] = (
            np.array(debug_dataframe["base_load_kwh_per_step"], dtype=float) / max(step_hours, 1e-12)
        )
        dataframe["pv_generation_kw"] = (
            np.array(debug_dataframe["pv_generation_kwh_per_step"], dtype=float) / max(step_hours, 1e-12)
        )

        dataframe["pv_to_ev_kw"] = (
            np.array(debug_dataframe["pv_to_ev_kwh_per_step"], dtype=float) / max(step_hours, 1e-12)
        )
        dataframe["grid_to_ev_kw"] = (
            np.array(debug_dataframe["grid_to_ev_kwh_per_step"], dtype=float) / max(step_hours, 1e-12)
        )

    else:
        base_load_kw = float((scenario.get("site") or {}).get("base_load_kw", 0.0))
        dataframe["base_load_kw"] = base_load_kw

        if generation_series is not None:
            # Annahme: generation_series ist kWh/step (intern). => kW = kWh/step / step_hours
            dataframe["pv_generation_kw"] = np.array(generation_series, dtype=float) / max(step_hours, 1e-12)
        else:
            dataframe["pv_generation_kw"] = 0.0

        dataframe["pv_to_ev_kw"] = np.nan
        dataframe["grid_to_ev_kw"] = np.nan

    if market_series is not None:
        dataframe["market_price_eur_per_mwh"] = np.array(market_series, dtype=float)
    else:
        dataframe["market_price_eur_per_mwh"] = np.nan

    if "timestamp" in dataframe.columns and "ts" not in dataframe.columns:
        dataframe["ts"] = pd.to_datetime(dataframe["timestamp"], errors="coerce")

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
        if remaining_site_kwh is None or float(remaining_site_kwh) <= 1e-6:
            continue

        arrival_time = session.get("arrival_time")
        departure_time = session.get("departure_time")

        parking_duration_min = np.nan
        if arrival_time is not None and departure_time is not None:
            try:
                parking_duration_min = (
                    pd.to_datetime(departure_time, errors="coerce")
                    - pd.to_datetime(arrival_time, errors="coerce")
                ).total_seconds() / 60.0
            except Exception:
                parking_duration_min = np.nan

        not_reached_rows.append(
            {
                "session_id": session.get("session_id"),
                "charger_id": session.get("charger_id"),
                "arrival_time": arrival_time,
                "parking_duration_min": float(parking_duration_min) if np.isfinite(parking_duration_min) else np.nan,
                "soc_arrival": float(session.get("state_of_charge_at_arrival", np.nan)),
                "soc_end": float(session.get("state_of_charge_end", np.nan)),
                "remaining_energy_kwh": float(remaining_site_kwh),
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
    normalize_to_internal: bool = True,
) -> Tuple[pd.Series, str, pd.Series, str]:
    """
    Liest Strategie-Signale strikt aus der Kombi-CSV (Grundlast/PV/Marktpreis).

    Erwartet:
    - scenario["localload_pv_market_csv"]
    - scenario["site"]["base_load_annual_kwh"]
    - scenario["site"]["pv_system_size_kwp"]

    Returns
    -------
    pv_generation_series, pv_ylabel, market_price_series, market_ylabel

    Hinweise
    --------
    - pv_generation_series ist intern kWh/step (wenn normalize_to_internal=True),
      sonst kW.
    - market_price_series ist €/MWh (immer).
    """
    if "localload_pv_market_csv" not in scenario:
        raise ValueError("Fehlt: scenario['localload_pv_market_csv']")

    site = scenario.get("site") or {}
    if "base_load_annual_kwh" not in site:
        raise ValueError("Fehlt: scenario['site']['base_load_annual_kwh']")
    if "pv_system_size_kwp" not in site:
        raise ValueError("Fehlt: scenario['site']['pv_system_size_kwp']")

    timezone = scenario.get("timezone", "Europe/Berlin")

    building_load_kwh_per_step, pv_generation_kwh_per_step, market_price_eur_per_mwh = (
        read_local_load_pv_market_from_csv(
            csv_path=str(scenario["localload_pv_market_csv"]),
            timestamps=timestamps,
            timezone=str(timezone) if timezone else None,
            base_load_annual_kwh=float(site["base_load_annual_kwh"]),
            pv_system_size_kwp=float(site["pv_system_size_kwp"]),
            profiles_are_normalized=bool(site.get("profiles_are_normalized", True)),
            datetime_format=str(site.get("localload_pv_market_datetime_format", "%d.%m.%Y %H:%M")),
            separator=str(site.get("localload_pv_market_separator", ";")),
            decimal=str(site.get("localload_pv_market_decimal", ",")),
        )
    )

    # PV: intern kWh/step oder extern kW
    if normalize_to_internal:
        pv_series = pv_generation_kwh_per_step
        pv_ylabel = "PV [kWh/step]"
    else:
        # Schrittweite aus timestamps ableiten (DST-safe, kein Fallback über scenario nötig)
        if len(timestamps) < 2:
            raise ValueError("timestamps muss mindestens 2 Einträge haben (für step_hours).")
        step_hours = float((timestamps[1] - timestamps[0]).total_seconds()) / 3600.0
        pv_series = pv_generation_kwh_per_step / max(step_hours, 1e-12)
        pv_ylabel = "PV [kW]"

    market_series = market_price_eur_per_mwh
    market_ylabel = "Preis [€/MWh]"

    return pv_series, pv_ylabel, market_series, market_ylabel


def build_strategy_inputs(
    scenario: dict,
    timestamps: pd.DatetimeIndex,
    normalize_to_internal: bool = True,
) -> Tuple[np.ndarray, Optional[Any], Optional[str], Optional[np.ndarray], Optional[Any], Optional[str]]:
    """
    Baut die optionalen Strategie-Zeitreihen (PV-Erzeugung, Marktpreis) in Notebook-freundlicher Form.

    Liest PV+Market strikt aus der Kombi-CSV über build_strategy_signal_series(...) nur EINMAL.
    """
    pv_generation_series, pv_generation_ylabel, market_price_series, market_price_ylabel = (
        build_strategy_signal_series(
            scenario=scenario,
            timestamps=timestamps,
            normalize_to_internal=bool(normalize_to_internal),
        )
    )

    pv_generation_kwh_per_step = np.array(pv_generation_series, dtype=float)
    market_price_eur_per_mwh = np.array(market_price_series, dtype=float)

    return (
        pv_generation_kwh_per_step,
        pv_generation_series,
        pv_generation_ylabel,
        market_price_eur_per_mwh,
        market_price_series,
        market_price_ylabel,
    )


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


POWER_COLUMN_CANDIDATES: List[str] = ["power_kw", "power_kw_site", "site_power_kw"]
SOC_COLUMN_CANDIDATES: List[str] = [
    "state_of_charge",
    "state_of_charge_fraction",
    "soc",
    "state_of_charge_trace",
]


def _get_column_name(dataframe: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def _require_non_empty_dataframe(dataframe: Optional[pd.DataFrame], dataframe_name: str) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        raise ValueError(f"{dataframe_name} ist leer.")
    return dataframe


def _filter_time_window(
    dataframe: pd.DataFrame,
    timestamp_column_name: str,
    start: Optional[datetime],
    end: Optional[datetime],
) -> pd.DataFrame:
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
    vehicle_curves_by_name: Optional[Dict[str, VehicleChargingCurve]] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prüft Trace-Leistungen gegen zwei obere Grenzen:

    1) Charger-Limit aus dem Scenario (rated_power_kw)          -> Site-/Ladepunktseite
    2) Fahrzeug-Limit aus der Master-Charging-Curve (SoC->kW)   -> Batterieseite

    Für einen sauberen Vergleich gilt:
        P_batt = P_site * charger_efficiency
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

    rated_power_kw_site = float(scenario["site"]["rated_power_kw"])

    charger_efficiency = float(scenario["site"].get("charger_efficiency", 1.0))
    if not (0.0 < charger_efficiency <= 1.0):
        raise ValueError("site.charger_efficiency muss im Bereich (0, 1] liegen.")

    if vehicle_curves_by_name is None:
        vehicle_curves_by_name = read_vehicle_load_profiles_from_csv(str(scenario["vehicles"]["vehicle_curve_csv"]))

    validation_rows: List[dict] = []

    for _, row in dataframe.iterrows():
        vehicle_name = str(row["vehicle_name"])
        curve = vehicle_curves_by_name.get(vehicle_name)

        power_site_kw = float(pd.to_numeric(row[power_column_name], errors="coerce"))
        soc_value = float(pd.to_numeric(row[soc_column_name], errors="coerce"))

        if not np.isfinite(power_site_kw) or not np.isfinite(soc_value):
            continue

        charger_limit_ok = power_site_kw <= rated_power_kw_site + 1e-9

        power_batt_kw = power_site_kw * charger_efficiency

        vehicle_limit_batt_kw = np.nan
        vehicle_limit_ok = True

        if curve is not None:
            soc_clipped = float(np.clip(soc_value, 0.0, 1.0))
            max_power_batt_kw_from_curve = float(
                np.interp(
                    soc_clipped,
                    curve.state_of_charge_fraction,
                    curve.power_kw,  # Batterieseite
                )
            )
            max_power_batt_kw_from_curve = max(max_power_batt_kw_from_curve, 0.0)
            vehicle_limit_batt_kw = max_power_batt_kw_from_curve
            vehicle_limit_ok = power_batt_kw <= vehicle_limit_batt_kw + 1e-9

        validation_rows.append(
            {
                "timestamp": pd.to_datetime(row["timestamp"]),
                "charger_id": int(row["charger_id"]) if "charger_id" in row else np.nan,
                "session_id": str(row["session_id"]) if "session_id" in row else "",
                "vehicle_name": vehicle_name,
                "state_of_charge": soc_value,
                # Backwards-compatible (alt)
                "power_kw": power_site_kw,
                "charger_limit_kw": rated_power_kw_site,
                "vehicle_limit_kw": float(vehicle_limit_batt_kw) if np.isfinite(vehicle_limit_batt_kw) else np.nan,
                # Explizit (neu)
                "power_site_kw": power_site_kw,
                "power_batt_kw": power_batt_kw,
                "charger_limit_kw_site": rated_power_kw_site,
                "vehicle_limit_kw_batt": float(vehicle_limit_batt_kw)
                if np.isfinite(vehicle_limit_batt_kw)
                else np.nan,
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

    market_price_eur_per_mwh: Optional[pd.Series] = None
    if "market_price_eur_per_mwh" in dataframe.columns:
        market_price_eur_per_mwh = pd.to_numeric(dataframe["market_price_eur_per_mwh"], errors="coerce").astype(float)
    elif "market_price" in dataframe.columns:
        market_price_eur_per_mwh = pd.to_numeric(dataframe["market_price"], errors="coerce").astype(float)

    return {
        "dataframe": dataframe,
        "base_load_kw": base_load_kw,
        "ev_load_kw": ev_load_kw,
        "total_load_kw": total_load_kw,
        "pv_generation_kw": pv_generation_kw,
        "market_price_eur_per_mwh": market_price_eur_per_mwh,
        "grid_limit_kw": grid_limit_kw,
        "charger_limit_kw_total": charger_limit_kw_total,
    }


def build_ev_power_by_mode_timeseries_dataframe(
    timeseries_dataframe: pd.DataFrame,
    sessions_out: Optional[List[dict]] = None,
    scenario: Optional[dict] = None,
) -> pd.DataFrame:
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
                "ev_generation_kw": pd.to_numeric(dataframe["ev_generation_kw"], errors="coerce")
                .astype(float)
                .fillna(0.0),
                "ev_market_kw": pd.to_numeric(dataframe["ev_market_kw"], errors="coerce").astype(float).fillna(0.0),
                "ev_immediate_kw": pd.to_numeric(dataframe["ev_immediate_kw"], errors="coerce")
                .astype(float)
                .fillna(0.0),
            }
        )

    if sessions_out is None or scenario is None:
        raise ValueError(
            "Mode-Daten fehlen: Weder Mode-Spalten im timeseries_dataframe "
            "noch sessions_out+scenario für Aggregation verfügbar."
        )

    step_hours = get_step_hours_from_scenario(scenario)

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
    vehicle_curves_by_name: Optional[Dict[str, VehicleChargingCurve]] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """
    Masterkurve (Batterieseite) vs. Ist-Punkte (Batterieseite).

    Trace-Leistung ist typischerweise Site-Seite -> Umrechnung:
        P_batt = P_site * charger_efficiency
    """
    charger_traces_dataframe = _require_non_empty_dataframe(charger_traces_dataframe, "charger_traces_dataframe")
    dataframe = charger_traces_dataframe.copy()

    for column_name in ["timestamp", "vehicle_name"]:
        if column_name not in dataframe.columns:
            raise ValueError(f"charger_traces_dataframe: Spalte '{column_name}' fehlt.")

    power_column_name = _get_column_name(dataframe, POWER_COLUMN_CANDIDATES)
    if power_column_name is None:
        raise ValueError(
            "charger_traces_dataframe: keine Leistungsspalte gefunden (power_kw/power_kw_site/site_power_kw)."
        )

    soc_column_name = _get_column_name(dataframe, SOC_COLUMN_CANDIDATES)
    if soc_column_name is None:
        raise ValueError("charger_traces_dataframe: keine SoC-Spalte gefunden (state_of_charge/.../soc).")

    charger_efficiency = float(scenario["site"].get("charger_efficiency", 1.0))
    if not (0.0 < charger_efficiency <= 1.0):
        raise ValueError("site.charger_efficiency muss im Bereich (0, 1] liegen.")

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
    actual_power_site_kw = pd.to_numeric(dataframe[power_column_name], errors="coerce").to_numpy(dtype=float)

    valid_mask = np.isfinite(actual_soc) & np.isfinite(actual_power_site_kw)
    actual_soc = actual_soc[valid_mask]
    actual_power_site_kw = actual_power_site_kw[valid_mask]

    if len(actual_soc) == 0:
        raise ValueError("Alle realen Punkte sind nach der Umwandlung ungültig (SoC/Power).")

    actual_power_batt_kw = actual_power_site_kw * charger_efficiency

    if vehicle_curves_by_name is None:
        vehicle_curves_by_name = read_vehicle_load_profiles_from_csv(str(scenario["vehicles"]["vehicle_curve_csv"]))

    if str(vehicle_name) not in vehicle_curves_by_name:
        example_names = list(vehicle_curves_by_name.keys())[:10]
        raise ValueError(f"Fahrzeug '{vehicle_name}' nicht in den Masterkurven gefunden. Beispiele: {example_names}")

    curve = vehicle_curves_by_name[str(vehicle_name)]

    master_soc = np.array(curve.state_of_charge_fraction, dtype=float)
    master_power_battery_kw = np.array(curve.power_kw, dtype=float)

    allowed_power_kw_at_actual = np.interp(actual_soc, master_soc, master_power_battery_kw)
    violation_mask = actual_power_batt_kw > (allowed_power_kw_at_actual + 1e-9)

    return {
        "vehicle_name": str(vehicle_name),
        "master_soc": master_soc,
        "master_power_battery_kw": master_power_battery_kw,
        "actual_soc": actual_soc,
        # Backwards-compatible (alt)
        "actual_power_kw": actual_power_site_kw,
        # Explizit (neu)
        "actual_power_site_kw": actual_power_site_kw,
        "actual_power_batt_kw": actual_power_batt_kw,
        "allowed_power_kw_at_actual": allowed_power_kw_at_actual,
        "violation_mask": violation_mask,
        "number_violations": int(np.sum(violation_mask)),
        "number_points": int(len(actual_soc)),
    }


def choose_vehicle_for_master_curve_plot(
    sessions_out: List[dict],
    charger_traces_dataframe: pd.DataFrame,
    scenario: dict,
    vehicle_curves_by_name: Optional[Dict[str, VehicleChargingCurve]] = None,
) -> Dict[str, Any]:
    vehicle_name = get_most_used_vehicle_name(
        sessions_out=sessions_out,
        charger_traces_dataframe=charger_traces_dataframe,
        only_plugged_sessions=True,
    )

    plot_data = build_master_curve_and_actual_points_for_vehicle(
        charger_traces_dataframe=charger_traces_dataframe,
        scenario=scenario,
        vehicle_name=vehicle_name,
        vehicle_curves_by_name=vehicle_curves_by_name,
        start=None,
        end=None,
    )

    return {"vehicle_name": vehicle_name, "plot_data": plot_data}


# -----------------------------------------------------------------------------
# Notebook Helpers (kleine Utilities)
# -----------------------------------------------------------------------------


def show_strategy_status(charging_strategy: str, strategy_status: str) -> None:
    strategy_name = (charging_strategy or "immediate").capitalize()
    status_name = (strategy_status or "immediate").capitalize()
    print(f"Charging Strategy: {strategy_name}")
    print(f"Strategy Status: {status_name}")


def decorate_title_with_status(base_title: str, charging_strategy: str, strategy_status: str) -> str:
    strategy_name = (charging_strategy or "immediate").capitalize()
    status_name = (strategy_status or "immediate").capitalize()
    return f"{base_title} ({strategy_name}, {status_name})"


def initialize_time_window(
    timestamps: pd.DatetimeIndex,
    scenario: dict,
    days: int = 1,
) -> Tuple[Optional[int], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if timestamps is None or len(timestamps) == 0:
        return None, None, None

    time_resolution_min = get_time_resolution_min_from_scenario(scenario)
    days = int(max(1, days))

    window_start = pd.to_datetime(timestamps[0])
    window_end_target = window_start + pd.Timedelta(days=days) - pd.Timedelta(minutes=time_resolution_min)

    idx = int(timestamps.get_indexer([window_end_target], method="nearest")[0])
    idx = max(0, min(idx, len(timestamps) - 1))

    window_end = pd.to_datetime(timestamps[idx])

    return None, window_start, window_end


def get_holiday_dates_from_scenario(
    scenario: dict,
    timestamps: pd.DatetimeIndex,
) -> Set[date]:
    holidays_configuration = scenario.get("holidays") or {}
    manual_dates = holidays_configuration.get("dates") or []

    holiday_dates: Set[date] = set()

    for date_text in manual_dates:
        parsed = pd.to_datetime(date_text, errors="coerce")
        if pd.notna(parsed):
            holiday_dates.add(parsed.date())

    try:
        import holidays as python_holidays

        country_code = str(holidays_configuration.get("country", "DE")).strip()
        subdivision_code = holidays_configuration.get("subdivision", None)

        years = sorted({pd.to_datetime(ts).year for ts in timestamps})
        holiday_calendar = python_holidays.country_holidays(
            country_code,
            subdiv=subdivision_code,
            years=years,
        )

        for day in holiday_calendar.keys():
            holiday_dates.add(day)

    except Exception:
        pass

    return holiday_dates


def get_daytype_calendar(
    start_datetime: datetime,
    horizon_days: int,
    holiday_dates: Set[date],
) -> Dict[str, List[date]]:
    out: Dict[str, List[date]] = {
        "working_day": [],
        "saturday": [],
        "sunday_holiday": [],
    }

    for day_index in range(int(horizon_days)):
        day_start = start_datetime + timedelta(days=day_index)
        day_type = _get_day_type(day_start, holiday_dates)
        out[day_type].append(day_start.date())

    return out


def group_sessions_by_day(sessions_out: List[dict], only_plugged: bool = False) -> Dict[Any, List[dict]]:
    grouped: Dict[Any, List[dict]] = {}
    for session in sessions_out:
        if only_plugged and session.get("status") != "plugged":
            continue

        arrival_time = session.get("arrival_time")
        if arrival_time is None:
            continue

        day = pd.to_datetime(arrival_time, errors="coerce")
        if pd.isna(day):
            continue

        grouped.setdefault(day.date(), []).append(session)

    return grouped


def build_base_load_kwh_per_step(scenario: dict, timestamps: pd.DatetimeIndex) -> np.ndarray:
    if "localload_pv_market_csv" not in scenario:
        raise ValueError("Fehlt: scenario['localload_pv_market_csv']")

    site = scenario.get("site") or {}
    if "base_load_annual_kwh" not in site:
        raise ValueError("Fehlt: scenario['site']['base_load_annual_kwh']")
    if "pv_system_size_kwp" not in site:
        raise ValueError("Fehlt: scenario['site']['pv_system_size_kwp']")

    timezone = scenario.get("timezone", "Europe/Berlin")

    base_load_kwh_per_step, _, _ = read_local_load_pv_market_from_csv(
        csv_path=str(scenario["localload_pv_market_csv"]),
        timestamps=timestamps,
        timezone=str(timezone) if timezone else None,
        base_load_annual_kwh=float(site["base_load_annual_kwh"]),
        pv_system_size_kwp=float(site["pv_system_size_kwp"]),
        profiles_are_normalized=bool(site.get("profiles_are_normalized", True)),
        datetime_format=str(site.get("localload_pv_market_datetime_format", "%d.%m.%Y %H:%M")),
        separator=str(site.get("localload_pv_market_separator", ";")),
        decimal=str(site.get("localload_pv_market_decimal", ",")),
    )

    return base_load_kwh_per_step.to_numpy(dtype=float)