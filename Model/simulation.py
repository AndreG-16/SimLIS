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


def _step_hours(time_resolution_min: int) -> float:
    """
    Rechnet eine Zeitauflösung in Minuten in Stunden pro Zeitschritt um.
    """
    return float(time_resolution_min) / 60.0


def _ensure_datetime_index(dataframe: pd.DataFrame, datetime_column_name: str) -> pd.DataFrame:
    """
    Ensure a DataFrame has a clean, sorted DatetimeIndex created from a given datetime column.

    Supported datetime formats (explicit, no guessing):
      1) ISO with seconds:        "2018-01-01 02:30:00"
      2) ISO without seconds:     "2025-06-04 00:00"
      3) ISO with 'T':            "2025-06-04T00:00:00"
      4) German long format:      "14.06.2025 14:00"
      5) German short year:       "01.01.25 00:00"
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
    Reindiziert eine Zeitreihe auf die Simulations-Zeitstempel.
    Robust gegen doppelte Zeitstempel (z.B. SMARD CSV): wir aggregieren Duplikate.
    """
    series = series.sort_index()

    # 1) Duplikate im Index fixen (sonst knallt reindex)
    if not series.index.is_unique:
        # Option A: Mittelwert (bei PV/Preisen ok, wenn Duplikate identisch oder sehr nah)
        series = series.groupby(level=0).mean()

        # Alternativ Option B: letztes gewinnt
        # series = series[~series.index.duplicated(keep="last")]

    # 2) Reindex
    return series.reindex(timestamps, method=method)


# =============================================================================
# 1) Daten-Reader: Gebäudeprofil / Marktpreise / Ladekurven / PV-Generation
# =============================================================================

@dataclass
class VehicleChargingCurve:
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
    Liest ein Gebäude-/Grundlastprofil aus CSV ein, skaliert es auf einen Jahreswert
    und liefert kWh/step zurück.
    value_column_one_based bezieht sich auf die Original-CSV-Spalten (inkl. Zeitspalte).
    """
    dataframe = _read_table_flex(csv_path, prefer_decimal_comma=True)

    datetime_column_name = "DateTime" if "DateTime" in dataframe.columns else str(dataframe.columns[0])

    # 1) Wertspalte anhand ORIGINAL-CSV bestimmen (vor set_index!)
    original_columns = list(dataframe.columns)
    value_column_zero_based = int(value_column_one_based) - 1
    if value_column_zero_based < 0 or value_column_zero_based >= len(original_columns):
        raise ValueError(
            f"base_load_value_col={value_column_one_based} ist außerhalb der CSV-Spaltenanzahl "
            f"({len(original_columns)})."
        )
    value_column_name = original_columns[value_column_zero_based]

    # 2) Jetzt DatetimeIndex setzen
    dataframe = _ensure_datetime_index(dataframe, datetime_column_name)

    # 3) Werte ziehen
    raw_values = pd.to_numeric(dataframe[value_column_name], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    cleaned_unit = (value_unit or "").strip()

    if cleaned_unit == "kWh":
        shape_kwh_per_step = np.maximum(raw_values, 0.0)
        shape_sum = float(np.sum(shape_kwh_per_step))
        if shape_sum <= 0.0:
            raise ValueError("Gebäudeprofil: Summe der Shape-Werte ist 0 (kWh).")
        scaling_factor = float(annual_scaling_value) / shape_sum
        scaled_kwh_per_step = shape_kwh_per_step * scaling_factor

    elif cleaned_unit == "kW":
        shape_kw = np.maximum(raw_values, 0.0)
        shape_mean_kw = float(np.mean(shape_kw))
        if shape_mean_kw <= 0.0:
            raise ValueError("Gebäudeprofil: Mittelwert der Shape-Werte ist 0 (kW).")
        scaling_factor = float(annual_scaling_value) / shape_mean_kw
        scaled_kw = shape_kw * scaling_factor
        scaled_kwh_per_step = scaled_kw * _step_hours(time_resolution_min)

    else:
        raise ValueError("value_unit muss 'kWh' oder 'kW' sein.")

    series = pd.Series(scaled_kwh_per_step, index=dataframe.index, name="base_load_kwh_per_step")
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

    # 1) Spaltenname anhand ORIGINALER CSV-Spalten bestimmen (vor set_index!)
    original_columns = list(dataframe.columns)
    value_column_zero_based = int(value_column_one_based) - 1
    if value_column_zero_based < 0 or value_column_zero_based >= len(original_columns):
        raise ValueError(
            f"market_strategy_value_col={value_column_one_based} ist außerhalb der CSV-Spaltenanzahl "
            f"({len(original_columns)})."
        )
    value_column_name = original_columns[value_column_zero_based]

    # 2) Jetzt DatetimeIndex setzen
    dataframe = _ensure_datetime_index(dataframe, datetime_column_name)

    # 3) Werte lesen + Einheit konvertieren
    raw_values = pd.to_numeric(dataframe[value_column_name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    prices_eur_per_kwh = _convert_price_series_to_eur_per_kwh(raw_values, value_unit)

    series = pd.Series(prices_eur_per_kwh, index=dataframe.index, name="market_price_eur_per_kwh")
    series = _reindex_to_simulation_timestamps(series, timestamps, method="nearest")
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
    """
    time_resolution_min = int(scenario["time_resolution_min"])
    simulation_horizon_days = int(scenario["simulation_horizon_days"])
    start_datetime = datetime.fromisoformat(str(scenario["start_datetime"]))

    number_steps_total = int(simulation_horizon_days) * int(24 * 60 / time_resolution_min)
    timestamps = pd.date_range(start=start_datetime, periods=number_steps_total, freq=f"{time_resolution_min}min")
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

    # 1) Spaltenname anhand ORIGINALER CSV-Spalten bestimmen (vor set_index!)
    original_columns = list(dataframe.columns)
    value_column_zero_based = int(value_column_one_based) - 1

    if value_column_zero_based < 0 or value_column_zero_based >= len(original_columns):
        raise ValueError(
            f"generation_strategy_value_col={value_column_one_based} ist außerhalb der CSV-Spaltenanzahl "
            f"({len(original_columns)})."
        )

    value_column_name = original_columns[value_column_zero_based]

    # 2) Jetzt DatetimeIndex setzen
    dataframe = _ensure_datetime_index(dataframe, datetime_column_name)

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
        # numpy: lognormal(mean, sigma) nutzt mean/sigma der zugrunde liegenden Normalverteilung
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
    Erzeugt (sampelt) Sessions für genau einen Tag.
    """
    time_resolution_min = int(scenario["time_resolution_min"])
    steps_per_day = int(24 * 60 / time_resolution_min)

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
        arrival_time = simulation_day_start + timedelta(minutes=arrival_step * time_resolution_min)

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

        departure_time = simulation_day_start + timedelta(minutes=departure_step * time_resolution_min)

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
    Hinweis: In deinem Modell wird die Charger-Grenze pro Session/Ladepunkt angewendet.
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
    return np.arange(int(session_arrival_step), int(session_departure_step), dtype=int)


def _get_charging_step_order_market(
    session_arrival_step: int,
    session_departure_step: int,
    market_price_eur_per_kwh: Optional[np.ndarray],
) -> np.ndarray:
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
    SoC zum Zeitpunkt dieses Slots aus *allen* bereits geplanten Energiemengen in Slots < absolute_step_index.

    Damit ist die SoC-Kopplung korrekt:
    - bei market (Slots werden sortiert geplant) trotzdem physikalisch richtig
    - beim generation-Fallback: Grid-Plan berücksichtigt bereits PV-Plan
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
    Market:
    - wählt die günstigsten Slots in der Standzeit zuerst
    - nutzt insgesamt verfügbare Energie (PV + Netz)
    - PV wird in jedem Slot physikalisch zuerst genutzt (unabhängig von der Strategie)
    - SoC-Kopplung ist korrekt trotz Slot-Sortierung (siehe generischer Planner)
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
    Generation (PV-first):
    - nutzt gezielt PV (nach Grundlast, minus bereits reserviertes PV->EV)
    - verteilt fair über alle PV-Slots der Standzeit
    - lädt danach optional zusätzlich in PV-starken Slots
    - nutzt in dieser Teilplanung kein Netz (Netz kommt erst im grid-only fallback)

    Rückgabe:
      plan_site_kwh_per_step, remaining_site_energy_kwh, allocations_site_kwh_by_absolute_step
    """
    number_steps = int(session_departure_step) - int(session_arrival_step)
    plan_site_kwh_per_step = np.zeros(number_steps, dtype=float)

    if remaining_site_energy_kwh <= 0.0 or number_steps <= 0:
        return plan_site_kwh_per_step, float(remaining_site_energy_kwh), {}

    charger_limit_kwh_per_step = _charger_limit_site_kwh_per_step(scenario)
    allocations: Dict[int, float] = {}

    step_indices = np.arange(int(session_arrival_step), int(session_departure_step), dtype=int)

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

    pv_slot_mask = pv_available_each_step > 1e-12
    pv_slot_indices = step_indices[pv_slot_mask]
    if len(pv_slot_indices) == 0:
        return plan_site_kwh_per_step, float(remaining_site_energy_kwh), allocations

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
        vehicle_limit_kwh_per_step = _vehicle_site_limit_kwh_per_step_from_curve(curve, current_state_of_charge, scenario)

        # Site-headroom schützt vor negativen Headrooms (durch bereits reservierte Gesamtlast)
        site_headroom_kwh_per_step = _available_site_energy_kwh_for_new_reservation(
            pv_generation_kwh_per_step=pv_generation_kwh_per_step,
            base_load_kwh_per_step=base_load_kwh_per_step,
            reserved_total_ev_energy_kwh_per_step=reserved_total_ev_energy_kwh_per_step,
            scenario=scenario,
            step_index=int(absolute_step_index),
        )

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
        reserved_pv_ev_energy_kwh_per_step[int(absolute_step_index)] += allocated_site_kwh  # PV-only Reservierung

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
        vehicle_limit_kwh_per_step = _vehicle_site_limit_kwh_per_step_from_curve(curve, current_state_of_charge, scenario)

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


def normalize_datetime_to_simulation_grid(
    datetime_value,
    timestamps: pd.DatetimeIndex,
    time_resolution_min: int,
) -> pd.Timestamp:
    """
    Normalisiert einen Zeitstempel auf das Simulations-Zeitraster.

    - passt Zeitzone an die Simulations-Timestamps an
    - entfernt Sekunden/Mikrosekunden
    - rundet nach unten auf das Zeitraster (floor auf time_resolution_min)
    """
    timestamps_timezone = getattr(timestamps, "tz", None)
    timestamp_value = pd.Timestamp(datetime_value)

    # Zeitzonen-Handling: an Simulations-Timestamps anpassen
    if timestamps_timezone is not None:
        if timestamp_value.tzinfo is None:
            timestamp_value = timestamp_value.tz_localize(timestamps_timezone)
        else:
            timestamp_value = timestamp_value.tz_convert(timestamps_timezone)
    else:
        # Simulations-Timestamps sind tz-naiv → Session-Zeiten ebenfalls tz-naiv halten
        if timestamp_value.tzinfo is not None:
            timestamp_value = timestamp_value.tz_convert(None).tz_localize(None)

    # Sekunden und Mikrosekunden entfernen und auf Raster runden
    timestamp_value = timestamp_value.replace(second=0, microsecond=0)
    timestamp_value = timestamp_value.flo


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
    - Daraus werden arrival_step/departure_step berechnet und auf der Session gesetzt (wichtig für alle Planner).
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

    timestamps_timezone = getattr(timestamps, "tz", None)

    def _normalize_datetime_to_timestamps_timezone(datetime_value) -> pd.Timestamp:
        """
        Normalisiert eine Zeitangabe auf:
        - gleiche Timezone wie timestamps (falls vorhanden)
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

        # 🔑 Wichtig: arrival_step / departure_step auf der Session setzen,
        # damit plan_ev_charging_session und plan_charging_* konsistent arbeiten.
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
    Baut die zentrale timeseries_dataframe für Notebook-Plots (inkl. pv/grid Aufteilung aus debug_rows, falls vorhanden).
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


# =============================================================================
# Notebook-kompatible Builder
# =============================================================================


def _get_column_name(dataframe: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Gibt den ersten Spaltennamen aus `candidates` zurück, der im DataFrame existiert.
    """
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


def _require_non_empty_dataframe(dataframe: Optional[pd.DataFrame], dataframe_name: str) -> pd.DataFrame:
    """
    Stellt sicher, dass ein DataFrame existiert und nicht leer ist.
    """
    if dataframe is None or len(dataframe) == 0:
        raise ValueError(f"{dataframe_name} ist leer.")
    return dataframe


def _filter_time_window(
    dataframe: pd.DataFrame,
    timestamp_column_name: str,
    start: Optional[datetime],
    end: Optional[datetime],
) -> pd.DataFrame:
    """
    Filtert ein DataFrame auf ein Zeitfenster [start, end] basierend auf einer Timestamp-Spalte.
    """
    dataframe = dataframe.copy()
    dataframe[timestamp_column_name] = pd.to_datetime(dataframe[timestamp_column_name])

    if start is not None:
        dataframe = dataframe[dataframe[timestamp_column_name] >= pd.to_datetime(start)]
    if end is not None:
        dataframe = dataframe[dataframe[timestamp_column_name] <= pd.to_datetime(end)]

    if len(dataframe) == 0:
        raise ValueError("Keine Daten im gewählten Zeitfenster.")
    return dataframe


def build_power_per_charger_timeseries(
    charger_traces_dataframe: pd.DataFrame,
    charger_id: int,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Liefert Zeitreihe der Ladeleistung eines einzelnen Ladepunkts.

    Rückgabe-Spalten:
      - timestamp
      - power_kw
    """
    charger_traces_dataframe = _require_non_empty_dataframe(charger_traces_dataframe, "charger_traces_dataframe")

    dataframe = charger_traces_dataframe.copy()
    dataframe = _filter_time_window(dataframe, "timestamp", start=start, end=end)

    dataframe = dataframe[dataframe["charger_id"] == int(charger_id)]
    if len(dataframe) == 0:
        raise ValueError(f"Keine Daten für charger_id={int(charger_id)} im gewählten Zeitfenster.")

    power_column_name = _get_column_name(dataframe, ["power_kw_site", "site_power_kw", "power_kw"])
    if power_column_name is None:
        raise ValueError("Keine Leistungsspalte gefunden (power_kw / power_kw_site / site_power_kw).")

    out = dataframe[["timestamp", power_column_name]].copy()
    out = out.rename(columns={power_column_name: "power_kw"})
    out["power_kw"] = out["power_kw"].astype(float).fillna(0.0)

    return out


def build_ev_power_by_source_timeseries(timeseries_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Liefert EV-Leistung nach Quelle (PV vs Grid).

    Erwartete Spalten im timeseries_dataframe:
      - pv_to_ev_kw
      - grid_to_ev_kw

    Rückgabe:
      - timestamp
      - ev_from_pv_kw
      - ev_from_grid_kw
    """
    timeseries_dataframe = _require_non_empty_dataframe(timeseries_dataframe, "timeseries_dataframe")

    if "pv_to_ev_kw" not in timeseries_dataframe.columns:
        raise ValueError("timeseries_dataframe: Spalte 'pv_to_ev_kw' fehlt (Debug nicht aktiv?).")
    if "grid_to_ev_kw" not in timeseries_dataframe.columns:
        raise ValueError("timeseries_dataframe: Spalte 'grid_to_ev_kw' fehlt (Debug nicht aktiv?).")

    dataframe = timeseries_dataframe.copy()
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])

    out = pd.DataFrame(
        {
            "timestamp": dataframe["timestamp"],
            "ev_from_pv_kw": dataframe["pv_to_ev_kw"].astype(float).fillna(0.0),
            "ev_from_grid_kw": dataframe["grid_to_ev_kw"].astype(float).fillna(0.0),
        }
    )
    return out


def build_soc_timeseries_by_charger(
    charger_traces_dataframe: pd.DataFrame,
    charger_ids: List[int],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Dict[int, pd.DataFrame]:
    """
    Build per-charger state-of-charge (SoC) time series for notebook plotting.

    The returned DataFrames include the following columns:
      - timestamp: datetime-like
      - soc: state of charge as float in [0, 1] (or whatever your trace uses)
      - session_id: identifier of the charging session (if available in traces)
      - vehicle_name: vehicle name (if available in traces)

    Including session_id enables notebook plots to break the SoC line whenever a
    charging session ends (unplug) and a new one begins.

    Parameters
    ----------
    charger_traces_dataframe:
        DataFrame containing charger trace points. Expected columns include:
        'timestamp', 'charger_id', and one SoC column such as
        'state_of_charge' / 'state_of_charge_fraction' / 'soc' / 'state_of_charge_trace'.
        If present, 'session_id' and 'vehicle_name' are propagated to the output.
    charger_ids:
        List of charger IDs to include.
    start, end:
        Optional inclusive time window. If provided, trace points are filtered to [start, end].

    Returns
    -------
    Dict[int, pd.DataFrame]
        Mapping charger_id -> DataFrame with columns:
        ['timestamp', 'soc', 'session_id'(optional), 'vehicle_name'(optional)].
    """
    charger_traces_dataframe = _require_non_empty_dataframe(
        charger_traces_dataframe, "charger_traces_dataframe"
    )

    dataframe = charger_traces_dataframe.copy()
    dataframe = _filter_time_window(dataframe, "timestamp", start=start, end=end)

    state_of_charge_column_name = _get_column_name(
        dataframe,
        ["state_of_charge", "state_of_charge_fraction", "soc", "state_of_charge_trace"],
    )
    if state_of_charge_column_name is None:
        raise ValueError(
            "No SoC column found (state_of_charge/state_of_charge_fraction/soc/state_of_charge_trace)."
        )

    has_session_id = "session_id" in dataframe.columns
    has_vehicle_name = "vehicle_name" in dataframe.columns

    output_by_charger_id: Dict[int, pd.DataFrame] = {}

    for charger_id in charger_ids:
        charger_dataframe = dataframe[dataframe["charger_id"] == int(charger_id)]
        if len(charger_dataframe) == 0:
            continue

        columns_to_take = ["timestamp", state_of_charge_column_name]
        if has_session_id:
            columns_to_take.append("session_id")
        if has_vehicle_name:
            columns_to_take.append("vehicle_name")

        out = charger_dataframe[columns_to_take].copy()
        out = out.rename(columns={state_of_charge_column_name: "soc"})
        out["soc"] = out["soc"].astype(float).fillna(0.0)

        # Keep ordering stable for plotting/grouping
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp")

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
    Validiert gemessene/zugewiesene Ladeleistung gegen:
      - Charger-Limit (rated_power_kw)
      - Vehicle-Limit aus Masterkurve (SoC->max kW an Batterie, über Effizienz auf Standortseite)

    Rückgabe:
      (validation_dataframe, violations_dataframe)
    """
    if charger_traces_dataframe is None or len(charger_traces_dataframe) == 0:
        return pd.DataFrame(), pd.DataFrame()

    dataframe = charger_traces_dataframe.copy()
    dataframe = _filter_time_window(dataframe, "timestamp", start=start, end=end)

    power_column_name = _get_column_name(dataframe, ["power_kw", "power_kw_site", "site_power_kw"])
    if power_column_name is None:
        raise ValueError("charger_traces_dataframe: keine Leistungsspalte gefunden (power_kw / power_kw_site / site_power_kw).")

    state_of_charge_column_name = _get_column_name(dataframe, ["state_of_charge", "state_of_charge_fraction", "soc"])
    if state_of_charge_column_name is None:
        raise ValueError("charger_traces_dataframe: keine SoC-Spalte gefunden (state_of_charge / state_of_charge_fraction / soc).")

    if "vehicle_name" not in dataframe.columns:
        raise ValueError("charger_traces_dataframe: vehicle_name fehlt.")

    rated_power_kw = float(scenario["site"]["rated_power_kw"])

    vehicle_curves_by_name = read_vehicle_load_profiles_from_csv(str(scenario["vehicles"]["vehicle_curve_csv"]))

    validation_rows: List[dict] = []

    for _, row in dataframe.iterrows():
        vehicle_name = str(row["vehicle_name"])
        curve = vehicle_curves_by_name.get(vehicle_name)

        power_kw_site = float(row[power_column_name])
        state_of_charge_fraction = float(row[state_of_charge_column_name])

        charger_limit_ok = power_kw_site <= rated_power_kw + 1e-9

        vehicle_limit_kw_site = np.nan
        vehicle_limit_ok = True

        if curve is not None:
            power_kw_at_battery = float(
                np.interp(
                    float(np.clip(state_of_charge_fraction, 0.0, 1.0)),
                    curve.state_of_charge_fraction,
                    curve.power_kw,
                )
            )
            power_kw_at_battery = max(power_kw_at_battery, 0.0)
            vehicle_limit_kw_site = power_kw_at_battery
            vehicle_limit_ok = power_kw_site <= vehicle_limit_kw_site + 1e-9

        validation_rows.append(
            {
                "timestamp": pd.to_datetime(row["timestamp"]),
                "charger_id": int(row["charger_id"]) if "charger_id" in row else np.nan,
                "session_id": str(row["session_id"]) if "session_id" in row else "",
                "vehicle_name": vehicle_name,
                "state_of_charge": state_of_charge_fraction,
                "power_kw_site": power_kw_site,
                "charger_limit_kw_site": rated_power_kw,
                "vehicle_limit_kw_site": float(vehicle_limit_kw_site) if np.isfinite(vehicle_limit_kw_site) else np.nan,
                "ok_charger_limit": bool(charger_limit_ok),
                "ok_vehicle_limit": bool(vehicle_limit_ok),
            }
        )

    validation_dataframe = pd.DataFrame(validation_rows)

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
    Liefert Heatmap-Matrix für Ladepunktleistung.

    Rückgabe:
      - matrix (np.ndarray)
      - charger_ids (List[int])
      - timestamps (List[pd.Timestamp])
    """
    charger_traces_dataframe = _require_non_empty_dataframe(charger_traces_dataframe, "charger_traces_dataframe")

    dataframe = charger_traces_dataframe.copy()
    dataframe = _filter_time_window(dataframe, "timestamp", start=start, end=end)

    power_column_name = _get_column_name(dataframe, ["power_kw", "power_kw_site", "site_power_kw"])
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
    Bereitet Daten für eine Standortübersicht (Notebook-Plot) vor.

    Liefert:
      - dataframe: gefilterter DF (mit 'timestamp' als datetime)
      - base_load_kw: pd.Series
      - ev_load_kw: pd.Series
      - total_load_kw: pd.Series
      - pv_generation_kw: Optional[pd.Series]
      - grid_limit_kw: float
      - charger_limit_kw_total: float

    WICHTIG:
      - keine plt-Aufrufe
      - keine print-Ausgaben
      - 'Grid + ChargerTotal' wird nur als Zahl geliefert, nicht geplottet
    """
    timeseries_dataframe = _require_non_empty_dataframe(timeseries_dataframe, "timeseries_dataframe")

    dataframe = timeseries_dataframe.copy()
    dataframe = _filter_time_window(dataframe, "timestamp", start=start, end=end)

    grid_limit_kw = float(scenario["site"]["grid_limit_p_avb_kw"])
    charger_limit_kw_total = float(scenario["site"]["rated_power_kw"]) * float(scenario["site"]["number_chargers"])

    if "base_load_kw" in dataframe.columns:
        base_load_kw = dataframe["base_load_kw"].astype(float).fillna(0.0)
    else:
        base_load_kw = pd.Series(np.zeros(len(dataframe), dtype=float), index=dataframe.index)

    if "ev_load_kw" in dataframe.columns:
        ev_load_kw = dataframe["ev_load_kw"].astype(float).fillna(0.0)
    else:
        ev_load_kw = pd.Series(np.zeros(len(dataframe), dtype=float), index=dataframe.index)

    total_load_kw = (base_load_kw + ev_load_kw).fillna(0.0)

    pv_generation_kw = None
    if "pv_generation_kw" in dataframe.columns:
        pv_generation_kw = dataframe["pv_generation_kw"].astype(float).fillna(0.0)

    return {
        "dataframe": dataframe,
        "base_load_kw": base_load_kw,
        "ev_load_kw": ev_load_kw,
        "total_load_kw": total_load_kw,
        "pv_generation_kw": pv_generation_kw,
        "grid_limit_kw": grid_limit_kw,
        "charger_limit_kw_total": charger_limit_kw_total,
    }


def build_ev_power_by_mode_timeseries_dataframe(
    timeseries_dataframe: pd.DataFrame,
    sessions_out: Optional[List[dict]] = None,
    scenario: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Erstellt ein DataFrame mit EV-Leistung nach Modus für Notebook-Plots.

    Ergebnis-Spalten:
      - timestamp
      - ev_generation_kw
      - ev_market_kw
      - ev_immediate_kw

    Datenquellen:
    A) Wenn timeseries_dataframe bereits Spalten enthält:
         ev_generation_kw, ev_market_kw, ev_immediate_kw
       -> werden direkt übernommen.
    B) Wenn nicht vorhanden, aber sessions_out + scenario gegeben:
       -> Aggregation aus plan_*_kwh_per_step über arrival_step/departure_step.

    WICHTIG:
      - keine plt-Aufrufe
      - keine print-Ausgaben
    """
    timeseries_dataframe = _require_non_empty_dataframe(timeseries_dataframe, "timeseries_dataframe")

    dataframe = timeseries_dataframe.copy()
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])

    has_generation = "ev_generation_kw" in dataframe.columns
    has_market = "ev_market_kw" in dataframe.columns
    has_immediate = "ev_immediate_kw" in dataframe.columns

    if has_generation and has_market and has_immediate:
        return pd.DataFrame(
            {
                "timestamp": dataframe["timestamp"],
                "ev_generation_kw": dataframe["ev_generation_kw"].astype(float).fillna(0.0),
                "ev_market_kw": dataframe["ev_market_kw"].astype(float).fillna(0.0),
                "ev_immediate_kw": dataframe["ev_immediate_kw"].astype(float).fillna(0.0),
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

        plan_generation = np.array(session.get("plan_pv_site_kwh_per_step", []), dtype=float)
        plan_market = np.array(session.get("plan_market_site_kwh_per_step", []), dtype=float)
        plan_immediate = np.array(session.get("plan_immediate_site_kwh_per_step", []), dtype=float)

        if len(plan_generation) == window_length:
            generation_kwh_per_step[arrival_step:departure_step] += plan_generation
        if len(plan_market) == window_length:
            market_kwh_per_step[arrival_step:departure_step] += plan_market
        if len(plan_immediate) == window_length:
            immediate_kwh_per_step[arrival_step:departure_step] += plan_immediate

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

# Ladekurvenvalidierung: BEV-spezifische Ladekurve vs. Ist-Ladekurve

def get_most_used_vehicle_name(
    sessions_out: Optional[list[dict]] = None,
    charger_traces_dataframe: Optional[pd.DataFrame] = None,
    only_plugged_sessions: bool = True,
) -> str:
    """
    Determine the most-used vehicle name.

    Priority:
    1) If sessions_out is provided: count vehicle_name by sessions (recommended).
       If only_plugged_sessions=True, only sessions with status == 'plugged' are counted.
    2) Otherwise, fall back to counting occurrences in charger_traces_dataframe.

    Returns
    -------
    str
        Vehicle name with the highest count.
    """
    if sessions_out is not None and len(sessions_out) > 0:
        if only_plugged_sessions:
            vehicle_names = [s.get("vehicle_name") for s in sessions_out if s.get("status") == "plugged"]
        else:
            vehicle_names = [s.get("vehicle_name") for s in sessions_out]

        vehicle_names = [v for v in vehicle_names if isinstance(v, str) and v.strip() != ""]
        if len(vehicle_names) > 0:
            return pd.Series(vehicle_names).value_counts().idxmax()

    if charger_traces_dataframe is None or len(charger_traces_dataframe) == 0:
        raise ValueError("Neither sessions_out (usable) nor charger_traces_dataframe (non-empty) provided.")

    if "vehicle_name" not in charger_traces_dataframe.columns:
        raise ValueError("charger_traces_dataframe: column 'vehicle_name' missing.")

    return charger_traces_dataframe["vehicle_name"].value_counts().idxmax()


def build_master_curve_and_actual_points_for_vehicle(
    charger_traces_dataframe: pd.DataFrame,
    scenario: dict,
    vehicle_name: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """
    Build plot/validation data for one vehicle:

    - Master curve from vehicle CSV: maximum battery power [kW] vs SoC [-]
    - Actual points from simulation traces: charged power [kW] vs SoC [-]
    - Violations: actual power > master max power at the same SoC

    Parameters
    ----------
    charger_traces_dataframe:
        Trace points from the simulation. Must contain at least:
        - timestamp
        - vehicle_name
        - power column (e.g. power_kw / power_kw_site / site_power_kw)
        - SoC column (e.g. state_of_charge / soc / ...)
    scenario:
        Scenario dict containing vehicles.vehicle_curve_csv.
    vehicle_name:
        Vehicle name to filter for (must exist in the vehicle curve CSV).
    start, end:
        Optional time window filter (inclusive). Use None for full horizon.

    Returns
    -------
    Dict[str, Any]
        Keys used by the notebook plot:
          - vehicle_name
          - master_soc
          - master_power_battery_kw
          - actual_soc
          - actual_power_kw
          - allowed_power_kw_at_actual
          - violation_mask
          - number_violations
          - number_points
    """
    if charger_traces_dataframe is None or len(charger_traces_dataframe) == 0:
        raise ValueError("charger_traces_dataframe is empty (did you set record_charger_traces=True?).")

    dataframe = charger_traces_dataframe.copy()

    # Required columns
    for column_name in ["timestamp", "vehicle_name"]:
        if column_name not in dataframe.columns:
            raise ValueError(f"charger_traces_dataframe: column '{column_name}' missing.")

    power_column_name = _get_column_name(dataframe, ["power_kw", "power_kw_site", "site_power_kw"])
    if power_column_name is None:
        raise ValueError("charger_traces_dataframe: no power column found (power_kw/power_kw_site/site_power_kw).")

    soc_column_name = _get_column_name(
        dataframe, ["state_of_charge", "state_of_charge_fraction", "soc", "state_of_charge_trace"]
    )
    if soc_column_name is None:
        raise ValueError("charger_traces_dataframe: no SoC column found (state_of_charge/.../soc).")

    # Parse + filter time window
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], errors="coerce")
    dataframe = dataframe.dropna(subset=["timestamp"])

    if start is not None:
        dataframe = dataframe[dataframe["timestamp"] >= pd.to_datetime(start)]
    if end is not None:
        dataframe = dataframe[dataframe["timestamp"] <= pd.to_datetime(end)]

    # Filter vehicle
    dataframe = dataframe[dataframe["vehicle_name"] == str(vehicle_name)]
    if len(dataframe) == 0:
        raise ValueError(f"No charger trace points found for vehicle_name='{vehicle_name}' in the selected window.")

    # Actual points
    actual_soc = pd.to_numeric(dataframe[soc_column_name], errors="coerce").to_numpy(dtype=float)
    actual_power_kw = pd.to_numeric(dataframe[power_column_name], errors="coerce").to_numpy(dtype=float)

    valid_mask = np.isfinite(actual_soc) & np.isfinite(actual_power_kw)
    actual_soc = actual_soc[valid_mask]
    actual_power_kw = actual_power_kw[valid_mask]

    if len(actual_soc) == 0:
        raise ValueError("All actual points are invalid after numeric conversion (soc/power).")

    # Master curve from CSV
    vehicle_curves_by_name = read_vehicle_load_profiles_from_csv(str(scenario["vehicles"]["vehicle_curve_csv"]))
    if str(vehicle_name) not in vehicle_curves_by_name:
        example_names = list(vehicle_curves_by_name.keys())[:10]
        raise ValueError(f"Vehicle '{vehicle_name}' not found in master curves. Example names: {example_names}")

    curve = vehicle_curves_by_name[str(vehicle_name)]
    master_soc = np.array(curve.state_of_charge_fraction, dtype=float)
    master_power_battery_kw = np.array(curve.power_kw, dtype=float)

    # Allowed power at each actual SoC (battery-side, same unit)
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
    sessions_out: list[dict],
    charger_traces_dataframe: pd.DataFrame,
    scenario: dict,
) -> Dict[str, Any]:
    """
    Select the globally most-used vehicle (based on sessions_out) and build master-curve
    vs actual charging points over the full simulation horizon.

    Returns
    -------
    Dict[str, Any]
        Keys:
          - vehicle_name
          - plot_data (dict): output of build_master_curve_and_actual_points_for_vehicle(...)
    """
    if charger_traces_dataframe is None or len(charger_traces_dataframe) == 0:
        raise ValueError("charger_traces_dataframe is empty (record_charger_traces=True?).")

    vehicle_name_global = get_most_used_vehicle_name(
        sessions_out=sessions_out,
        charger_traces_dataframe=charger_traces_dataframe,
        only_plugged_sessions=True,
    )

    plot_data = build_master_curve_and_actual_points_for_vehicle(
        charger_traces_dataframe=charger_traces_dataframe,
        scenario=scenario,
        vehicle_name=vehicle_name_global,
        start=None,
        end=None,
    )

    return {
        "vehicle_name": vehicle_name_global,
        "plot_data": plot_data,
    }

# =============================================================================
# Notebook helpers (optional)
# =============================================================================

def show_strategy_status(charging_strategy: str, strategy_status: str) -> None:
    """
    Print a short status block for the chosen charging strategy.

    Parameters
    ----------
    charging_strategy:
        Strategy name from scenario (e.g. "immediate", "market", "generation").
    strategy_status:
        Status label (e.g. "ACTIVE", "IMMEDIATE").
    """
    strategy_name = (charging_strategy or "immediate").capitalize()
    status_name = (strategy_status or "immediate").capitalize()
    print(f"Charging Strategy: {strategy_name}")
    print(f"Strategy Status: {status_name}")


def decorate_title_with_status(base_title: str, charging_strategy: str, strategy_status: str) -> str:
    """
    Build a plot title that includes charging strategy and status.

    Parameters
    ----------
    base_title:
        Base plot title.
    charging_strategy:
        Strategy name.
    strategy_status:
        Status label.

    Returns
    -------
    str
        Combined title string.
    """
    strategy_name = (charging_strategy or "immediate").capitalize()
    status_name = (strategy_status or "immediate").capitalize()
    return f"{base_title} ({strategy_name}, {status_name})"


def initialize_time_window(
    timestamps: pd.DatetimeIndex,
    scenario: dict,
    days: int = 1,
) -> tuple[Optional[int], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Create a zoom window for plots based on the first N days of the simulation.

    Parameters
    ----------
    timestamps:
        Simulation timestamps.
    scenario:
        Scenario dict (must include "time_resolution_min").
    days:
        Number of days to include in the zoom window.

    Returns
    -------
    (steps_per_day, window_start, window_end)
    """
    if timestamps is None or len(timestamps) == 0:
        return None, None, None

    time_resolution_min = int(scenario["time_resolution_min"])
    steps_per_day = int(24 * 60 / time_resolution_min)
    steps_total = int(max(1, days)) * steps_per_day

    window_start = pd.to_datetime(timestamps[0])
    window_end = pd.to_datetime(timestamps[min(len(timestamps) - 1, steps_total - 1)])
    return steps_per_day, window_start, window_end


def get_holiday_dates_from_scenario(scenario: dict, timestamps: pd.DatetimeIndex) -> list[datetime]:
    """
    Read holiday dates from scenario configuration.

    Supports:
      - Manual ISO dates in scenario["holidays"]["dates"]
      - Optional python-holidays lookup (country + subdivision)

    Parameters
    ----------
    scenario:
        Scenario dict.
    timestamps:
        Simulation timestamps (used to derive years).

    Returns
    -------
    list[datetime]
        Unique holiday dates normalized to date (00:00).
    """
    holidays_configuration = (scenario.get("holidays") or {})
    manual_dates = holidays_configuration.get("dates") or []
    holiday_dates: list[datetime] = []

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


def get_daytype_calendar(start_datetime: datetime, horizon_days: int, holiday_dates: list[datetime]) -> dict:
    """
    Create a day-type calendar (working_day/saturday/sunday_holiday) over the simulation horizon.

    NOTE: This function expects that _get_day_type(...) exists in this module.

    Parameters
    ----------
    start_datetime:
        Start of simulation day 0.
    horizon_days:
        Number of simulated days.
    holiday_dates:
        Holiday list.

    Returns
    -------
    dict
        {"working_day": [...], "saturday": [...], "sunday_holiday": [...]}
    """
    out = {"working_day": [], "saturday": [], "sunday_holiday": []}
    for day_index in range(int(horizon_days)):
        day_start = start_datetime + timedelta(days=day_index)
        day_type = _get_day_type(day_start, holiday_dates)  # <-- IMPORTANT: no "sim."
        out[day_type].append(day_start.date())
    return out


def group_sessions_by_day(sessions_out: list[dict], only_plugged: bool = False) -> dict:
    """
    Group sessions by arrival date.

    Parameters
    ----------
    sessions_out:
        Simulation output session dicts.
    only_plugged:
        If True, only sessions with status == "plugged" are included.

    Returns
    -------
    dict
        {date: [session_dict, ...], ...}
    """
    grouped: dict = {}
    for session in sessions_out:
        if only_plugged and session.get("status") != "plugged":
            continue
        day = pd.to_datetime(session.get("arrival_time")).date()
        grouped.setdefault(day, []).append(session)
    return grouped


def resolve_paths_relative_to_yaml(scenario: dict, scenario_path: str) -> dict:
    """
    Resolve relative CSV paths in a loaded scenario dict relative to the YAML file location.

    Parameters
    ----------
    scenario:
        Loaded scenario dict.
    scenario_path:
        Path to scenario YAML.

    Returns
    -------
    dict
        Scenario copy with absolute paths.
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
    Build the central timeseries DataFrame for plotting.

    This is a thin wrapper around build_timeseries_dataframe(...) that also ensures
    a pandas datetime column "ts" exists.

    Returns
    -------
    pd.DataFrame
    """
    dataframe = build_timeseries_dataframe(  # <-- IMPORTANT: no "sim."
        timestamps=timestamps,
        ev_load_kw=ev_load_kw,
        scenario=scenario,
        debug_rows=debug_rows,
        generation_series=pv_generation_series,
        market_series=market_price_series,
    )
    if "timestamp" in dataframe.columns and "ts" not in dataframe.columns:
        dataframe["ts"] = pd.to_datetime(dataframe["timestamp"])
    return dataframe
