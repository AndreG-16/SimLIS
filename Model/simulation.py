from __future__ import annotations

import yaml
import numpy as np
import pandas as pd
import warnings

from dataclasses import dataclass
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
    """
    Erzeugt den tz-aware Simulationszeitindex (DST-robust).

    Die Simulation arbeitet konsequent tz-aware ("Europe/Berlin").
    `simulation_horizon_days` wird als Kalendertage interpretiert, damit DST-Tage
    automatisch 23h bzw. 25h haben.

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
    """
    Lokalisiert einen tz-naiven Wandzeit-Index DST-robust in eine Zeitzone.
    
    Diese Hilfsfunktion geht davon aus, dass die Zeitstempel aus der CSV lokale Wandzeit
    ohne Zeitzoneninformation sind.
    
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
        - charger_power_kw
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
        "charger_power_kw",
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

        Interpretation und Skalierung (jetzt in kW)
        -------------------------------------------
        Wenn `profiles_are_normalized=True`:
        - Grundlast (Spalte 1) ist ein normiertes Profil (Gewichte) und wird so skaliert,
        dass die Summe über das Jahr `base_load_annual_kwh` entspricht. Danach Umrechnung in Leistung [kW].
        - PV (Spalte 2) ist ein Kapazitätsfaktor (0..1) und wird mit `pv_system_size_kwp`
        direkt in Leistung [kW] umgerechnet.

        Wenn `profiles_are_normalized=False`:
        - Spalte 1 und 2 werden als bereits vorliegende Leistungen [kW] je CSV-Zeitschritt interpretiert.

        Rückgabe
        --------
        - Grundlast und PV als *kW pro Simulationsschritt* (mittlere Leistung je Schritt).
        - Marktpreis als *€/MWh* je Simulationsschritt.

        Resampling / Alignment
        ----------------------
        - Wenn Simulationstakt == CSV-Takt: reindex (ohne Interpolation).
        - Wenn Simulation feiner als CSV (Upsampling): Forward-Fill der Leistung.
        - Wenn Simulation gröber als CSV (Downsampling): Resample der Leistung per Mittelwert.

        Ausnahmen
        ---------
        ValueError
            - wenn `timestamps` tz-naiv ist
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
        raise ValueError(
            "`timestamps` muss mindestens zwei Einträge enthalten. "
            f"Aktuell: len={len(simulation_time_index)}. "
            "Prüfe in der YAML insbesondere 'simulation_horizon_days' (sollte > 0 sein), "
            "Start/Ende der Simulation und 'time_resolution_min'."
        )

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
    input_step_hours = float(input_time_resolution_min) / 60.0

    # Profile interpretieren / skalieren (auf Input-Gitter), Ergebnis jeweils in kW.
    base_profile = dataframe[base_col]
    pv_profile = dataframe[pv_col]
    price_series = dataframe[price_col]

    if profiles_are_normalized:
        # Grundlast: normierte Gewichte -> skaliere auf Jahresenergie -> dann in kW umrechnen.
        total_weight = float(base_profile.sum())
        if base_load_annual_kwh <= 0.0 or total_weight <= 0.0 or input_step_hours <= 0.0:
            base_kw_input = base_profile * 0.0
        else:
            base_kwh_input = (base_profile / total_weight) * float(base_load_annual_kwh)
            base_kw_input = base_kwh_input / input_step_hours

        # PV: Kapazitätsfaktor -> direkt kW.
        pv_kw_input = pv_profile * float(max(pv_system_size_kwp, 0.0))
    else:
        # Profile sind bereits Leistungen [kW] je CSV-Zeitschritt.
        base_kw_input = base_profile
        pv_kw_input = pv_profile

    # Leistung auf Simulationsraster ausrichten (resample/reindex).
    if sim_step_min == int(input_time_resolution_min):
        # Gleiche Auflösung: direkt auf Raster legen.
        base_kw_aligned = base_kw_input.reindex(simulation_time_index)
        pv_kw_aligned = pv_kw_input.reindex(simulation_time_index)
        price_aligned = price_series.reindex(simulation_time_index, method="ffill")

    elif sim_step_min < int(input_time_resolution_min):
        # Simulation feiner als CSV: stückweise konstante Leistung bis zur nächsten Messung.
        base_kw_aligned = base_kw_input.reindex(simulation_time_index, method="ffill")
        pv_kw_aligned = pv_kw_input.reindex(simulation_time_index, method="ffill")
        price_aligned = price_series.reindex(simulation_time_index, method="ffill")

    else:
        # Simulation gröber als CSV: mittlere Leistung über das größere Intervall.
        rule = f"{sim_step_min}min"
        base_kw_aligned = base_kw_input.resample(rule).mean().reindex(simulation_time_index)
        pv_kw_aligned = pv_kw_input.resample(rule).mean().reindex(simulation_time_index)
        price_aligned = price_series.resample(rule).mean().reindex(simulation_time_index, method="ffill")

    # Fehlende Werte als 0 interpretieren (z. B. außerhalb CSV-Bereich).
    base_kw_aligned = base_kw_aligned.fillna(0.0)
    pv_kw_aligned = pv_kw_aligned.fillna(0.0)

    return (
        base_kw_aligned.astype(float),
        pv_kw_aligned.astype(float),
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
            vehicle_class=vehicle_class,
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


def define_sample_from_distribution(spec: Any, random_generator: np.random.Generator) -> float:
    """
    Zieht genau einen Sample-Wert aus einer Verteilungsspezifikation.

        Unterstützte Eingaben
        ---------------------
        - Konstante: 3.5
        - Uniform-Range: [min, max]
        - Einzelkomponente: {"distribution": "normal|beta"}
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

    raise ValueError(f"Unbekannte distribution in spec: '{distribution_name}'")


def sample_vehicle_for_session(
    vehicle_curves_by_name: Dict[str, "VehicleChargingCurve"],
    fleet_mix: Dict[str, float],
    random_generator: np.random.Generator,
    vehicle_type: Optional[Any] = None,
) -> Tuple[str, str]:
    """
    Wählt ein Fahrzeug für eine Ladesession.

        Auswahllogik
        ------------
        1) Zuerst wird die Fahrzeugklasse gewählt:
        - Wenn `fleet_mix` gesetzt ist: gemäß Gewichten (renormalisiert auf verfügbare Klassen).
        - Sonst: gleichverteilt über verfügbare Klassen.

        2) Dann wird ein Fahrzeugname innerhalb der gewählten Klasse gewählt:
        - Wenn `vehicle_type` gesetzt ist:
            * Für Klassen, in denen mindestens ein Name angegeben wurde, wird nur aus diesen Namen gewählt.
            * Für Klassen ohne Angabe in `vehicle_type` bleiben alle Fahrzeuge der Klasse auswählbar.
        - Wenn `vehicle_type` nicht gesetzt ist: alle Fahrzeuge der Klasse sind auswählbar.

        Parameter
        ----------
        vehicle_curves_by_name:
            Zuordnung {vehicle_name: VehicleChargingCurve}.
        fleet_mix:
            Klassen-Gewichte, z. B. {"PKW": 0.98, "Transporter": 0.02}. Kann leer sein.
        random_generator:
            Numpy RNG.
        vehicle_type:
            Optional: String oder Liste[String], max. 10 Fahrzeugnamen.
            Wirkt als Whitelist innerhalb der jeweiligen Fahrzeugklassen.

        Rückgabe
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

    # ---- vehicle_type normalisieren (optional) ----
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
        elif len(selected_vehicle_names) > 10:
            raise ValueError("vehicles.vehicle_type darf maximal 10 Fahrzeuge enthalten.")

    # ---- pro Klasse erlaubte Fahrzeuge bauen ----
    allowed_names_by_class: Dict[str, List[str]] = {}
    for name, curve in vehicle_curves_by_name.items():
        cls = str(curve.vehicle_class)
        allowed_names_by_class.setdefault(cls, []).append(str(name))

    # Falls vehicle_type gesetzt: nur innerhalb der betroffenen Klassen einschränken
    if selected_vehicle_names is not None:
        unknown_names = [name for name in selected_vehicle_names if name not in vehicle_curves_by_name]
        if unknown_names:
            raise ValueError(f"vehicles.vehicle_type enthält unbekannte Fahrzeugnamen: {unknown_names}")

        restricted_by_class: Dict[str, List[str]] = {}
        for name in selected_vehicle_names:
            cls = str(vehicle_curves_by_name[name].vehicle_class)
            restricted_by_class.setdefault(cls, []).append(str(name))

        # In Klassen, die in vehicle_type vorkommen, wird die erlaubte Liste durch die Restriktion ersetzt.
        for cls, names in restricted_by_class.items():
            allowed_names_by_class[cls] = list(dict.fromkeys(names))

    # Verfügbare Klassen = Klassen mit mind. einem Fahrzeug
    available_classes = sorted([cls for cls, names in allowed_names_by_class.items() if len(names) > 0])
    if len(available_classes) == 0:
        raise ValueError("Keine Fahrzeugklassen verfügbar.")

    # ---- Klasse wählen ----
    if fleet_mix:
        class_names = [cls for cls in fleet_mix.keys() if cls in available_classes]
        if len(class_names) == 0:
            raise ValueError("fleet_mix passt zu keiner verfügbaren Fahrzeugklasse.")
        class_weights = np.array([float(fleet_mix[cls]) for cls in class_names], dtype=float)
        if np.any(class_weights < 0.0) or float(class_weights.sum()) <= 0.0:
            raise ValueError("fleet_mix Gewichte müssen >= 0 sein und eine positive Summe haben.")
        chosen_class = str(random_generator.choice(class_names, p=class_weights / float(class_weights.sum())))
    else:
        chosen_class = str(random_generator.choice(available_classes))

    # ---- Fahrzeug innerhalb der Klasse wählen ----
    vehicle_names_in_class = allowed_names_by_class.get(chosen_class, [])
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
    Zieht stochastische Parameter für eine einzelne Ladesession aus den im Szenario
    hinterlegten Verteilungen.

    Die Funktion sampelt (abhängig vom `day_type`) eine Ankunftszeit, eine Parkdauer
    sowie einen SoC bei Ankunft und wählt anschließend ein Fahrzeug (Name und Klasse)
    gemäß Fleet-Mix und optionaler Fahrzeug-Filterung (`vehicles.vehicle_type`).

        Parameter
        ----------
        scenario:
            Szenario-Konfiguration mit den relevanten Verteilungsdefinitionen, u. a.
            `arrival_time_distribution`, `parking_duration_distribution`,
            `soc_at_arrival_distribution` sowie `vehicles` (inkl. `fleet_mix`,
            optional `vehicle_type`).
        day_type:
            Tageskategorie zur Auswahl der passenden Ankunftsverteilung, z. B.
            `"workday"`, `"weekend"`, `"holiday"` (abhängig vom Szenario).
        random_generator:
            Numpy-Zufallszahlengenerator für reproduzierbares Sampling.
        vehicle_curves_by_name:
            Mapping von `vehicle_name` auf `VehicleChargingCurve` zur Fahrzeugauswahl.

        Rückgabe
        --------
        Tuple[float, float, float, str, str]
            (arrival_hours, parking_duration_minutes, soc_at_arrival, vehicle_name, vehicle_class)

            - arrival_hours:
                Ankunftszeit als Stunden seit Tagesbeginn (z. B. 7.5 für 07:30).
            - parking_duration_minutes:
                Parkdauer in Minuten (nach Clamping auf Min/Max aus dem Szenario).
            - soc_at_arrival:
                SoC bei Ankunft als Anteil [0..1] (nach Clamping auf [0..max_soc]).
            - vehicle_name:
                Name des ausgewählten Fahrzeugs.
            - vehicle_class:
                Fahrzeugklasse des ausgewählten Fahrzeugs.

        Raises
        ------
        ValueError
            Wenn für `day_type` keine gültige Ankunftsverteilung definiert ist, wenn
            Parkdauer- oder SoC-Verteilungen fehlen/ungültig sind, wenn
            `max_duration_minutes < min_duration_minutes` oder wenn
            `soc_at_arrival_distribution.max_soc` nicht in [0, 1] liegt.
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

def order_sessions_for_planning(
    sessions: list[SampledSession],
    charging_strategy: str,
) -> list[SampledSession]:
    """
    Sortiert Ladesessions in der Reihenfolge, in der sie in der Tagesplanung
    bearbeitet werden sollen.

    Die Sortierlogik hängt von der konfigurierten Lade-Strategie ab:

    - `"immediate"`: First-Come-First-Served (FCFS) – frühere Ankunft wird zuerst geplant.
    - alle anderen Strategien (z. B. `"market"`, `"pv"`, `"generation"`):
    kürzeste Standzeit zuerst, um kurze Aufenthalte priorisiert zu bedienen.

        Parameter
        ----------
        sessions:
            Liste der zu sortierenden Sessions.
        charging_strategy:
            Name der Lade-Strategie (z. B. `"immediate"`, `"market"`, `"pv"`). Groß-/Kleinschreibung
            wird ignoriert, führende/trailing Spaces werden entfernt.

        Rückgabe
        --------
        list[SampledSession]
            Neue Liste mit Sessions in Planungsreihenfolge (Original-Liste bleibt unverändert).
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


def available_power_for_session_step(
    *,
    step_index: int,
    scenario: dict,
    curve: VehicleChargingCurve,
    state_of_charge_fraction: float,
    remaining_site_energy_kwh: float,
    pv_generation_kw_per_step: np.ndarray,
    base_load_kw_per_step: np.ndarray,
    reserved_total_ev_power_kw_per_step: np.ndarray,
    reserved_pv_ev_power_kw_per_step: Optional[np.ndarray] = None,
    already_allocated_on_this_charger_kw: float = 0.0,
    supply_mode: Literal["site", "pv_only", "grid_only"] = "site",
) -> Tuple[float, float]:
    """
    Bestimmt die maximal allokierbare Ladeleistung (kW) für eine Session in einem Schritt.

    Die verfügbare Leistung ergibt sich aus PV und Netzlimit nach Deckung der Grundlast sowie nach
    Abzug bereits reservierter EV-Leistung. Zusätzlich wird die Allokation durch das Ladepunktlimit,
    die Fahrzeug-Ladekurve (SoC-abhängige maximale Ladeleistung) und den Restenergiebedarf begrenzt.

    Vor der EV-Allokation wird geprüft, ob die Grundlast im Schritt durch PV und Netzlimit versorgbar
    ist. Ist dies nicht möglich, wird ein Fehler ausgelöst.

        Parameter
        ----------
        step_index:
            Absoluter Index im Simulationsraster.
        scenario:
            Szenario-Dictionary mit mindestens ``site.charger_power_kw``, ``site.grid_limit_p_avb_kw``,
            ``site.charger_efficiency`` und ``time_resolution_min``.
        curve:
            Fahrzeug-Ladekurve (SoC-Stützstellen und Ladeleistung auf Batterieseite) in kW.
        state_of_charge_fraction:
            SoC zu Beginn des Schritts als Anteil [0..1].
        remaining_site_energy_kwh:
            Noch benötigte Energie auf Standortseite (kWh). Wird intern über ``step_hours`` in eine
            äquivalente maximale Schritt-Leistungsgrenze umgerechnet: ``need_limit_kw = remaining_kwh / step_hours``.
        pv_generation_kw_per_step, base_load_kw_per_step:
            Zeitreihen für PV-Erzeugung und Grundlast in kW (mittlere Leistung je Simulationsschritt).
        reserved_total_ev_power_kw_per_step, reserved_pv_ev_power_kw_per_step:
            Bereits reservierte EV-Leistung (gesamt und PV-Anteil) in kW je Schritt.
        already_allocated_on_this_charger_kw:
            Bereits im selben Schritt auf demselben Ladepunkt allokierte Leistung (kW).
        supply_mode:
            Versorgungsmodus: ``"site"`` (PV+Netz), ``"pv_only"`` (nur PV) oder ``"grid_only"`` (nur Netz).

        Rückgabe
        -------
        Tuple[float, float]
            ``(allocated_site_kw, pv_share_kw)`` mit der allokierten Leistung auf Standortseite und
            dem PV-Anteil daran, jeweils in kW.

        Ausnahmen
        ------
        IndexError
            Wenn ``step_index`` außerhalb der Zeitreihenlänge liegt.
        ValueError
            Wenn die Grundlast im Schritt nicht durch PV und Netzlimit versorgbar ist oder Parameter
            im Szenario ungültig sind.
    """
    if step_index < 0 or step_index >= len(base_load_kw_per_step):
        raise IndexError(f"step_index={step_index} außerhalb der Zeitreihenlänge.")

    site_configuration = scenario["site"]
    time_resolution_min = int(scenario["time_resolution_min"])
    step_hours = float(time_resolution_min) / 60.0
    step_hours_safe = float(max(step_hours, 1e-12))

    charger_efficiency = float(site_configuration.get("charger_efficiency", 1.0))
    if not (0.0 < charger_efficiency <= 1.0):
        raise ValueError("site.charger_efficiency muss im Bereich (0, 1] liegen.")

    pv_power_kw = float(pv_generation_kw_per_step[step_index])
    grid_power_limit_kw = float(site_configuration["grid_limit_p_avb_kw"])
    building_power_kw = float(base_load_kw_per_step[step_index])

    reserved_total_ev_power_kw = float(reserved_total_ev_power_kw_per_step[step_index])
    reserved_pv_ev_power_kw = (
        float(reserved_pv_ev_power_kw_per_step[step_index])
        if reserved_pv_ev_power_kw_per_step is not None
        else 0.0
    )

    # (1) Harte Prüfung: Gebäude muss versorgbar sein (PV zuerst, Rest aus Netz bis Limit).
    building_power_from_grid_kw = building_power_kw - float(np.minimum(pv_power_kw, building_power_kw))
    if building_power_from_grid_kw > grid_power_limit_kw + 1e-9:
        raise ValueError(
            "Strombedarf (Grundlast) höher als verfügbare Leistung am Standort: "
            f"building_power_kw={building_power_kw:.6f}, pv_power_kw={pv_power_kw:.6f}, "
            f"grid_power_limit_kw={grid_power_limit_kw:.6f}."
        )

    # (2) Verfügbare Leistung für EV (PV nach Grundlast + Grid nach Grundlast)
    pv_after_building_kw = float(max(pv_power_kw - building_power_kw, 0.0))
    pv_remaining_for_ev_kw = float(max(pv_after_building_kw - reserved_pv_ev_power_kw, 0.0))

    grid_remaining_after_building_kw = float(grid_power_limit_kw - building_power_from_grid_kw)

    # "Grid-EV bereits" = reserved_total - (physikalisch möglicher PV-Anteil)
    pv_ev_physical_kw = float(np.minimum(reserved_pv_ev_power_kw, pv_after_building_kw))
    grid_ev_already_kw = float(max(reserved_total_ev_power_kw - pv_ev_physical_kw, 0.0))
    grid_remaining_for_ev_kw = float(max(grid_remaining_after_building_kw - grid_ev_already_kw, 0.0))

    if supply_mode == "pv_only":
        supply_headroom_kw = pv_remaining_for_ev_kw
    elif supply_mode == "grid_only":
        supply_headroom_kw = grid_remaining_for_ev_kw
    elif supply_mode == "site":
        supply_headroom_kw = pv_remaining_for_ev_kw + grid_remaining_for_ev_kw
    else:
        raise ValueError("supply_mode muss 'site', 'pv_only' oder 'grid_only' sein.")

    # (3) Chargerlimit
    charger_limit_kw = float(site_configuration["charger_power_kw"])
    charger_headroom_kw = float(max(charger_limit_kw - float(already_allocated_on_this_charger_kw), 0.0))

    # (4) Fahrzeuglimit aus Ladekurve (Batterieseite -> Standortseite)
    state_of_charge = float(np.clip(state_of_charge_fraction, 0.0, 1.0))
    battery_power_kw = float(np.interp(state_of_charge, curve.state_of_charge_fraction, curve.power_kw))
    vehicle_site_limit_kw = float(max(battery_power_kw / max(charger_efficiency, 1e-12), 0.0))

    # (5) Restenergiebedarf als Leistungsgrenze für diesen Schritt (kW)
    remaining_site_energy_kwh = float(max(remaining_site_energy_kwh, 0.0))
    need_limit_kw = float(remaining_site_energy_kwh / step_hours_safe)

    allocated_site_kw = float(
        np.minimum.reduce(
            np.array(
                [supply_headroom_kw, charger_headroom_kw, vehicle_site_limit_kw, need_limit_kw],
                dtype=float,
            )
        )
    )
    allocated_site_kw = float(max(allocated_site_kw, 0.0))
    if allocated_site_kw <= 1e-12:
        return 0.0, 0.0

    if supply_mode == "pv_only":
        pv_share_kw = allocated_site_kw
    elif supply_mode == "grid_only":
        pv_share_kw = 0.0
    else:
        pv_share_kw = float(np.minimum(allocated_site_kw, pv_remaining_for_ev_kw))

    return allocated_site_kw, float(max(pv_share_kw, 0.0))


def charging_strategy_immediate(
    *,
    session_arrival_step: int,
    session_departure_step: int,
    required_site_energy_kwh: float,
    pv_generation_kw_per_step: np.ndarray,
    base_load_kw_per_step: np.ndarray,
    reserved_total_ev_power_kw_per_step: np.ndarray,
    reserved_pv_ev_power_kw_per_step: np.ndarray,
    curve: VehicleChargingCurve,
    state_of_charge_at_arrival: float,
    scenario: dict,
    preallocated_site_kw_per_step: Optional[np.ndarray] = None,
    preallocated_pv_site_kw_per_step: Optional[np.ndarray] = None,
    supply_mode: Literal["site", "pv_only", "grid_only"] = "site",
) -> Dict[str, Any]:
    """
    Vergibt Ladeleistung an eine Session nach der Immediate-Strategie
    (chronologisch laden, solange möglich).

    Die Funktion unterstützt optional eine Vorbelegung ("preallocation") je Zeitschritt,
    z. B. wenn zuvor PV-only geplant wurde und nun immediate als Fallback weitere Leistung
    hinzufügen soll. Die Vorbelegung wird:
      - in die Plan-Zeitreihen übernommen,
      - in der SoC-/Restenergie-Fortschreibung berücksichtigt,
      - und als bereits belegte Ladepunktleistung an ``available_power_for_session_step(...)``
        übergeben (``already_allocated_on_this_charger_kw``), damit das Ladepunktlimit korrekt
        eingehalten wird.

    In jedem Simulationsschritt innerhalb der Standzeit wird die maximal allokierbare
    Zusatz-Leistung über ``available_power_for_session_step(...)`` bestimmt. Diese Leistung wird

    1) in die session-lokalen Plan-Zeitreihen geschrieben und
    2) in die globalen Reservierungs-Zeitreihen addiert (in-place),

    sodass nachfolgende Sessions die bereits belegte Standortkapazität berücksichtigen.

        Einheiten und Konventionen
        -------------------------
        - ``plan_*_kw_per_step`` und ``reserved_*_kw_per_step`` sind mittlere Leistungen
          je Simulationsschritt in kW.
        - KPI-/Zustandsgrößen bleiben Energien in kWh:
          die pro Schritt zugewiesene Energie ergibt sich als ``alloc_kw * step_hours``.
        - Der SoC wird mit Batterieseiten-Energie aktualisiert:
          ``battery_energy_kwh = alloc_site_kwh * charger_efficiency``.
        - Masterkurve/Leistungslimits in ``available_power_for_session_step`` werden je nach
          ``supply_mode`` aus PV und/oder Netz abgeleitet.

        Parameter
        ----------
        session_arrival_step:
            Ankunfts-Schritt (inklusive) im Simulationsraster.
        session_departure_step:
            Abfahrts-Schritt (exklusive) im Simulationsraster.
        required_site_energy_kwh:
            Benötigte Energie auf Standortseite (kWh), um den Ziel-SoC zu erreichen.
            (Wird intern um bereits vorallokierte Energie reduziert.)
        pv_generation_kw_per_step:
            PV-Erzeugung je Simulationsschritt als Leistung (kW).
        base_load_kw_per_step:
            Grundlast je Simulationsschritt als Leistung (kW).
        reserved_total_ev_power_kw_per_step:
            Bereits reservierte EV-Ladeleistung gesamt (PV + Netz) je Schritt (kW).
            Wird in-place um die in dieser Funktion geplanten Zusatzleistungen erhöht.
        reserved_pv_ev_power_kw_per_step:
            Bereits reservierte EV-Ladeleistung aus PV je Schritt (kW).
            Wird in-place um die in dieser Funktion geplanten PV-Zusatzanteile erhöht.
        curve:
            Fahrzeug-Ladekurve (SoC-Stützstellen und Batterieseiten-Leistung).
        state_of_charge_at_arrival:
            SoC zu Beginn der Session als Anteil [0..1].
        scenario:
            Szenario-Konfiguration. Verwendet u. a.:
            ``scenario["time_resolution_min"]`` und ``scenario["site"]["charger_efficiency"]``.
        preallocated_site_kw_per_step:
            Optional: bereits geplante Standort-Leistung je Schritt (kW) innerhalb des Session-Fensters.
            Wird als Baseline in die Pläne übernommen und in SoC/Restenergie berücksichtigt.
        preallocated_pv_site_kw_per_step:
            Optional: PV-Anteil der bereits geplanten Leistung je Schritt (kW).
            Wird auf ``[0, preallocated_site_kw_per_step]`` geklemmt.
        supply_mode:
            Versorgungsmodus für die Zusatzleistung:
            - ``"site"``: PV + Netz (Standardverhalten wie bisher)
            - ``"pv_only"``: nur PV-Überschuss
            - ``"grid_only"``: nur Netz (typisch für PV-Fallback)

        Rückgabe
        --------
        Dict[str, Any]
            Session-relevante Ergebnisse:
            - ``plan_site_kw_per_step``: gesamte geplante Leistung (Baseline + Zusatz) je Schritt (kW)
            - ``plan_pv_site_kw_per_step``: PV-Anteil an der geplanten Leistung je Schritt (kW)
            - ``plan_market_kw_per_step``: in dieser Strategie 0 (Array gleicher Länge)
            - ``charged_site_kwh``: insgesamt zugewiesene Standortenergie (kWh)
            - ``charged_pv_site_kwh``: insgesamt zugewiesene PV-Energie (kWh)
            - ``remaining_site_kwh``: verbleibender Restbedarf auf Standortseite (kWh)
            - ``final_soc``: SoC am Ende der geplanten/allokierten Schritte [0..1]
    """
    n_total = int(len(reserved_total_ev_power_kw_per_step))

    site_cfg = scenario["site"]
    charger_efficiency = float(site_cfg.get("charger_efficiency", 1.0))

    time_resolution_min = int(scenario["time_resolution_min"])
    step_hours = float(time_resolution_min) / 60.0
    step_hours_safe = float(max(step_hours, 1e-12))

    start = int(max(0, session_arrival_step))
    end_excl = int(min(int(session_departure_step), n_total))
    window_len = int(max(0, end_excl - start))

    # -----------------------------
    # Baseline (Vorallokation) je Step
    # -----------------------------
    base_site_kw = np.zeros(window_len, dtype=float)
    base_pv_kw = np.zeros(window_len, dtype=float)

    if preallocated_site_kw_per_step is not None:
        tmp = np.asarray(preallocated_site_kw_per_step, float).reshape(-1)
        base_site_kw[: min(window_len, len(tmp))] = tmp[: min(window_len, len(tmp))]

    if preallocated_pv_site_kw_per_step is not None:
        tmp = np.asarray(preallocated_pv_site_kw_per_step, float).reshape(-1)
        base_pv_kw[: min(window_len, len(tmp))] = tmp[: min(window_len, len(tmp))]

    base_site_kw = np.maximum(base_site_kw, 0.0)
    base_pv_kw = np.clip(np.maximum(base_pv_kw, 0.0), 0.0, base_site_kw)

    # Output-Pläne: enthalten Baseline + Zusatz
    plan_site_kw_per_step = base_site_kw.copy()
    plan_pv_site_kw_per_step = base_pv_kw.copy()
    plan_market_kw_per_step = np.zeros(window_len, dtype=float)

    remaining_site_kwh = float(max(required_site_energy_kwh, 0.0))
    state_of_charge = float(np.clip(state_of_charge_at_arrival, 0.0, 1.0))

    battery_capacity_kwh = float(max(curve.battery_capacity_kwh, 1e-12))
    eff = float(charger_efficiency)

    for local_i, abs_step in enumerate(range(start, end_excl)):
        if remaining_site_kwh <= 1e-12 or state_of_charge >= 1.0 - 1e-12:
            break

        # (1) Baseline dieses Steps berücksichtigen (SoC / Restenergie)
        base_kw_i = float(base_site_kw[local_i])
        if base_kw_i > 1e-12:
            base_kwh_i = base_kw_i * step_hours_safe
            remaining_site_kwh -= base_kwh_i
            state_of_charge = float(min(1.0, state_of_charge + (base_kwh_i * eff) / battery_capacity_kwh))
            if remaining_site_kwh <= 1e-12 or state_of_charge >= 1.0 - 1e-12:
                break

        # (2) Zusatzleistung immediate (chronologisch)
        alloc_site_kw, alloc_pv_kw = available_power_for_session_step(
            step_index=int(abs_step),
            scenario=scenario,
            curve=curve,
            state_of_charge_fraction=float(state_of_charge),
            remaining_site_energy_kwh=float(remaining_site_kwh),
            pv_generation_kw_per_step=pv_generation_kw_per_step,
            base_load_kw_per_step=base_load_kw_per_step,
            reserved_total_ev_power_kw_per_step=reserved_total_ev_power_kw_per_step,
            reserved_pv_ev_power_kw_per_step=reserved_pv_ev_power_kw_per_step,
            already_allocated_on_this_charger_kw=float(base_kw_i),
            supply_mode=supply_mode,
        )

        alloc_site_kw = float(alloc_site_kw)
        if alloc_site_kw <= 1e-12:
            continue

        alloc_pv_kw = float(np.clip(float(alloc_pv_kw), 0.0, alloc_site_kw))

        plan_site_kw_per_step[local_i] += alloc_site_kw
        plan_pv_site_kw_per_step[local_i] += alloc_pv_kw

        reserved_total_ev_power_kw_per_step[abs_step] += alloc_site_kw
        reserved_pv_ev_power_kw_per_step[abs_step] += alloc_pv_kw

        alloc_site_kwh = alloc_site_kw * step_hours_safe
        remaining_site_kwh -= alloc_site_kwh

        battery_energy_kwh = alloc_site_kwh * eff
        state_of_charge = float(min(1.0, state_of_charge + battery_energy_kwh / battery_capacity_kwh))

    total_charged_site_kwh = float(np.sum(plan_site_kw_per_step) * step_hours_safe)
    total_charged_pv_kwh = float(
        np.sum(np.clip(plan_pv_site_kw_per_step, 0.0, plan_site_kw_per_step)) * step_hours_safe
    )

    return {
        "plan_site_kw_per_step": plan_site_kw_per_step,
        "plan_pv_site_kw_per_step": plan_pv_site_kw_per_step,
        "plan_market_kw_per_step": plan_market_kw_per_step, 
        "charged_site_kwh": total_charged_site_kwh,
        "charged_pv_site_kwh": total_charged_pv_kwh,
        "remaining_site_kwh": float(max(required_site_energy_kwh - total_charged_site_kwh, 0.0)),
        "final_soc": float(state_of_charge),
    }


def charging_strategy_market(
    *,
    session_arrival_step: int,
    session_departure_step: int,
    required_site_energy_kwh: float,
    market_price_eur_per_mwh: np.ndarray,
    pv_generation_kw_per_step: np.ndarray,
    base_load_kw_per_step: np.ndarray,
    reserved_total_ev_power_kw_per_step: np.ndarray,
    reserved_pv_ev_power_kw_per_step: np.ndarray,
    curve: VehicleChargingCurve,
    state_of_charge_at_arrival: float,
    scenario: dict,
    preallocated_site_kw_per_step: Optional[np.ndarray] = None,
    preallocated_pv_site_kw_per_step: Optional[np.ndarray] = None,
    supply_mode: Literal["site", "pv_only", "grid_only"] = "site",
) -> Dict[str, Any]:
    """
    Vergibt Ladeleistung an eine Session nach der Market-Strategie
    (günstige Zeitschritte nutzen).

    Die Strategie besteht aus zwei Phasen:

    1) Slot-Auswahl (preisgeführt, ohne expliziten Fallback)
    - Alle Zeitschritte der Standzeit werden nach Marktpreis aufsteigend sortiert.
    - Die billigsten Schritte werden nacheinander in eine ``allowed``-Maske aufgenommen.
    - Nach jedem Hinzufügen wird ein chronologischer Dry-Run durchgeführt
        (SoC-konsistent, ohne Side-Effects).
    - Sobald die Zielenergie erreichbar ist, wird die Slot-Auswahl beendet.
    - Ist die Zielenergie innerhalb der Standzeit nicht erreichbar, werden nach dem
        letzten Schritt automatisch alle Slots erlaubt (weil alle schrittweise hinzugefügt
        wurden). Das Fahrzeug kann dann trotzdem unter Ziel-SoC bleiben.

    2) Finale Allokation (chronologisch, mit Side-Effects)
    - In Zeitreihen-Reihenfolge wird nur in erlaubten Slots zusätzliche Leistung über
        ``available_power_for_session_step(...)`` zugewiesen.
    - Dabei werden die Plan-Zeitreihen pro Session gefüllt und die globalen Reservierungen
        (in-place) erhöht.

        Wiederverwendung / Vorallokation
        --------------------------------
        Die Funktion kann als „Market-Fallback“ für bereits geplante Leistung verwendet werden
        (z. B. PV-only zuerst, danach Market/Grid-only):

        - ``preallocated_site_kw_per_step`` und ``preallocated_pv_site_kw_per_step`` bilden
        eine Baseline, die im Dry-Run und in der finalen Allokation immer berücksichtigt wird.
        Diese Baseline wird in ``plan_site_kw_per_step`` und ``plan_pv_site_kw_per_step``
        bereits vorbefüllt.
        - ``already_allocated_on_this_charger_kw`` wird in ``available_power_for_session_step(...)``
        mit der Baseline pro Schritt gesetzt, damit das Ladepunktlimit korrekt eingehalten wird.
        - ``supply_mode`` wird an ``available_power_for_session_step(...)`` durchgereicht
        (z. B. ``"site"``, ``"pv_only"``, ``"grid_only"``).

        Einheiten und Konventionen
        --------------------------
        - ``plan_*_kw_per_step`` und ``reserved_*_kw_per_step`` sind mittlere Leistungen je
        Simulationsschritt in kW.
        - KPI-/Zustandsgrößen sind Energien in kWh:
        pro Schritt gilt ``alloc_kwh = alloc_kw * step_hours``.
        - Der SoC wird mit Batterieseiten-Energie aktualisiert:
        ``battery_energy_kwh = alloc_site_kwh * charger_efficiency``.

        Parameter
        ----------
        session_arrival_step:
            Ankunfts-Schritt (inklusive) im Simulationsraster.
        session_departure_step:
            Abfahrts-Schritt (exklusive) im Simulationsraster.
        required_site_energy_kwh:
            Benötigte Restenergie auf Standortseite (kWh), um den Ziel-SoC zu erreichen.
        market_price_eur_per_mwh:
            Marktpreis je Simulationsschritt (€/MWh); dient zur Slot-Sortierung.
        pv_generation_kw_per_step:
            PV-Erzeugung je Simulationsschritt als Leistung (kW).
        base_load_kw_per_step:
            Grundlast je Simulationsschritt als Leistung (kW).
        reserved_total_ev_power_kw_per_step:
            Bereits reservierte EV-Ladeleistung gesamt (PV + Netz) je Schritt (kW).
            Wird in-place um die in dieser Funktion geplanten Zusatzleistungen erhöht.
        reserved_pv_ev_power_kw_per_step:
            Bereits reservierte EV-Ladeleistung aus PV je Schritt (kW).
            Wird in-place um die in dieser Funktion geplanten PV-Zusatzanteile erhöht.
        curve:
            Fahrzeug-Ladekurve (SoC-Stützstellen und Batterieseiten-Leistung).
        state_of_charge_at_arrival:
            SoC zu Beginn der Session als Anteil [0..1].
        scenario:
            Szenario-Konfiguration. Verwendet u. a. ``scenario["time_resolution_min"]``
            und ``scenario["site"]["charger_efficiency"]``.
        preallocated_site_kw_per_step:
            Optional: bereits geplante Standort-Leistung je Schritt (kW) innerhalb des Session-Fensters.
            Wird als Baseline in die Pläne übernommen und in Dry-Run/Allokation berücksichtigt.
        preallocated_pv_site_kw_per_step:
            Optional: PV-Anteil der bereits geplanten Leistung je Schritt (kW).
            Wird auf ``[0, preallocated_site_kw_per_step]`` geklemmt.
        supply_mode:
            Versorgungsmodus für ``available_power_for_session_step(...)``:
            ``"site"`` (PV+Netz), ``"pv_only"`` (nur PV) oder ``"grid_only"`` (nur Netz).

        Rückgabe
        --------
        Dict[str, Any]
            Session-relevante Ergebnisse:
            - ``plan_site_kw_per_step``: gesamte geplante Leistung (Baseline + Zusatz) je Schritt (kW)
            - ``plan_pv_site_kw_per_step``: PV-Anteil der geplanten Leistung je Schritt (kW)
            - ``plan_market_kw_per_step``: Zusatzleistung dieser Strategie je Schritt (kW)
            (Hinweis: je nach Definition kann dies „Netz/Market“-Anteil repräsentieren)
            - ``charged_site_kwh``: gesamte zugewiesene Standortenergie (kWh), über alle Schritte integriert
            - ``charged_pv_site_kwh``: gesamte zugewiesene PV-Energie (kWh), über alle Schritte integriert
            - ``remaining_site_kwh``: verbleibender Restbedarf auf Standortseite (kWh)
            - ``final_soc``: SoC am Ende der geplanten/allokierten Schritte [0..1]
    """
    n_total = int(len(reserved_total_ev_power_kw_per_step))

    site_cfg = scenario["site"]
    charger_efficiency = float(site_cfg.get("charger_efficiency", 1.0))

    time_resolution_min = int(scenario["time_resolution_min"])
    step_hours = float(time_resolution_min) / 60.0
    step_hours_safe = float(max(step_hours, 1e-12))

    start = int(max(0, session_arrival_step))
    end_excl = int(min(int(session_departure_step), n_total))
    window_len = int(max(0, end_excl - start))

    required_site_energy_kwh = float(max(required_site_energy_kwh, 0.0))
    state_of_charge_at_arrival = float(np.clip(state_of_charge_at_arrival, 0.0, 1.0))

    # -----------------------------
    # Baseline (Vorallokation) je Step
    # -----------------------------
    base_site_kw = np.zeros(window_len, dtype=float)
    base_pv_kw = np.zeros(window_len, dtype=float)

    if preallocated_site_kw_per_step is not None:
        tmp = np.asarray(preallocated_site_kw_per_step, float).reshape(-1)
        base_site_kw[: min(window_len, len(tmp))] = tmp[: min(window_len, len(tmp))]
    if preallocated_pv_site_kw_per_step is not None:
        tmp = np.asarray(preallocated_pv_site_kw_per_step, float).reshape(-1)
        base_pv_kw[: min(window_len, len(tmp))] = tmp[: min(window_len, len(tmp))]

    base_site_kw = np.maximum(base_site_kw, 0.0)
    base_pv_kw = np.clip(np.maximum(base_pv_kw, 0.0), 0.0, base_site_kw)

    # Output-Pläne: enthalten Baseline + Zusatz
    plan_site_kw_per_step = base_site_kw.copy()
    plan_pv_site_kw_per_step = base_pv_kw.copy()
    plan_market_kw_per_step = np.zeros(window_len, dtype=float)

    if window_len == 0 or required_site_energy_kwh <= 1e-12:
        return {
            "plan_site_kw_per_step": plan_site_kw_per_step,
            "plan_pv_site_kw_per_step": plan_pv_site_kw_per_step,
            "plan_market_kw_per_step": plan_market_kw_per_step,
            "charged_site_kwh": 0.0,
            "charged_pv_site_kwh": 0.0,
            "remaining_site_kwh": float(max(required_site_energy_kwh, 0.0)),
            "final_soc": float(state_of_charge_at_arrival),
        }

    session_steps = np.arange(start, end_excl, dtype=int)
    prices_window = market_price_eur_per_mwh[session_steps].astype(float)
    order_by_price = np.argsort(prices_window) 

    battery_capacity_kwh = float(max(curve.battery_capacity_kwh, 1e-12))
    eff = float(charger_efficiency)

    def _dry_run_remaining_kwh(allowed_mask: np.ndarray) -> float:
        """Chronologischer Dry-Run: keine Side-Effects auf Reservierungen/Pläne."""
        remaining_kwh = float(required_site_energy_kwh)
        soc = float(state_of_charge_at_arrival)

        for local_i, abs_step in enumerate(range(start, end_excl)):
            if remaining_kwh <= 1e-12 or soc >= 1.0 - 1e-12:
                break

            base_kw_i = float(base_site_kw[local_i])
            if base_kw_i > 1e-12:
                base_kwh_i = base_kw_i * step_hours_safe
                remaining_kwh -= base_kwh_i
                soc = float(min(1.0, soc + (base_kwh_i * eff) / battery_capacity_kwh))
                if remaining_kwh <= 1e-12 or soc >= 1.0 - 1e-12:
                    break

            if not bool(allowed_mask[local_i]):
                continue

            alloc_site_kw, _alloc_pv_kw = available_power_for_session_step(
                step_index=int(abs_step),
                scenario=scenario,
                curve=curve,
                state_of_charge_fraction=float(soc),
                remaining_site_energy_kwh=float(remaining_kwh),
                pv_generation_kw_per_step=pv_generation_kw_per_step,
                base_load_kw_per_step=base_load_kw_per_step,
                reserved_total_ev_power_kw_per_step=reserved_total_ev_power_kw_per_step,
                reserved_pv_ev_power_kw_per_step=reserved_pv_ev_power_kw_per_step,
                already_allocated_on_this_charger_kw=float(base_kw_i), 
                supply_mode=supply_mode,
            )

            alloc_site_kw = float(alloc_site_kw)
            if alloc_site_kw <= 1e-12:
                continue

            alloc_site_kwh = alloc_site_kw * step_hours_safe
            remaining_kwh -= alloc_site_kwh

            soc = float(min(1.0, soc + (alloc_site_kwh * eff) / battery_capacity_kwh))

        return float(max(remaining_kwh, 0.0))

    # ---------------------------------------------------------
    # Slot-Auswahl: billigste Slots nacheinander hinzufügen,
    # jeweils mit CHRONOLOGISCHEM Dry-Run prüfen.
    # ---------------------------------------------------------
    allowed = np.zeros(window_len, dtype=bool)
    for idx in order_by_price:
        allowed[idx] = True
        if _dry_run_remaining_kwh(allowed) <= 1e-9:
            break

    # ---------------------------------------------------------
    # Finale Zuweisung: chronologisch, nur in allowed-Slots
    # ---------------------------------------------------------
    remaining_site_kwh = float(required_site_energy_kwh)
    state_of_charge = float(state_of_charge_at_arrival)

    charged_site_kwh = 0.0
    charged_pv_site_kwh = 0.0

    for local_i, abs_step in enumerate(range(start, end_excl)):
        if remaining_site_kwh <= 1e-12 or state_of_charge >= 1.0 - 1e-12:
            break

        base_kw_i = float(base_site_kw[local_i])
        if base_kw_i > 1e-12:
            base_kwh_i = base_kw_i * step_hours_safe
            remaining_site_kwh -= base_kwh_i
            state_of_charge = float(min(1.0, state_of_charge + (base_kwh_i * eff) / battery_capacity_kwh))
            if remaining_site_kwh <= 1e-12 or state_of_charge >= 1.0 - 1e-12:
                break

        if not bool(allowed[local_i]):
            continue

        alloc_site_kw, alloc_pv_kw = available_power_for_session_step(
            step_index=int(abs_step),
            scenario=scenario,
            curve=curve,
            state_of_charge_fraction=float(state_of_charge),
            remaining_site_energy_kwh=float(remaining_site_kwh),
            pv_generation_kw_per_step=pv_generation_kw_per_step,
            base_load_kw_per_step=base_load_kw_per_step,
            reserved_total_ev_power_kw_per_step=reserved_total_ev_power_kw_per_step,
            reserved_pv_ev_power_kw_per_step=reserved_pv_ev_power_kw_per_step,
            already_allocated_on_this_charger_kw=float(base_kw_i),
            supply_mode=supply_mode,
        )

        alloc_site_kw = float(alloc_site_kw)
        if alloc_site_kw <= 1e-12:
            continue

        alloc_pv_kw = float(np.clip(float(alloc_pv_kw), 0.0, alloc_site_kw))

        plan_site_kw_per_step[local_i] += alloc_site_kw
        plan_pv_site_kw_per_step[local_i] += alloc_pv_kw
        plan_market_kw_per_step[local_i] += alloc_site_kw  

        reserved_total_ev_power_kw_per_step[abs_step] += alloc_site_kw
        reserved_pv_ev_power_kw_per_step[abs_step] += alloc_pv_kw

        alloc_site_kwh = alloc_site_kw * step_hours_safe
        alloc_pv_kwh = alloc_pv_kw * step_hours_safe

        charged_site_kwh += alloc_site_kwh
        charged_pv_site_kwh += alloc_pv_kwh
        remaining_site_kwh -= alloc_site_kwh

        state_of_charge = float(min(1.0, state_of_charge + (alloc_site_kwh * eff) / battery_capacity_kwh))

    total_charged_site_kwh = float(np.sum(plan_site_kw_per_step) * step_hours_safe)
    total_charged_pv_kwh = float(np.sum(np.clip(plan_pv_site_kw_per_step, 0.0, plan_site_kw_per_step)) * step_hours_safe)

    return {
        "plan_site_kw_per_step": plan_site_kw_per_step,
        "plan_pv_site_kw_per_step": plan_pv_site_kw_per_step,
        "plan_market_kw_per_step": plan_market_kw_per_step,
        "charged_site_kwh": total_charged_site_kwh,
        "charged_pv_site_kwh": total_charged_pv_kwh,
        "remaining_site_kwh": float(max(required_site_energy_kwh - total_charged_site_kwh, 0.0)),
        "final_soc": float(state_of_charge),
    }


def charging_strategy_pv(
    *,
    scenario: dict,
    session_arrival_step: int,
    session_departure_step: int,
    required_site_energy_kwh: float,
    market_price_eur_per_mwh: np.ndarray,
    pv_generation_kw_per_step: np.ndarray,
    base_load_kw_per_step: np.ndarray,
    reserved_total_ev_power_kw_per_step: np.ndarray,
    reserved_pv_ev_power_kw_per_step: np.ndarray,
    curve: VehicleChargingCurve,
    state_of_charge_at_arrival: float,
) -> Dict[str, Any]:
    """
    Plant und allokiert Ladeleistung nach dem Prinzip „PV zuerst, danach Netz“.

    Ablauf (innerhalb der Session-Standzeit):

    1) PV-only (chronologisch)
       Es wird ausschließlich PV-Überschuss genutzt (``supply_mode="pv_only"`` via
       ``available_power_for_session_step(...)``). Die geplanten Leistungen werden in die
       Plan-Zeitreihen geschrieben und in die globalen Reservierungen (in-place) übernommen.
       SoC und Restenergiebedarf werden pro Schritt fortgeschrieben.

    2) Netz-Fallback (optional)
       Falls nach PV-only noch Energiebedarf übrig ist, wird Netzbezug nachgeladen.
       Der Fallback wird über ``scenario["charging_strategy_fallback"]`` gesteuert:

       - ``"market"`` (Standard): preisgeführt über
         ``charging_strategy_market(..., supply_mode="grid_only")`` -> Beiträge in
         ``plan_market_kw_per_step``.
       - ``"immediate"``: chronologisch über
         ``charging_strategy_immediate(..., supply_mode="grid_only")`` -> Beiträge in
         ``plan_immediate_kw_per_step``.

       Bereits geplante PV-Leistungen werden als Vorbelegung über
       ``preallocated_site_kw_per_step`` und ``preallocated_pv_site_kw_per_step`` übergeben,
       damit nur zusätzliche Netzleistung geplant wird.

    Einheiten
    ---------
    - Plan-/Reservierungs-Zeitreihen sind mittlere Leistungen je Schritt (kW).
    - Energien (kWh) ergeben sich je Schritt aus ``alloc_kwh = alloc_kw * step_hours``.
    - SoC-Update erfolgt über Batterieseiten-Energie:
      ``battery_energy_kwh = alloc_site_kwh * charger_efficiency``.

    Returns
    -------
    Dict[str, Any]
        - ``plan_site_kw_per_step``: gesamte geplante Leistung (PV + Netz) je Schritt (kW)
        - ``plan_pv_site_kw_per_step``: PV-Anteil je Schritt (kW)
        - ``plan_market_kw_per_step``: preisgeführter Netzanteil je Schritt (kW)
        - ``plan_immediate_kw_per_step``: chronologischer Netzanteil je Schritt (kW)
        - ``charged_site_kwh``: insgesamt geladene Energie (kWh)
        - ``charged_pv_site_kwh``: aus PV geladene Energie (kWh)
        - ``charged_market_kwh``: aus preisgeführtem Netzladen (kWh)
        - ``charged_immediate_kwh``: aus chronologischem Netzladen (kWh)
        - ``remaining_site_kwh``: verbleibender Bedarf auf Standortseite (kWh)
        - ``final_soc``: finaler SoC [0..1]
    """
    number_of_total_steps = int(len(reserved_total_ev_power_kw_per_step))

    site_configuration = scenario["site"]
    charger_efficiency = float(site_configuration.get("charger_efficiency", 1.0))

    time_resolution_min = int(scenario["time_resolution_min"])
    step_hours = float(time_resolution_min) / 60.0
    step_hours_safe = float(max(step_hours, 1e-12))

    start_step = int(max(0, session_arrival_step))
    end_step_exclusive = int(min(int(session_departure_step), number_of_total_steps))
    session_window_length = int(max(0, end_step_exclusive - start_step))

    plan_site_kw_per_step = np.zeros(session_window_length, dtype=float)
    plan_pv_site_kw_per_step = np.zeros(session_window_length, dtype=float)
    plan_market_kw_per_step = np.zeros(session_window_length, dtype=float)
    plan_immediate_kw_per_step = np.zeros(session_window_length, dtype=float)

    required_site_energy_kwh = float(max(required_site_energy_kwh, 0.0))
    remaining_site_kwh = float(required_site_energy_kwh)

    state_of_charge_at_arrival = float(np.clip(state_of_charge_at_arrival, 0.0, 1.0))
    state_of_charge = float(state_of_charge_at_arrival)

    if session_window_length == 0 or remaining_site_kwh <= 1e-12:
        return {
            "plan_site_kw_per_step": plan_site_kw_per_step,
            "plan_pv_site_kw_per_step": plan_pv_site_kw_per_step,
            "plan_market_kw_per_step": plan_market_kw_per_step,
            "plan_immediate_kw_per_step": plan_immediate_kw_per_step,
            "charged_site_kwh": 0.0,
            "charged_pv_site_kwh": 0.0,
            "charged_market_kwh": 0.0,
            "charged_immediate_kwh": 0.0,
            "remaining_site_kwh": float(max(remaining_site_kwh, 0.0)),
            "final_soc": float(state_of_charge),
        }

    battery_capacity_kwh = float(max(curve.battery_capacity_kwh, 1e-12))
    charger_efficiency_fraction = float(np.clip(charger_efficiency, 1e-12, 1.0))

    # ---------------------------------------------------------------------
    # (1) PV-only
    # ---------------------------------------------------------------------
    for local_index, absolute_step in enumerate(range(start_step, end_step_exclusive)):
        if remaining_site_kwh <= 1e-12 or state_of_charge >= 1.0 - 1e-12:
            break

        allocated_site_kw, allocated_pv_kw = available_power_for_session_step(
            step_index=int(absolute_step),
            scenario=scenario,
            curve=curve,
            state_of_charge_fraction=float(state_of_charge),
            remaining_site_energy_kwh=float(remaining_site_kwh),
            pv_generation_kw_per_step=pv_generation_kw_per_step,
            base_load_kw_per_step=base_load_kw_per_step,
            reserved_total_ev_power_kw_per_step=reserved_total_ev_power_kw_per_step,
            reserved_pv_ev_power_kw_per_step=reserved_pv_ev_power_kw_per_step,
            already_allocated_on_this_charger_kw=0.0,
            supply_mode="pv_only",
        )

        allocated_site_kw = float(max(allocated_site_kw, 0.0))
        if allocated_site_kw <= 1e-12:
            continue

        allocated_pv_kw = float(np.clip(float(allocated_pv_kw), 0.0, allocated_site_kw))

        plan_site_kw_per_step[local_index] += allocated_site_kw
        plan_pv_site_kw_per_step[local_index] += allocated_pv_kw

        reserved_total_ev_power_kw_per_step[absolute_step] += allocated_site_kw
        reserved_pv_ev_power_kw_per_step[absolute_step] += allocated_pv_kw

        allocated_site_kwh = allocated_site_kw * step_hours_safe
        remaining_site_kwh -= allocated_site_kwh

        battery_energy_kwh = allocated_site_kwh * charger_efficiency_fraction
        state_of_charge = float(min(1.0, state_of_charge + battery_energy_kwh / battery_capacity_kwh))

    # PV-only reicht aus
    if remaining_site_kwh <= 1e-12 or state_of_charge >= 1.0 - 1e-12:
        charged_site_kwh = float(np.sum(plan_site_kw_per_step) * step_hours_safe)
        charged_pv_site_kwh = float(
            np.sum(np.clip(plan_pv_site_kw_per_step, 0.0, plan_site_kw_per_step)) * step_hours_safe
        )
        return {
            "plan_site_kw_per_step": plan_site_kw_per_step,
            "plan_pv_site_kw_per_step": plan_pv_site_kw_per_step,
            "plan_market_kw_per_step": plan_market_kw_per_step,
            "plan_immediate_kw_per_step": plan_immediate_kw_per_step,
            "charged_site_kwh": charged_site_kwh,
            "charged_pv_site_kwh": charged_pv_site_kwh,
            "charged_market_kwh": 0.0,
            "charged_immediate_kwh": 0.0,
            "remaining_site_kwh": float(max(required_site_energy_kwh - charged_site_kwh, 0.0)),
            "final_soc": float(state_of_charge),
        }

    # ---------------------------------------------------------------------
    # (2) Netz-Fallback
    # ---------------------------------------------------------------------
    fallback_strategy = str(scenario.get("charging_strategy_fallback") or "market").strip().lower()
    if fallback_strategy not in ("market", "immediate"):
        raise ValueError("scenario['charging_strategy_fallback'] muss 'market' oder 'immediate' sein.")

    if fallback_strategy == "market":
        charging_strategy_market_result = charging_strategy_market(
            session_arrival_step=int(start_step),
            session_departure_step=int(end_step_exclusive),
            required_site_energy_kwh=float(required_site_energy_kwh),
            market_price_eur_per_mwh=market_price_eur_per_mwh,
            pv_generation_kw_per_step=pv_generation_kw_per_step,
            base_load_kw_per_step=base_load_kw_per_step,
            reserved_total_ev_power_kw_per_step=reserved_total_ev_power_kw_per_step,
            reserved_pv_ev_power_kw_per_step=reserved_pv_ev_power_kw_per_step,
            curve=curve,
            state_of_charge_at_arrival=float(state_of_charge_at_arrival),
            scenario=scenario,
            preallocated_site_kw_per_step=plan_site_kw_per_step,
            preallocated_pv_site_kw_per_step=plan_pv_site_kw_per_step,
            supply_mode="grid_only",
        )

        charged_market_kwh = float(
            np.sum(np.asarray(charging_strategy_market_result["plan_market_kw_per_step"], float)) * step_hours_safe
        )

        charging_strategy_market_result.setdefault("plan_immediate_kw_per_step", np.zeros_like(plan_site_kw_per_step))
        charging_strategy_market_result["charged_market_kwh"] = charged_market_kwh
        charging_strategy_market_result.setdefault("charged_immediate_kwh", 0.0)
        return charging_strategy_market_result

    charging_strategy_immediate_result = charging_strategy_immediate(
        scenario=scenario,
        session_arrival_step=int(start_step),
        session_departure_step=int(end_step_exclusive),
        required_site_energy_kwh=float(required_site_energy_kwh),
        pv_generation_kw_per_step=pv_generation_kw_per_step,
        base_load_kw_per_step=base_load_kw_per_step,
        reserved_total_ev_power_kw_per_step=reserved_total_ev_power_kw_per_step,
        reserved_pv_ev_power_kw_per_step=reserved_pv_ev_power_kw_per_step,
        curve=curve,
        state_of_charge_at_arrival=float(state_of_charge_at_arrival),
        preallocated_site_kw_per_step=plan_site_kw_per_step,
        preallocated_pv_site_kw_per_step=plan_pv_site_kw_per_step,
        supply_mode="grid_only",
    )

    # Netzanteil aus dem (gesamt-)Plan bestimmen
    planned_site_kw_per_step = np.asarray(charging_strategy_immediate_result["plan_site_kw_per_step"], float)
    planned_pv_kw_per_step = np.asarray(charging_strategy_immediate_result["plan_pv_site_kw_per_step"], float)
    planned_grid_kw_per_step_total = np.maximum(planned_site_kw_per_step - planned_pv_kw_per_step, 0.0)

    # Falls Vorbelegung doch bereits Netzanteile enthielte: nur die Zusatzleistung als "immediate"
    preallocated_grid_kw_per_step = np.maximum(plan_site_kw_per_step - plan_pv_site_kw_per_step, 0.0)
    planned_grid_kw_per_step_increment = np.maximum(planned_grid_kw_per_step_total - preallocated_grid_kw_per_step, 0.0)

    charging_strategy_immediate_result["plan_market_kw_per_step"] = np.zeros_like(planned_grid_kw_per_step_increment)
    charging_strategy_immediate_result["plan_immediate_kw_per_step"] = planned_grid_kw_per_step_increment

    charging_strategy_immediate_result["charged_market_kwh"] = 0.0
    charging_strategy_immediate_result["charged_immediate_kwh"] = float(
        np.sum(planned_grid_kw_per_step_increment) * step_hours_safe
    )
    return charging_strategy_immediate_result


def plan_charging_for_day(
    *,
    sessions: list[SampledSession],
    scenario: dict,
    vehicle_curves_by_name: dict[str, VehicleChargingCurve],
    pv_generation_kw_per_step: np.ndarray,
    base_load_kw_per_step: np.ndarray,
    reserved_total_ev_power_kw_per_step: np.ndarray,
    reserved_pv_ev_power_kw_per_step: np.ndarray,
    market_price_eur_per_mwh: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """
    Plant für einen Kalendertag die Ladeleistungs-Zuweisung (kW pro Step) für alle Ladesessions.

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
        pv_generation_kw_per_step:
            PV-Erzeugung pro Simulationsschritt [kW].
        base_load_kw_per_step:
            Grundlast pro Simulationsschritt [kW].
        reserved_total_ev_power_kw_per_step:
            Bereits reservierte EV-Ladeleistung (gesamt) pro Schritt [kW].
            Wird in-place um die geplanten Zuweisungen dieser Funktion erhöht.
        reserved_pv_ev_power_kw_per_step:
            Bereits reservierter PV-Anteil der EV-Ladeleistung pro Schritt [kW].
            Wird in-place um die geplanten PV-Zuweisungen dieser Funktion erhöht.
        market_price_eur_per_mwh:
            Marktpreis-Zeitreihe [€/MWh]. Erforderlich für ``"market"`` sowie als
            Fallback-Ranking für ``"pv"`` / ``"generation"``.

        Rückgabe
        -------
        list[dict[str, Any]]
            Liste von Ergebnis-Dictionaries, je Session ein Eintrag. Enthält u. a.
            Plan-Zeitreihen (z. B. ``plan_site_kw_per_step``) sowie Summenwerte
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
                pv_generation_kw_per_step=pv_generation_kw_per_step,
                base_load_kw_per_step=base_load_kw_per_step,
                reserved_total_ev_power_kw_per_step=reserved_total_ev_power_kw_per_step,
                reserved_pv_ev_power_kw_per_step=reserved_pv_ev_power_kw_per_step,
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
                pv_generation_kw_per_step=pv_generation_kw_per_step,
                base_load_kw_per_step=base_load_kw_per_step,
                reserved_total_ev_power_kw_per_step=reserved_total_ev_power_kw_per_step,
                reserved_pv_ev_power_kw_per_step=reserved_pv_ev_power_kw_per_step,
                curve=curve,
                state_of_charge_at_arrival=float(session.state_of_charge_at_arrival),
            )

        elif strategy in ("pv", "generation"):
            if market_price_eur_per_mwh is None:
                raise ValueError("charging_strategy='pv' benötigt market_price_eur_per_mwh (Fallback-Ranking).")

            session_result = charging_strategy_pv(
                scenario=scenario,
                session_arrival_step=int(session.arrival_step),
                session_departure_step=int(session.departure_step),
                required_site_energy_kwh=float(required_site_energy_kwh),
                market_price_eur_per_mwh=market_price_eur_per_mwh,
                pv_generation_kw_per_step=pv_generation_kw_per_step,
                base_load_kw_per_step=base_load_kw_per_step,
                reserved_total_ev_power_kw_per_step=reserved_total_ev_power_kw_per_step,
                reserved_pv_ev_power_kw_per_step=reserved_pv_ev_power_kw_per_step,
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


def simulate_site_fcfs_with_planning(
    *,
    scenario: dict,
    timestamps: pd.DatetimeIndex,
    sessions: list[SampledSession],
    vehicle_curves_by_name: dict[str, VehicleChargingCurve],
    pv_generation_kw_per_step: np.ndarray,
    base_load_kw_per_step: np.ndarray,
    reserved_total_ev_power_kw_per_step: Optional[np.ndarray] = None,
    reserved_pv_ev_power_kw_per_step: Optional[np.ndarray] = None,
    market_price_eur_per_mwh: Optional[np.ndarray] = None,
    record_debug: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    """
    Kern-Simulation: Admission (FCFS) + Tagesplanung (Strategie) + EV-Lastgang.

        Ablauf
        ------
        1) FCFS-Admission: Charger-Vergabe, Sessions ohne freien Charger -> "drive_off"
        2) Tagesplanung: pro Arrival-Tag ``plan_charging_for_day(...)`` (nutzt Strategien + Reservierungen)
        3) Lastgang: ``ev_load_kw`` entspricht der reservierten EV-Leistung (kW/Step)
        4) Optional: Debug-Bilanz als DataFrame

        Rückgabe
        -------
        (ev_load_kw, sessions_out, reserved_total_kw, reserved_pv_kw, debug_df)
    """
    n_total = int(len(timestamps))

    # Reservierungs-Arrays (global, in-place durch Strategien gefüllt)
    if reserved_total_ev_power_kw_per_step is None:
        reserved_total_ev_power_kw_per_step = np.zeros(n_total, dtype=float)
    if reserved_pv_ev_power_kw_per_step is None:
        reserved_pv_ev_power_kw_per_step = np.zeros(n_total, dtype=float)

    reserved_total = reserved_total_ev_power_kw_per_step
    reserved_pv = reserved_pv_ev_power_kw_per_step

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
                    "plan_site_kw_per_step": np.zeros(0, dtype=float),
                    "plan_pv_site_kw_per_step": np.zeros(0, dtype=float),
                    "plan_market_kw_per_step": np.zeros(0, dtype=float),
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
            pv_generation_kw_per_step=np.asarray(pv_generation_kw_per_step, dtype=float),
            base_load_kw_per_step=np.asarray(base_load_kw_per_step, dtype=float),
            reserved_total_ev_power_kw_per_step=reserved_total,
            reserved_pv_ev_power_kw_per_step=reserved_pv,
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

    # EV-Lastgang ist die reservierte Leistung
    ev_load_kw = np.asarray(reserved_total, dtype=float)

    debug_df = None
    if record_debug:
        debug_df = _compute_debug_balance(
            timestamps=timestamps,
            scenario=scenario,
            pv_generation_kw_per_step=pv_generation_kw_per_step,
            base_load_kw_per_step=base_load_kw_per_step,
            reserved_total_ev_power_kw_per_step=reserved_total,
            reserved_pv_ev_power_kw_per_step=reserved_pv,
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
    pv_generation_kw_per_step: np.ndarray,
    base_load_kw_per_step: np.ndarray,
    reserved_total_ev_power_kw_per_step: np.ndarray,
    reserved_pv_ev_power_kw_per_step: np.ndarray,
) -> pd.DataFrame:
    """
    Erstellt eine Debug-Leistungsbilanz pro Simulationsschritt als DataFrame.

    Anpassungen ggü. der Ausgangsversion:
    - PV→EV wird zusätzlich durch die tatsächliche EV-Leistung `ev` begrenzt.
    - Zusätzliche Debug-Spalten: PV-Überschuss (Export/Abregelung), unbediente Grundlast/EV,
      Gesamt-Netzbezug, Plausibilitäts-/Bilanzchecks.
    - Eingänge werden defensiv auf >= 0 geklippt (optional/Debug-freundlich).

    Hinweis: rein für Debugging/Analyse, ohne Einfluss auf die Ladeplanung.
    """
    grid_limit_kw = float(scenario["site"]["grid_limit_p_avb_kw"])

    pv = np.asarray(pv_generation_kw_per_step, dtype=float)
    base = np.asarray(base_load_kw_per_step, dtype=float)
    ev = np.asarray(reserved_total_ev_power_kw_per_step, dtype=float)
    pv_ev_tracked = np.asarray(reserved_pv_ev_power_kw_per_step, dtype=float)

    n = len(pd.to_datetime(timestamps))
    if not (len(pv) == len(base) == len(ev) == len(pv_ev_tracked) == n):
        raise ValueError(
            "Längen passen nicht zusammen: timestamps/pv/base/ev/pv_ev_tracked müssen gleich lang sein."
        )

    # Debug-robust: negative Werte sind in dieser Bilanz i. d. R. nicht sinnvoll
    pv = np.clip(pv, 0.0, None)
    base = np.clip(base, 0.0, None)
    ev = np.clip(ev, 0.0, None)
    pv_ev_tracked = np.clip(pv_ev_tracked, 0.0, None)

    # 1) PV -> Grundlast
    pv_to_base = np.minimum(pv, base)
    base_remaining = base - pv_to_base  # >= 0

    # 2) PV -> EV (begrenzt durch getrackten PV-Anteil, verfügbaren PV-Überschuss und tatsächliche EV-Leistung)
    pv_after_base = pv - pv_to_base  # >= 0
    pv_to_ev = np.minimum.reduce([pv_ev_tracked, pv_after_base, ev])
    ev_remaining = ev - pv_to_ev  # >= 0

    # 3) Netz -> Grundlast (bis Netzlimit)
    grid_to_base = np.minimum(base_remaining, grid_limit_kw)
    grid_headroom_after_base = grid_limit_kw - grid_to_base  # >= 0

    # 4) Netz -> EV (nur Restspielraum nach Grundlast)
    grid_to_ev = np.minimum(ev_remaining, grid_headroom_after_base)

    # Zusatz: Unbediente Lasten (falls Netzlimit nicht reicht)
    unmet_base = base_remaining - grid_to_base  # >= 0
    unmet_ev = ev_remaining - grid_to_ev        # >= 0

    # Zusatz: PV-Überschuss (Export/Abregelung), falls PV nach Base+EV noch übrig bleibt
    pv_surplus = pv_after_base - pv_to_ev  # >= 0

    # Zusatz: Summen/Checks
    grid_import = grid_to_base + grid_to_ev
    pv_used = pv_to_base + pv_to_ev
    served_total = pv_used + grid_import
    demand_total = base + ev

    # Wenn alles bedient und kein PV-Überschuss: served_total == demand_total
    # Allgemein gilt: served_total + unmet_total == demand_total
    unmet_total = unmet_base + unmet_ev
    balance_served_plus_unmet_minus_demand = (served_total + unmet_total) - demand_total  # ~0

    dataframe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),

            "pv_generation_kw_per_step": pv,
            "base_load_kw_per_step": base,
            "ev_load_kw_per_step": ev,
            "pv_ev_tracked_kw_per_step": pv_ev_tracked,

            "pv_to_base_kw_per_step": pv_to_base,
            "pv_to_ev_kw_per_step": pv_to_ev,
            "pv_surplus_kw_per_step": pv_surplus, 

            "grid_to_base_kw_per_step": grid_to_base,
            "grid_to_ev_kw_per_step": grid_to_ev,
            "grid_import_kw_per_step": grid_import,

            "unmet_base_kw_per_step": unmet_base,
            "unmet_ev_kw_per_step": unmet_ev,
            "unmet_total_kw_per_step": unmet_total,

            "pv_used_kw_per_step": pv_used,
            "served_total_kw_per_step": served_total,
            "demand_total_kw_per_step": demand_total,

            "grid_limit_kw_per_step": grid_limit_kw,

            "balance_served_plus_unmet_minus_demand_kw_per_step": balance_served_plus_unmet_minus_demand,
        }
    )
    return dataframe


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


def build_plugged_sessions_preview_table(
    sessions_out: list[dict[str, Any]], *, n: int = 10
) -> pd.DataFrame:
    """
    Erzeugt eine Vorschau-Tabelle für die ersten n erfolgreichen Ladesessions.

    Angezeigte Spalten:
    - Session-ID
    - Ladepunkt
    - Ankunft
    - Abfahrt
    - Parkdauer [min]
    - SoC Ankunft
    - SoC Ende
    - Geladene Energie [kWh]
    - Durchschnittliche Ladeleistung [kW]
    - Maximale Ladeleistung [kW]
    - Fahrzeug
    """
    rows: list[dict[str, Any]] = []

    def _format_datetime(value) -> str | None:
        if value is None:
            return None
        try:
            dt = pd.to_datetime(value)
            return dt.strftime("%d.%m.%y, %H:%M")
        except Exception:
            return None

    def _soc_pct(value) -> float:
        try:
            value = float(value)
            return value * 100.0 if np.isfinite(value) else np.nan
        except Exception:
            return np.nan

    def _parking_duration_min(value) -> float | int:
        try:
            value = float(value)
            return int(round(value)) if np.isfinite(value) else np.nan
        except Exception:
            return np.nan

    for session in sessions_out:
        if str(session.get("status")) != "plugged":
            continue

        a = session.get("arrival_time")
        d = session.get("departure_time")

        duration_h = np.nan
        try:
            if a is not None and d is not None:
                duration_h = float((pd.to_datetime(d) - pd.to_datetime(a)).total_seconds()) / 3600.0
        except Exception:
            duration_h = np.nan
        duration_h = float(duration_h) if np.isfinite(duration_h) and duration_h > 0.0 else np.nan

        charged_site_kwh = float(session.get("charged_site_kwh", 0.0) or 0.0)
        avg_site_kw = charged_site_kwh / duration_h if np.isfinite(duration_h) else np.nan

        max_site_kw = np.nan
        try:
            plan_site = np.asarray(session.get("plan_site_kw_per_step", np.zeros(0)), float).reshape(-1)
            if plan_site.size > 0:
                plan_site = np.maximum(plan_site, 0.0)
                max_site_kw = float(np.nanmax(plan_site))
        except Exception:
            max_site_kw = np.nan

        rows.append(
            {
                "Session-ID": session.get("session_id"),
                "Ladepunkt": session.get("charger_id"),
                "Ankunft": _format_datetime(a),
                "Abfahrt": _format_datetime(d),
                "Parkdauer [min]": _parking_duration_min(session.get("parking_duration_min")),
                "SoC Ankunft [%]": _soc_pct(session.get("state_of_charge_at_arrival")),
                "SoC Ende [%]": _soc_pct(session.get("final_soc")),
                "Geladene Energie [kWh]": charged_site_kwh,
                "Durchschnittliche Ladeleistung [kW]": avg_site_kw,
                "Maximale Ladeleistung [kW]": max_site_kw,
                "Fahrzeug": session.get("vehicle_name"),
            }
        )

    dataframe = pd.DataFrame(rows)
    if len(dataframe) == 0:
        return dataframe

    dataframe = (
        dataframe.sort_values(["Ankunft", "Session-ID"], na_position="last")
        .head(int(n))
        .reset_index(drop=True)
    )
    if "Parkdauer [min]" in dataframe.columns:
        dataframe["Parkdauer [min]"] = pd.to_numeric(
            dataframe["Parkdauer [min]"], errors="coerce"
        ).astype("Int64")

    numeric_cols = dataframe.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "Parkdauer [min]"]
    dataframe[numeric_cols] = dataframe[numeric_cols].round(2)

    return dataframe


def build_timeseries_dataframe(
    *,
    timestamps: pd.DatetimeIndex,
    scenario: dict,
    base_load_kw_per_step: np.ndarray,
    pv_generation_kw_per_step: np.ndarray,
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
    base_load_kw_per_step:
        Grundlast als Leistung pro Schritt [kW/Schritt].
    pv_generation_kw_per_step:
        PV-Erzeugung als Leistung pro Schritt [kW/Schritt].
    ev_load_kw:
        EV-Ladeleistung als Leistung pro Schritt [kW/Schritt].
    market_price_eur_per_mwh:
        Optional: Marktpreis pro Schritt.
    debug_df:
        Optional: Debug-Bilanzdaten; Spalten werden übernommen, falls kompatibel.

    Rückgabe
    --------
    pd.DataFrame
        Einheitliche Tabelle (timestamp, base_load_kw, pv_generation_kw, ev_load_kw, ...).
    """
    dataframe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "base_load_kw": np.asarray(base_load_kw_per_step, dtype=float),
            "pv_generation_kw": np.asarray(pv_generation_kw_per_step, dtype=float),
            "ev_load_kw": np.asarray(ev_load_kw, dtype=float),
        }
    )

    if market_price_eur_per_mwh is not None:
        dataframe["market_price_eur_per_mwh"] = np.asarray(market_price_eur_per_mwh, dtype=float)

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

    Erwartet: timeseries_dataframe enthält Leistungen in kW (ev_load_kw etc.).
    Optional können Debug-Spalten bereits als kW vorliegen.

    Rückgabe
    --------
    pd.DataFrame
        Spalten: timestamp, ev_from_pv_kw, ev_from_grid_kw
    """
    dataframe = timeseries_dataframe.copy()
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])

    ev_kw = dataframe["ev_load_kw"].astype(float).fillna(0.0).to_numpy()

    # Falls Debug-Bilanz bereits in kW existiert, direkt nutzen.
    if "pv_to_ev_kw" in dataframe.columns:
        pv_kw = dataframe["pv_to_ev_kw"].astype(float).fillna(0.0).to_numpy()
        grid_series = dataframe.get("grid_to_ev_kw")
        if isinstance(grid_series, pd.Series):
            grid_kw = grid_series.astype(float).fillna(0.0).to_numpy()
        else:
            grid_kw = np.maximum(ev_kw - pv_kw, 0.0)

    elif "pv_ev_tracked_kw_per_step" in dataframe.columns:
        pv_kw = dataframe["pv_ev_tracked_kw_per_step"].astype(float).fillna(0.0).to_numpy()
        pv_kw = np.clip(pv_kw, 0.0, ev_kw)
        grid_kw = np.maximum(ev_kw - pv_kw, 0.0)

    elif "pv_to_ev_kwh_per_step" in dataframe.columns or "pv_ev_tracked_kwh_per_step" in dataframe.columns:
        step_h = (
            float((dataframe["timestamp"].iloc[1] - dataframe["timestamp"].iloc[0]).total_seconds()) / 3600.0
            if len(dataframe) >= 2
            else 0.25
        )
        step_h = max(step_h, 1e-12)

        if "pv_to_ev_kwh_per_step" in dataframe.columns:
            pv_kw = dataframe["pv_to_ev_kwh_per_step"].astype(float).fillna(0.0).to_numpy() / step_h
            grid_series = dataframe.get("grid_to_ev_kwh_per_step")
            if isinstance(grid_series, pd.Series):
                grid_kw = grid_series.astype(float).fillna(0.0).to_numpy() / step_h
            else:
                grid_kw = np.maximum(ev_kw - pv_kw, 0.0)
        else:
            pv_kw = dataframe["pv_ev_tracked_kwh_per_step"].astype(float).fillna(0.0).to_numpy() / step_h
            pv_kw = np.clip(pv_kw, 0.0, ev_kw)
            grid_kw = np.maximum(ev_kw - pv_kw, 0.0)

    else:
        pv_kw = np.zeros_like(ev_kw)
        grid_kw = ev_kw

    return pd.DataFrame(
        {
            "timestamp": dataframe["timestamp"],
            "ev_from_pv_kw": pv_kw,
            "ev_from_grid_kw": grid_kw,
        }
    )


def build_charger_traces_dataframe(
    *,
    sessions_out: list[dict[str, Any]],
    scenario: dict,
    vehicle_curves_by_name: dict[str, VehicleChargingCurve],
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
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

        plan_site = (
            np.asarray(session.get("plan_site_kw_per_step", np.zeros(0)), dtype=float)
            .reshape(-1)[:window_length_steps]
        )
        plan_pv = (
            np.asarray(session.get("plan_pv_site_kw_per_step", np.zeros(0)), dtype=float)
            .reshape(-1)[:window_length_steps]
        )

        if len(plan_site) < window_length_steps:
            tmp = np.zeros(window_length_steps, dtype=float)
            tmp[:len(plan_site)] = plan_site
            plan_site = tmp
        if len(plan_pv) < window_length_steps:
            tmp = np.zeros(window_length_steps, dtype=float)
            tmp[:len(plan_pv)] = plan_pv
            plan_pv = tmp

        soc0 = session.get("state_of_charge_at_arrival")
        soc0 = float(np.clip(float(soc0), 0.0, 1.0)) if soc0 is not None else np.nan

        curve = vehicle_curves_by_name.get(str(session.get("vehicle_name", "")))
        cap = float(max(getattr(curve, "battery_capacity_kwh", 1e-12), 1e-12))

        batt_added_before = np.concatenate(([0.0], np.cumsum(plan_site[:-1]) * step_h * eff))

        soc_trace = (
            np.clip(soc0 + batt_added_before / cap, 0.0, 1.0)
            if np.isfinite(soc0)
            else np.full(window_length_steps, np.nan)
        )

        for i in range(window_length_steps):
            abs_step = a + i
            site_kw = float(plan_site[i])
            pv_kw = float(np.clip(plan_pv[i], 0.0, site_kw))

            rows.append(
                {
                    "timestamp": pd.Timestamp(timestamps[abs_step]),
                    "charger_id": None if session.get("charger_id") is None else int(session["charger_id"]),
                    "session_id": session.get("session_id"),
                    "vehicle_name": session.get("vehicle_name"),
                    "power_kw": site_kw,
                    "pv_power_kw": pv_kw,
                    "soc": float(soc_trace[i]) if np.isfinite(soc_trace[i]) else np.nan,
                    "is_plugged": True,
                    "is_charging": bool(site_kw > 1e-12),
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


def build_ev_power_by_mode_timeseries_dataframe(
    *,
    timeseries_dataframe: pd.DataFrame,
    sessions_out: list[dict[str, Any]],
    scenario: dict,
) -> pd.DataFrame:
    """
    Aggregiert EV-Leistung nach Lademodus (PV, Market, Immediate).

    Die Zuordnung erfolgt primär über die in den Session-Ergebnissen enthaltenen Plan-Arrays:
    - ``plan_pv_site_kw_per_step`` -> PV-Anteil (Generation)
    - ``plan_market_kw_per_step`` -> preisgeführter Netzanteil (Market)
    - ``plan_immediate_kw_per_step`` -> chronologischer Netzanteil (Immediate)

    Falls ``plan_immediate_kw_per_step`` nicht vorhanden ist (Legacy), wird der verbleibende
    Netzanteil aus ``plan_site_kw_per_step - plan_pv_site_kw_per_step - plan_market_kw_per_step``
    als Immediate interpretiert.

    Parameters
    ----------
    timeseries_dataframe:
        Zeitreihen-DataFrame der Simulation (muss ``timestamp`` enthalten).
    sessions_out:
        Session-Ergebnisse inkl. Plan-Arrays und ``arrival_step``/``departure_step``.
    scenario:
        Szenario-Dictionary (wird hier nicht zur Modus-Entscheidung verwendet, nur als Kontext).

    Returns
    -------
    pd.DataFrame
        Spalten: ``timestamp``, ``ev_generation_kw``, ``ev_market_kw``, ``ev_immediate_kw``.
    """
    dataframe = timeseries_dataframe.copy()
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
    number_of_steps = int(len(dataframe))

    ev_generation_kw = np.zeros(number_of_steps, dtype=float)
    ev_market_kw = np.zeros(number_of_steps, dtype=float)
    ev_immediate_kw = np.zeros(number_of_steps, dtype=float)

    for session in sessions_out:
        if str(session.get("status")) != "plugged":
            continue

        arrival_step = int(np.clip(int(session.get("arrival_step", 0)), 0, number_of_steps))
        departure_step = int(np.clip(int(session.get("departure_step", arrival_step)), 0, number_of_steps))
        if departure_step <= arrival_step:
            continue

        window_length_steps = int(departure_step - arrival_step)

        planned_site_kw_per_step = (
            np.asarray(session.get("plan_site_kw_per_step", np.zeros(0)), dtype=float).reshape(-1)[:window_length_steps]
        )
        planned_pv_kw_per_step = (
            np.asarray(session.get("plan_pv_site_kw_per_step", np.zeros(0)), dtype=float).reshape(-1)[:window_length_steps]
        )
        planned_market_kw_per_step = (
            np.asarray(session.get("plan_market_kw_per_step", np.zeros(0)), dtype=float).reshape(-1)[:window_length_steps]
        )
        planned_immediate_kw_per_step = (
            np.asarray(session.get("plan_immediate_kw_per_step", np.zeros(0)), dtype=float).reshape(-1)[:window_length_steps]
        )

        # Auf Fensterlänge auffüllen
        if len(planned_site_kw_per_step) < window_length_steps:
            padded = np.zeros(window_length_steps, dtype=float)
            padded[:len(planned_site_kw_per_step)] = planned_site_kw_per_step
            planned_site_kw_per_step = padded

        if len(planned_pv_kw_per_step) < window_length_steps:
            padded = np.zeros(window_length_steps, dtype=float)
            padded[:len(planned_pv_kw_per_step)] = planned_pv_kw_per_step
            planned_pv_kw_per_step = padded

        if len(planned_market_kw_per_step) < window_length_steps:
            padded = np.zeros(window_length_steps, dtype=float)
            padded[:len(planned_market_kw_per_step)] = planned_market_kw_per_step
            planned_market_kw_per_step = padded

        if len(planned_immediate_kw_per_step) < window_length_steps:
            padded = np.zeros(window_length_steps, dtype=float)
            padded[:len(planned_immediate_kw_per_step)] = planned_immediate_kw_per_step
            planned_immediate_kw_per_step = padded

        # Physikalisch konsistent clampen
        planned_site_kw_per_step = np.maximum(planned_site_kw_per_step, 0.0)
        planned_pv_kw_per_step = np.clip(planned_pv_kw_per_step, 0.0, planned_site_kw_per_step)

        # Market/Immediate zunächst aus den expliziten Arrays
        planned_market_kw_per_step = np.clip(planned_market_kw_per_step, 0.0, planned_site_kw_per_step - planned_pv_kw_per_step)

        # Legacy-Fallback: wenn immediate nicht explizit vorhanden ist, Rest als immediate interpretieren
        if float(np.sum(planned_immediate_kw_per_step)) <= 1e-12:
            planned_immediate_kw_per_step = np.maximum(
                planned_site_kw_per_step - planned_pv_kw_per_step - planned_market_kw_per_step,
                0.0,
            )
        else:
            planned_immediate_kw_per_step = np.clip(
                planned_immediate_kw_per_step,
                0.0,
                planned_site_kw_per_step - planned_pv_kw_per_step - planned_market_kw_per_step,
            )

        ev_generation_kw[arrival_step:departure_step] += planned_pv_kw_per_step
        ev_market_kw[arrival_step:departure_step] += planned_market_kw_per_step
        ev_immediate_kw[arrival_step:departure_step] += planned_immediate_kw_per_step

    return pd.DataFrame(
        {
            "timestamp": dataframe["timestamp"],
            "ev_generation_kw": ev_generation_kw,
            "ev_market_kw": ev_market_kw,
            "ev_immediate_kw": ev_immediate_kw,
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


def build_not_reached_sessions_table(summary: dict[str, Any]) -> pd.DataFrame:
    """
    Erstellt eine formatierte Tabelle für Sessions, die den Ziel-SoC nicht erreicht haben.

    Spalten:
    - Session-ID
    - Ladepunkt
    - Ankunft
    - Parkdauer [min]
    - SoC Ankunft [%]
    - SoC Ende [%]
    - Fehlende Energie [kWh]
    """
    rows = summary.get("not_reached_rows", []) or []
    dataframe = pd.DataFrame(rows)

    if len(dataframe) == 0:
        return pd.DataFrame(
            columns=[
                "Session-ID",
                "Ladepunkt",
                "Ankunft",
                "Parkdauer [min]",
                "SoC Ankunft [%]",
                "SoC Ende [%]",
                "Fehlende Energie [kWh]",
            ]
        )

    rename_map = {}
    if "remaining_energy" in dataframe.columns and "remaining_energy_kwh" not in dataframe.columns:
        rename_map["remaining_energy"] = "remaining_energy_kwh"
    if "state_of_charge_at_arrival" in dataframe.columns and "soc_arrival" not in dataframe.columns:
        rename_map["state_of_charge_at_arrival"] = "soc_arrival"
    if "final_soc" in dataframe.columns and "soc_end" not in dataframe.columns:
        rename_map["final_soc"] = "soc_end"
    dataframe = dataframe.rename(columns=rename_map)

    if "vehicle_name" in dataframe.columns:
        dataframe = dataframe.drop(columns=["vehicle_name"])

    if "arrival_time" in dataframe.columns:
        arrival_ts = pd.to_datetime(dataframe["arrival_time"], errors="coerce")
        dataframe["arrival_time"] = arrival_ts.dt.strftime("%d.%m.%y, %H:%M:%S")

    for col in ["soc_arrival", "soc_end"]:
        if col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce") * 100.0

    if "parking_duration_min" in dataframe.columns:
        dataframe["parking_duration_min"] = pd.to_numeric(dataframe["parking_duration_min"], errors="coerce")

    if "remaining_energy_kwh" in dataframe.columns:
        dataframe["remaining_energy_kwh"] = pd.to_numeric(dataframe["remaining_energy_kwh"], errors="coerce")
        dataframe = dataframe.sort_values("remaining_energy_kwh", ascending=False, na_position="last")
    else:
        dataframe = dataframe.reset_index(drop=True)

    wanted_cols = [
        "session_id",
        "charger_id",
        "arrival_time",
        "parking_duration_min",
        "soc_arrival",
        "soc_end",
        "remaining_energy_kwh",
    ]
    existing_wanted_cols = [c for c in wanted_cols if c in dataframe.columns]
    other_cols = [c for c in dataframe.columns if c not in existing_wanted_cols]
    dataframe = dataframe[existing_wanted_cols + other_cols].reset_index(drop=True)

    dataframe = dataframe.rename(
        columns={
            "session_id": "Session-ID",
            "charger_id": "Ladepunkt",
            "arrival_time": "Ankunft",
            "parking_duration_min": "Parkdauer [min]",
            "soc_arrival": "SoC Ankunft [%]",
            "soc_end": "SoC Ende [%]",
            "remaining_energy_kwh": "Fehlende Energie [kWh]",
        }
    )

    numeric_cols = dataframe.select_dtypes(include=["number"]).columns
    dataframe[numeric_cols] = dataframe[numeric_cols].round(2)

    return dataframe


def build_master_curve_and_actual_points_for_vehicle(
    *,
    charger_traces_dataframe: pd.DataFrame,
    scenario: dict,
    curve: VehicleChargingCurve,
    power_tolerance_kw: float = 1e-6,
) -> dict[str, object]:
    """
    Bereitet Daten für den Vergleich von Master-Ladekurve und Ist-Ladepunkten auf.

    Hinweis zur Leistungsdefinition
    -------------------------------
    - Masterkurve ``curve.power_kw`` ist auf **Batterieseite**.
    - Trace-Leistung ist auf **Standort-/Ladepunktseite** (typisch Spalte ``power_kw``).
      Für den Vergleich wird auf Batterieseite umgerechnet:
      ``power_batt_kw = power_site_kw * charger_efficiency``.

    Erwartete Spalten im charger_traces_dataframe
    ---------------------------------------------
    - vehicle_name, timestamp, soc
    - Leistungsspalte: bevorzugt ``power_kw``; alternativ wird auch ``site_power_kw`` akzeptiert.
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

    required_base_cols = {"vehicle_name", "timestamp", "soc"}
    if not required_base_cols.issubset(set(dataframe.columns)):
        return empty_result

    power_col = None
    if "power_kw" in dataframe.columns:
        power_col = "power_kw"
    elif "site_power_kw" in dataframe.columns:
        power_col = "site_power_kw"
    else:
        return empty_result

    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], errors="coerce")
    dataframe = dataframe[dataframe["vehicle_name"].astype(str) == str(curve.vehicle_name)]

    actual_soc = pd.to_numeric(dataframe["soc"], errors="coerce").to_numpy(dtype=float)
    actual_power_site_kw = pd.to_numeric(dataframe[power_col], errors="coerce").to_numpy(dtype=float)

    eff = float(np.clip(float(scenario["site"].get("charger_efficiency", 1.0)), 1e-12, 1.0))
    actual_power_batt_kw = np.maximum(actual_power_site_kw * eff, 0.0)

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


def build_site_energy_summary_table(
    *,
    timeseries_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Erstellt eine kompakte Energiebilanz über den gesamten Simulationshorizont.

    Zeilen:
    - Grundlast
    - Ladeinfrastruktur
    - Standort Gesamt

    Spalten:
    - PV-Energieverbrauch [kWh]
    - Netz-Energieverbrauch [kWh]
    - Summe Energieverbrauch [kWh]
    - Autarkiegrad [%]
      (= PV / Gesamtverbrauch * 100)
    - PV-Eigenverbrauchsquote [%]
      (= PV-Verbrauch / PV-Erzeugung * 100)

    Falls Debug-Spalten (pv_to_* / grid_to_*) existieren, werden diese für die Aufteilung genutzt.
    Andernfalls wird PV-first angenommen (PV deckt Grundlast zuerst, Rest-PV deckt EV).
    """
    df = timeseries_dataframe.copy()

    def _get_kwh_per_step(col_kwh: str, col_kw: str, step_h: float) -> np.ndarray:
        if col_kwh in df.columns:
            return df[col_kwh].astype(float).fillna(0.0).to_numpy()
        if col_kw in df.columns:
            return df[col_kw].astype(float).fillna(0.0).to_numpy() * step_h
        return np.zeros(len(df), dtype=float)

    if "timestamp" in df.columns and len(df) >= 2:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        step_h = float((ts.iloc[1] - ts.iloc[0]).total_seconds()) / 3600.0
    else:
        step_h = 0.25
    step_h = float(max(step_h, 1e-12))

    base_kwh = np.maximum(_get_kwh_per_step("base_load_kwh_per_step", "base_load_kw", step_h), 0.0)
    pv_gen_kwh = np.maximum(_get_kwh_per_step("pv_generation_kwh_per_step", "pv_generation_kw", step_h), 0.0)
    ev_kwh = np.maximum(_get_kwh_per_step("ev_load_kwh_per_step", "ev_load_kw", step_h), 0.0)

    has_debug = (
        "pv_to_base_kwh_per_step" in df.columns
        and "pv_to_ev_kwh_per_step" in df.columns
        and "grid_to_base_kwh_per_step" in df.columns
        and "grid_to_ev_kwh_per_step" in df.columns
    )

    if has_debug:
        pv_to_base = np.maximum(df["pv_to_base_kwh_per_step"].astype(float).fillna(0.0).to_numpy(), 0.0)
        pv_to_ev = np.maximum(df["pv_to_ev_kwh_per_step"].astype(float).fillna(0.0).to_numpy(), 0.0)
        grid_to_base = np.maximum(df["grid_to_base_kwh_per_step"].astype(float).fillna(0.0).to_numpy(), 0.0)
        grid_to_ev = np.maximum(df["grid_to_ev_kwh_per_step"].astype(float).fillna(0.0).to_numpy(), 0.0)
    else:
        pv_to_base = np.minimum(pv_gen_kwh, base_kwh)
        pv_after_base = np.maximum(pv_gen_kwh - pv_to_base, 0.0)
        pv_to_ev = np.minimum(pv_after_base, ev_kwh)

        grid_to_base = np.maximum(base_kwh - pv_to_base, 0.0)
        grid_to_ev = np.maximum(ev_kwh - pv_to_ev, 0.0)

    # Summen
    pv_gen_total = float(np.sum(pv_gen_kwh))

    base_pv = float(np.sum(pv_to_base))
    base_grid = float(np.sum(grid_to_base))
    base_total = base_pv + base_grid

    ev_pv = float(np.sum(pv_to_ev))
    ev_grid = float(np.sum(grid_to_ev))
    ev_total = ev_pv + ev_grid

    site_pv = base_pv + ev_pv
    site_grid = base_grid + ev_grid
    site_total = site_pv + site_grid

    def _ss_pct(pv: float, total: float) -> float:
        return float(100.0 * pv / total) if total > 1e-12 else 0.0

    def _pv_self_consumption_pct(pv: float, pv_gen: float) -> float:
        return float(100.0 * pv / pv_gen) if pv_gen > 1e-12 else 0.0

    out = pd.DataFrame(
        [
            {
                "row": "Grundlast",
                "PV-Energieverbrauch": base_pv,
                "Netz-Energieverbrauch": base_grid,
                "Summe Energieverbrauch": base_total,
                "Autarkiegrad": _ss_pct(base_pv, base_total),
                "PV-Eigenverbrauchsquote": _pv_self_consumption_pct(base_pv, pv_gen_total),
            },
            {
                "row": "Ladeinfrastruktur",
                "PV-Energieverbrauch": ev_pv,
                "Netz-Energieverbrauch": ev_grid,
                "Summe Energieverbrauch": ev_total,
                "Autarkiegrad": _ss_pct(ev_pv, ev_total),
                "PV-Eigenverbrauchsquote": _pv_self_consumption_pct(ev_pv, pv_gen_total),
            },
            {
                "row": "Standort Gesamt",
                "PV-Energieverbrauch": site_pv,
                "Netz-Energieverbrauch": site_grid,
                "Summe Energieverbrauch": site_total,
                "Autarkiegrad": _ss_pct(site_pv, site_total),
                "PV-Eigenverbrauchsquote": _pv_self_consumption_pct(site_pv, pv_gen_total),
            },
        ]
    ).set_index("row")

    out.index.name = None
    out.columns.name = None

    return out.round(2)


def build_pv_generation_and_surplus_table(*, timeseries_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt eine kompakte Tabelle zur PV-Erzeugung am Standort.

    Zeilen:
    - PV-Erzeugung
    - PV-Verbrauch
    - PV-Überschuss
    """
    df = timeseries_dataframe.copy()

    def _step_hours_from_timestamp(data: pd.DataFrame) -> float:
        if "timestamp" in data.columns and len(data) >= 2:
            ts = pd.to_datetime(data["timestamp"], errors="coerce")
            step_h = float((ts.iloc[1] - ts.iloc[0]).total_seconds()) / 3600.0
            return float(max(step_h, 1e-12))
        return 0.25  # Fallback

    def _get_kwh_per_step(col_kwh: str, col_kw: str, step_h: float) -> np.ndarray:
        if col_kwh in df.columns:
            return df[col_kwh].astype(float).fillna(0.0).to_numpy()
        if col_kw in df.columns:
            return df[col_kw].astype(float).fillna(0.0).to_numpy() * step_h
        return np.zeros(len(df), dtype=float)

    step_h = _step_hours_from_timestamp(df)

    base_kwh = np.maximum(_get_kwh_per_step("base_load_kwh_per_step", "base_load_kw", step_h), 0.0)
    pv_gen_kwh = np.maximum(_get_kwh_per_step("pv_generation_kwh_per_step", "pv_generation_kw", step_h), 0.0)
    ev_kwh = np.maximum(_get_kwh_per_step("ev_load_kwh_per_step", "ev_load_kw", step_h), 0.0)

    has_debug = (
        "pv_to_base_kwh_per_step" in df.columns
        and "pv_to_ev_kwh_per_step" in df.columns
        and "grid_to_base_kwh_per_step" in df.columns
        and "grid_to_ev_kwh_per_step" in df.columns
    )

    if has_debug:
        pv_to_base = np.maximum(df["pv_to_base_kwh_per_step"].astype(float).fillna(0.0).to_numpy(), 0.0)
        pv_to_ev = np.maximum(df["pv_to_ev_kwh_per_step"].astype(float).fillna(0.0).to_numpy(), 0.0)
    else:
        pv_to_base = np.minimum(pv_gen_kwh, base_kwh)
        pv_after_base = np.maximum(pv_gen_kwh - pv_to_base, 0.0)
        pv_to_ev = np.minimum(pv_after_base, ev_kwh)

    pv_gen_total = float(np.sum(pv_gen_kwh))
    pv_consumption = float(np.sum(pv_to_base) + np.sum(pv_to_ev))
    pv_surplus = float(max(pv_gen_total - pv_consumption, 0.0))

    out = pd.DataFrame(
        [
        {"row": "PV-Erzeugung", "Energie [kWh]": pv_gen_total},
        {"row": "PV-Verbrauch", "Energie [kWh]": pv_consumption},
        {"row": "PV-Überschuss", "Energie [kWh]": pv_surplus},
        ]
    ).set_index("row")

    out.index.name = None
    out.columns.name = None
    return out.round(2)

def build_energy_ev_profitability_table(timeseries_dataframe: pd.DataFrame, scenario: dict):
    df = timeseries_dataframe.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Schrittbreite in Stunden
    if len(df) >= 2 and pd.notna(df["timestamp"].iloc[0]) and pd.notna(df["timestamp"].iloc[1]):
        step_h = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds() / 3600.0
    else:
        step_h = float(scenario.get("time_resolution_min", 15)) / 60.0
    step_h = max(float(step_h), 1e-12)

    # --- Config lesen ---
    ee = scenario.get("energy_expenses") or {}
    cost_source = str(ee.get("energy_cost_source", "fixed")).strip().lower()

    sell_ct = float(ee.get("energy_selling_price_ct_per_kwh", 0.0) or 0.0)
    sell_eur = sell_ct / 100.0  # ct/kWh -> €/kWh

    # --- Basis-Spalten prüfen ---
    if "ev_load_kw" not in df.columns:
        raise ValueError("timeseries_dataframe muss die Spalte 'ev_load_kw' enthalten.")
    if "base_load_kw" not in df.columns:
        raise ValueError("timeseries_dataframe muss die Spalte 'base_load_kw' enthalten.")

    ev_kw = df["ev_load_kw"].astype(float).fillna(0.0).to_numpy()
    base_kw = df["base_load_kw"].astype(float).fillna(0.0).to_numpy()
    ev_sold_kwh = float(np.sum(ev_kw) * step_h)

    # ------------------------------------------------------------
    # Netzanteile bestimmen: Grundlast (grid_to_base_kw) & LIS (grid_to_ev_kw)
    # Priorität:
    #   1) Debug-Spalten (exakt)
    #   2) PV-first Näherung (pv deckt base zuerst, Rest pv -> ev)
    # ------------------------------------------------------------
    if {"grid_to_ev_kw_per_step", "grid_to_base_kw_per_step"}.issubset(df.columns):
        grid_to_ev_kw = df["grid_to_ev_kw_per_step"].astype(float).fillna(0.0).to_numpy()
        grid_to_base_kw = df["grid_to_base_kw_per_step"].astype(float).fillna(0.0).to_numpy()

    else:
        # PV-first Näherung benötigt pv_generation_kw
        if "pv_generation_kw" not in df.columns:
            raise ValueError(
                "Für die Aufteilung Grundlast/LIS ohne Debug-Spalten muss 'pv_generation_kw' vorhanden sein "
                "(oder nutze debug_df, damit 'grid_to_base_kw_per_step'/'grid_to_ev_kw_per_step' vorhanden sind)."
            )

        pv_kw = df["pv_generation_kw"].astype(float).fillna(0.0).to_numpy()

        pv_to_base_kw = np.minimum(pv_kw, base_kw)
        base_grid_kw = np.maximum(base_kw - pv_to_base_kw, 0.0)

        pv_after_base_kw = np.maximum(pv_kw - pv_to_base_kw, 0.0)
        pv_to_ev_kw = np.minimum(pv_after_base_kw, ev_kw)
        ev_grid_kw = np.maximum(ev_kw - pv_to_ev_kw, 0.0)

        grid_to_base_kw = base_grid_kw
        grid_to_ev_kw = ev_grid_kw

    # Energiemengen
    grid_base_kwh = float(np.sum(np.maximum(grid_to_base_kw, 0.0)) * step_h)
    grid_ev_kwh = float(np.sum(np.maximum(grid_to_ev_kw, 0.0)) * step_h)
    grid_site_kwh = float(grid_base_kwh + grid_ev_kwh)

    # ------------------------------------------------------------
    # Energiekosten je nach Quelle (fixed / csv)
    # -> getrennt für Grundlast, LIS, Standort
    # ------------------------------------------------------------
    def compute_costs_and_avg_price_ct(grid_kw_per_step: np.ndarray):
        grid_kw_per_step = np.maximum(np.asarray(grid_kw_per_step, float), 0.0)
        grid_kwh_per_step = grid_kw_per_step * step_h
        grid_kwh_total = float(np.sum(grid_kwh_per_step))

        if cost_source == "fixed":
            buy_ct = float(ee.get("fixed_energy_cost_ct_per_kwh", 0.0) or 0.0)
            buy_eur = buy_ct / 100.0
            costs_eur = float(grid_kwh_total * buy_eur)
            avg_ct = float(buy_ct) if grid_kwh_total > 1e-12 else 0.0
            return grid_kwh_total, avg_ct, costs_eur

        if cost_source == "csv":
            if "market_price_eur_per_mwh" not in df.columns:
                raise ValueError(
                    "energy_cost_source='csv' gesetzt, aber 'market_price_eur_per_mwh' fehlt im timeseries_dataframe."
                )

            price_eur_per_mwh = (
                df["market_price_eur_per_mwh"]
                .astype(float)
                .ffill()
                .fillna(0.0)
                .to_numpy()
            )
            price_eur_per_kwh = price_eur_per_mwh / 1000.0

            costs_eur = float(np.sum(grid_kwh_per_step * price_eur_per_kwh))
            avg_eur = float(costs_eur / grid_kwh_total) if grid_kwh_total > 1e-12 else 0.0
            avg_ct = float(avg_eur * 100.0)
            return grid_kwh_total, avg_ct, costs_eur

        raise ValueError("energy_cost_source muss 'fixed' oder 'csv' sein.")

    # Kosten/Preise pro Teilbereich
    _, avg_buy_base_ct, energy_costs_base_eur = compute_costs_and_avg_price_ct(grid_to_base_kw)
    _, avg_buy_ev_ct, energy_costs_ev_eur = compute_costs_and_avg_price_ct(grid_to_ev_kw)

    # Standort aggregiert (gewichtet korrekt über Kosten / kWh)
    energy_costs_site_eur = float(energy_costs_base_eur + energy_costs_ev_eur)
    avg_buy_site_ct = float((energy_costs_site_eur / grid_site_kwh) * 100.0) if grid_site_kwh > 1e-12 else 0.0

    # Erlöse/Marge (nur LIS verkauft)
    revenues_eur = float(ev_sold_kwh * sell_eur)
    margin_eur = float(revenues_eur - energy_costs_ev_eur)

    rows = [
        ("Energiekostenquelle", "CSV" if cost_source == "csv" else "Fixpreis"),

        ("Netzbezug Grundlast [kWh]", grid_base_kwh),
        ("Ø Einkaufspreis Grundlast [ct/kWh]", avg_buy_base_ct),
        ("Energiekosten Netzbezug Grundlast [€]", energy_costs_base_eur),

        ("Netzbezug LIS [kWh]", grid_ev_kwh),
        ("Ø Einkaufspreis LIS [ct/kWh]", avg_buy_ev_ct),
        ("Energiekosten Netzbezug LIS [€]", energy_costs_ev_eur),

        ("Netzbezug Standort [kWh]", grid_site_kwh),
        ("Ø Einkaufspreis Standort [ct/kWh]", avg_buy_site_ct),
        ("Energiekosten Netzbezug Standort [€]", energy_costs_site_eur),

        ("", ""),  # optische Trennung

        ("Verkaufte Energie LIS [kWh]", ev_sold_kwh),
        ("Verkaufspreis [ct/kWh]", sell_ct),
        ("Erlöse Ladeinfrastruktur [€]", revenues_eur),
        ("Energiemarge [€]", margin_eur),
    ]

    out = pd.DataFrame(rows, columns=["", ""])

    sty = out.style
    try:
        sty = sty.hide(axis="index").hide(axis="columns")
    except Exception:
        sty = sty.hide_index().hide_columns()

    return sty


def build_grid_work_and_peak_table(
    *,
    timeseries_dataframe: pd.DataFrame,
    scenario: dict,
    include_peak_timestamp: bool = True,
):
    dataframe = timeseries_dataframe.copy()
    if "timestamp" not in dataframe:
        raise ValueError("timeseries_dataframe muss eine Spalte 'timestamp' enthalten.")

    timestamp_series = pd.to_datetime(dataframe["timestamp"], errors="coerce")
    if timestamp_series.isna().all():
        raise ValueError("Spalte 'timestamp' konnte nicht geparst werden.")

    timezone_name = str(scenario.get("timezone", "Europe/Berlin"))
    try:
        timestamp_series = (
            timestamp_series.dt.tz_localize(timezone_name, ambiguous="infer", nonexistent="shift_forward")
            if timestamp_series.dt.tz is None
            else timestamp_series.dt.tz_convert(timezone_name)
        )
    except Exception:
        pass

    dataframe = (
        dataframe.assign(timestamp=timestamp_series)
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    # grid_kw (debug bevorzugt, sonst fallback)
    if {"grid_to_base_kw_per_step", "grid_to_ev_kw_per_step"}.issubset(dataframe.columns):
        grid_power_kw = (
            dataframe[["grid_to_base_kw_per_step", "grid_to_ev_kw_per_step"]]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .sum(axis=1)
        )
    else:
        if not {"base_load_kw", "ev_load_kw"}.issubset(dataframe.columns):
            raise ValueError(
                "Ohne Debug-Spalten braucht es mindestens 'base_load_kw' und 'ev_load_kw' "
                "(optional 'pv_generation_kw')."
            )

        base_load_kw = pd.to_numeric(dataframe["base_load_kw"], errors="coerce").fillna(0.0)
        ev_load_kw = pd.to_numeric(dataframe["ev_load_kw"], errors="coerce").fillna(0.0)
        pv_generation_kw = (
            pd.to_numeric(dataframe["pv_generation_kw"], errors="coerce").fillna(0.0)
            if "pv_generation_kw" in dataframe
            else 0.0
        )

        grid_power_kw = (base_load_kw + ev_load_kw - pv_generation_kw).clip(lower=0.0)

        grid_limit_kw = float((scenario.get("site") or {}).get("grid_limit_p_avb_kw", np.inf))
        if np.isfinite(grid_limit_kw) and grid_limit_kw > 0:
            grid_power_kw = grid_power_kw.clip(upper=grid_limit_kw)

    timestamp_index = dataframe.index
    time_step_hours = (timestamp_index.to_series().shift(-1) - timestamp_index.to_series()).dt.total_seconds().div(3600.0)

    typical_time_step_hours = float(np.nanmedian(time_step_hours)) if len(time_step_hours) else np.nan
    if not np.isfinite(typical_time_step_hours) or typical_time_step_hours <= 0:
        typical_time_step_hours = float(scenario.get("time_resolution_min", 15)) / 60.0

    time_step_hours = time_step_hours.fillna(typical_time_step_hours).clip(lower=0.0)

    grid_timeseries_dataframe = pd.DataFrame(
        {"grid_kw": grid_power_kw.clip(lower=0.0).astype(float)},
        index=timestamp_index,
    )
    grid_timeseries_dataframe["grid_kwh"] = grid_timeseries_dataframe["grid_kw"] * time_step_hours.to_numpy()

    month_key = (
        (timestamp_index.tz_localize(None) if timestamp_index.tz is not None else timestamp_index)
        .to_period("M")
    )
    grouped_by_month = grid_timeseries_dataframe.groupby(month_key)

    monthly_dataframe = pd.DataFrame(
        {
            "Arbeit W [kWh]": grouped_by_month["grid_kwh"].sum(),
            "Höchstleistung P [kW]": grouped_by_month["grid_kw"].max(),
        }
    )

    total_row = {
        "Arbeit W [kWh]": float(grid_timeseries_dataframe["grid_kwh"].sum()),
        "Höchstleistung P [kW]": float(grid_timeseries_dataframe["grid_kw"].max()) if len(grid_timeseries_dataframe) else 0.0,
    }
    monthly_dataframe.loc["Gesamt"] = total_row

    numeric_columns = ["Arbeit W [kWh]", "Höchstleistung P [kW]"]
    monthly_dataframe[numeric_columns] = monthly_dataframe[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).round(2)

    output_dataframe = monthly_dataframe.rename_axis("Monat").reset_index()
    styler = output_dataframe.style.format({column_name: "{:.2f}" for column_name in numeric_columns})
    try:
        return styler.hide(axis="index")
    except Exception:
        return styler.hide_index()


def build_site_power_balance_top10_table(
    *,
    timeseries_dataframe: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """
    Erstellt eine Tabelle der n Zeitpunkte mit dem höchsten Standortverbrauch.

    Spalten:
    - Zeitpunkt
    - Leistung aus NAP [kW]
    - Leistung PV-Erzeugung [kW]
    - Versorgungsleistung Standort [kW]
    - Verbrauch Grundlast [kW]
    - Verbrauch LIS [kW]
    - Verbrauch Standort [kW]

    Hinweis:
    - 'Versorgungsleistung Standort' ist die tatsächlich für den Verbrauch
      genutzte Leistung:
          Leistung aus NAP + PV-Verbrauch
      und entspricht damit dem Standortverbrauch.
    - Falls Debug-/Flussspalten vorhanden sind, werden diese bevorzugt genutzt.
      Andernfalls wird PV-first angenommen.
    """
    df = timeseries_dataframe.copy()

    if "timestamp" not in df.columns:
        raise ValueError("timeseries_dataframe muss die Spalte 'timestamp' enthalten.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    def _get_kw(col_kw: str) -> np.ndarray:
        if col_kw in df.columns:
            return df[col_kw].astype(float).fillna(0.0).to_numpy()
        return np.zeros(len(df), dtype=float)

    base_kw = np.maximum(_get_kw("base_load_kw"), 0.0)
    pv_gen_kw = np.maximum(_get_kw("pv_generation_kw"), 0.0)
    ev_kw = np.maximum(_get_kw("ev_load_kw"), 0.0)

    has_debug_kw = {
        "pv_to_base_kw_per_step",
        "pv_to_ev_kw_per_step",
        "grid_to_base_kw_per_step",
        "grid_to_ev_kw_per_step",
    }.issubset(df.columns)

    has_debug_kwh = {
        "pv_to_base_kwh_per_step",
        "pv_to_ev_kwh_per_step",
        "grid_to_base_kwh_per_step",
        "grid_to_ev_kwh_per_step",
    }.issubset(df.columns)

    if len(df) >= 2:
        step_h = float((df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds()) / 3600.0
    else:
        step_h = 0.25
    step_h = float(max(step_h, 1e-12))

    if has_debug_kw:
        pv_to_base_kw = np.maximum(df["pv_to_base_kw_per_step"].astype(float).fillna(0.0).to_numpy(), 0.0)
        pv_to_ev_kw = np.maximum(df["pv_to_ev_kw_per_step"].astype(float).fillna(0.0).to_numpy(), 0.0)
        grid_to_base_kw = np.maximum(df["grid_to_base_kw_per_step"].astype(float).fillna(0.0).to_numpy(), 0.0)
        grid_to_ev_kw = np.maximum(df["grid_to_ev_kw_per_step"].astype(float).fillna(0.0).to_numpy(), 0.0)

    elif has_debug_kwh:
        pv_to_base_kw = np.maximum(df["pv_to_base_kwh_per_step"].astype(float).fillna(0.0).to_numpy() / step_h, 0.0)
        pv_to_ev_kw = np.maximum(df["pv_to_ev_kwh_per_step"].astype(float).fillna(0.0).to_numpy() / step_h, 0.0)
        grid_to_base_kw = np.maximum(df["grid_to_base_kwh_per_step"].astype(float).fillna(0.0).to_numpy() / step_h, 0.0)
        grid_to_ev_kw = np.maximum(df["grid_to_ev_kwh_per_step"].astype(float).fillna(0.0).to_numpy() / step_h, 0.0)

    else:
        pv_to_base_kw = np.minimum(pv_gen_kw, base_kw)
        pv_after_base_kw = np.maximum(pv_gen_kw - pv_to_base_kw, 0.0)
        pv_to_ev_kw = np.minimum(pv_after_base_kw, ev_kw)

        grid_to_base_kw = np.maximum(base_kw - pv_to_base_kw, 0.0)
        grid_to_ev_kw = np.maximum(ev_kw - pv_to_ev_kw, 0.0)

    power_from_nap_kw = grid_to_base_kw + grid_to_ev_kw
    pv_used_kw = pv_to_base_kw + pv_to_ev_kw

    supplied_site_kw = power_from_nap_kw + pv_used_kw
    site_consumption_kw = base_kw + ev_kw

    out = pd.DataFrame(
        {
            "Zeitpunkt": df["timestamp"].dt.strftime("%d.%m.%y, %H:%M:%S"),
            "Leistung aus NAP [kW]": power_from_nap_kw,
            "Leistung PV-Erzeugung [kW]": pv_gen_kw,
            "Versorgungsleistung Standort [kW]": supplied_site_kw,
            "Verbrauch Grundlast [kW]": base_kw,
            "Verbrauch LIS [kW]": ev_kw,
            "Verbrauch Standort [kW]": site_consumption_kw,
        }
    )

    out = (
        out.sort_values(
            by=["Verbrauch Standort [kW]", "Zeitpunkt"],
            ascending=[False, True],
        )
        .head(int(n))
        .reset_index(drop=True)
    )

    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].round(2)

    return out


def export_site_energy_balance_excel(
    *,
    timeseries_dataframe: pd.DataFrame,
    scenario: dict,
    excel_path: str | Path = "site_energy_balance.xlsx",
    sheet_name: str = "Lastgang",
) -> pd.DataFrame:
    """
    Exportiert den Lastgang als Excel in kWh pro 15-Minuten-Schritt.

    Spalten:
    1. Datum + Uhrzeit
    2. PV-Erzeugung [kWh/15 min]
    3. Netzbezug [kWh/15 min]
    4. Verfügbare Energie [kWh/15 min]
    5. Grundlast aus Netz [kWh/15 min]
    6. Grundlast aus PV [kWh/15 min]
    7. Grundlast gesamt [kWh/15 min]
    8. LIS aus Netz [kWh/15 min]
    9. LIS aus PV [kWh/15 min]
    10. LIS gesamt [kWh/15 min]
    11. Standortverbrauch gesamt [kWh/15 min]
    12. Marktpreis [€/MWh]
    13. Freie Netzkapazität [kWh/15 min]
    14. PV-Überschuss [kWh/15 min]

    Rückgabe
    --------
    pd.DataFrame
        Exportiertes DataFrame; zusätzlich wird eine Excel-Datei geschrieben.
    """
    df = timeseries_dataframe.copy()
    if "timestamp" not in df.columns:
        raise ValueError("timeseries_dataframe muss die Spalte 'timestamp' enthalten.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if len(df) >= 2:
        step_h = float((df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds()) / 3600.0
    else:
        step_h = float(scenario.get("time_resolution_min", 15)) / 60.0
    step_h = float(max(step_h, 1e-12))

    def _get_kwh_per_step(col_kwh: str, col_kw: str) -> np.ndarray:
        if col_kwh in df.columns:
            return df[col_kwh].astype(float).fillna(0.0).to_numpy()
        if col_kw in df.columns:
            return df[col_kw].astype(float).fillna(0.0).to_numpy() * step_h
        return np.zeros(len(df), dtype=float)

    base_kwh = np.maximum(_get_kwh_per_step("base_load_kwh_per_step", "base_load_kw"), 0.0)
    pv_gen_kwh = np.maximum(_get_kwh_per_step("pv_generation_kwh_per_step", "pv_generation_kw"), 0.0)
    ev_kwh = np.maximum(_get_kwh_per_step("ev_load_kwh_per_step", "ev_load_kw"), 0.0)

    has_debug_kwh = {
        "pv_to_base_kwh_per_step",
        "pv_to_ev_kwh_per_step",
        "grid_to_base_kwh_per_step",
        "grid_to_ev_kwh_per_step",
    }.issubset(df.columns)

    has_debug_kw = {
        "pv_to_base_kw_per_step",
        "pv_to_ev_kw_per_step",
        "grid_to_base_kw_per_step",
        "grid_to_ev_kw_per_step",
    }.issubset(df.columns)

    if has_debug_kwh:
        pv_to_base = np.maximum(df["pv_to_base_kwh_per_step"].astype(float).fillna(0.0).to_numpy(), 0.0)
        pv_to_ev = np.maximum(df["pv_to_ev_kwh_per_step"].astype(float).fillna(0.0).to_numpy(), 0.0)
        grid_to_base = np.maximum(df["grid_to_base_kwh_per_step"].astype(float).fillna(0.0).to_numpy(), 0.0)
        grid_to_ev = np.maximum(df["grid_to_ev_kwh_per_step"].astype(float).fillna(0.0).to_numpy(), 0.0)

    elif has_debug_kw:
        pv_to_base = np.maximum(df["pv_to_base_kw_per_step"].astype(float).fillna(0.0).to_numpy() * step_h, 0.0)
        pv_to_ev = np.maximum(df["pv_to_ev_kw_per_step"].astype(float).fillna(0.0).to_numpy() * step_h, 0.0)
        grid_to_base = np.maximum(df["grid_to_base_kw_per_step"].astype(float).fillna(0.0).to_numpy() * step_h, 0.0)
        grid_to_ev = np.maximum(df["grid_to_ev_kw_per_step"].astype(float).fillna(0.0).to_numpy() * step_h, 0.0)

    else:
        pv_to_base = np.minimum(pv_gen_kwh, base_kwh)
        pv_after_base = np.maximum(pv_gen_kwh - pv_to_base, 0.0)
        pv_to_ev = np.minimum(pv_after_base, ev_kwh)

        grid_to_base = np.maximum(base_kwh - pv_to_base, 0.0)
        grid_to_ev = np.maximum(ev_kwh - pv_to_ev, 0.0)

    energy_from_nap = grid_to_base + grid_to_ev
    total_energy_available = pv_gen_kwh + energy_from_nap

    base_sum = grid_to_base + pv_to_base
    ev_sum = grid_to_ev + pv_to_ev
    site_sum = base_sum + ev_sum

    pv_surplus = np.maximum(pv_gen_kwh - (pv_to_base + pv_to_ev), 0.0)

    grid_limit_kw = float((scenario.get("site") or {}).get("grid_limit_p_avb_kw", 0.0) or 0.0)
    remaining_grid_energy = np.maximum(grid_limit_kw * step_h - energy_from_nap, 0.0)

    if "market_price_eur_per_mwh" in df.columns:
        market_price = df["market_price_eur_per_mwh"].astype(float).ffill().fillna(0.0).to_numpy()
    else:
        market_price = np.full(len(df), np.nan)

    out = pd.DataFrame(
        {
            "Datum + Uhrzeit": df["timestamp"].dt.strftime("%d.%m.%y, %H:%M"),
            "PV-Erzeugung [kWh/15 min]": pv_gen_kwh,
            "Netzbezug [kWh/15 min]": energy_from_nap,
            "Verfügbage Energie [kWh/15 min]": total_energy_available,
            "Grundlast aus Netz [kWh/15 min]": grid_to_base,
            "Grundlast aus PV [kWh/15 min]": pv_to_base,
            "Grundlast gesamt [kWh/15 min]": base_sum,
            "LIS aus Netz [kWh/15 min]": grid_to_ev,
            "LIS aus PV [kWh/15 min]": pv_to_ev,
            "LIS gesamt [kWh/15 min]": ev_sum,
            "Standortverbrauch gesamt [kWh/15 min]": site_sum,
            "Marktpreis [€/MWh]": market_price,
            "Freie Netzkapazität [kWh/15 min]": remaining_grid_energy,
            "PV-Überschuss [kWh/15 min]": pv_surplus,
        }
    ).round(2)

    excel_path = Path(excel_path)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name=sheet_name, index=False)

        worksheet = writer.sheets[sheet_name]
        worksheet.freeze_panes = "A2"

        for column_cells in worksheet.columns:
            max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
            worksheet.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 28)

    return out

