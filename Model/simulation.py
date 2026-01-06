import yaml
import numpy as np
from pathlib import Path
import csv  # CSV-Parsing (Fahrzeugkurven + Markt/Erzeugungssignal)
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Any


# =============================================================================
# 0) Projekt-/Pfad-Utilities (NEU)
# =============================================================================

def resolve_path_relative_to_scenario(scenario: dict[str, Any], p: str) -> str:
    """
    Löst Dateipfade robust auf:
      - Absolute Pfade bleiben unverändert
      - Relative Pfade werden relativ zum YAML-Ordner (scenario["_scenario_dir"]) interpretiert

    Ziel: Portables Projekt für verschiedene Nutzer/Computer.
    """
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return str(pp)

    base = Path(scenario.get("_scenario_dir", "."))
    return str((base / pp).resolve())


# =============================================================================
# 0b) Robustes Zahlen-Parsing (NEU)
# =============================================================================

def parse_number_de_or_en(raw: str) -> float:
    """
    Robust gegenüber:
      - deutschem Format:  "1.234,56"
      - englischem Format: "1,234.56" oder "90.91"
      - deutschem Dezimalkomma ohne Tausender: "90,91"

    Regeln:
      - Falls ',' UND '.' vorkommen:
          - Wenn das letzte Trennzeichen ',' ist -> de: '.' Tausender, ',' Dezimal
          - Wenn das letzte Trennzeichen '.' ist -> en: ',' Tausender, '.' Dezimal
      - Falls nur ',' vorkommt: ',' Dezimal
      - Falls nur '.' vorkommt: '.' Dezimal
    """
    s = (raw or "").strip().replace(" ", "")
    if s == "" or s == "-":
        raise ValueError("Empty numeric cell")

    has_comma = "," in s
    has_dot = "." in s

    if has_comma and has_dot:
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_comma > last_dot:
            # deutsch: 1.234,56
            s = s.replace(".", "").replace(",", ".")
            return float(s)
        else:
            # englisch: 1,234.56
            s = s.replace(",", "")
            return float(s)

    if has_comma and not has_dot:
        # deutsch: 90,91
        s = s.replace(",", ".")
        return float(s)

    # nur '.' oder kein Separator
    # englisch: 90.91 oder "1234"
    return float(s)


# =============================================================================
# 0c) HTML-Statusausgabe (NEU)
# =============================================================================

def show_strategy_status_html(strategy: str, status: str) -> None:
    """
    Zeigt im Jupyter Notebook eine farbige Statuszeile (HTML).
    Fallback: normaler print, wenn IPython nicht verfügbar ist.

    status: "ACTIVE" | "PARTIAL" | "INACTIVE" | "IMMEDIATE"
    """
    status = (status or "IMMEDIATE").upper()
    strategy = (strategy or "immediate").upper()

    color_map = {
        "ACTIVE": "green",
        "PARTIAL": "orange",
        "INACTIVE": "red",
        "IMMEDIATE": "gray",
    }
    emoji_map = {
        "ACTIVE": "🟢",
        "PARTIAL": "🟡",
        "INACTIVE": "🔴",
        "IMMEDIATE": "⚪",
    }

    color = color_map.get(status, "gray")
    emoji = emoji_map.get(status, "⚪")

    html = (
        f"<div style='font-size:18px; font-weight:700; color:{color}; "
        f"padding:6px 10px; border:1px solid #ddd; border-radius:8px; "
        f"display:inline-block; margin:6px 0;'>"
        f"{emoji} Charging strategy: {strategy} — {status}"
        f"</div>"
    )

    try:
        from IPython.display import display, HTML  # type: ignore
        display(HTML(html))
    except Exception:
        print(f"{emoji} Charging strategy: {strategy} — {status}")


# =============================================================================
# 1) Fahrzeuge & fahrzeugspezifische Ladekurven
# =============================================================================

@dataclass
class VehicleProfile:
    name: str
    battery_capacity_kwh: float
    vehicle_class: str            # z.B. "PKW", "Transporter"
    soc_grid: np.ndarray          # SoC-Stützstellen (0..1)
    power_grid_kw: np.ndarray     # Ladeleistung in kW zu den SoC-Stützstellen


def load_vehicle_profiles_from_csv(path: str) -> list[VehicleProfile]:
    """
    Liest eine CSV mit Fahrzeugen und SoC-abhängigen Ladekurven und gibt
    eine Liste von VehicleProfile-Objekten zurück.

    Erwartete Struktur (Spaltentrenner ';'):
      Zeile 1: Hersteller (ignoriert)
      Zeile 2: Modellnamen
      Zeile 3: Fahrzeugklasse
      Zeile 4: max. Kapazität (kWh)
      Zeile 5: "SoC [%]" (Header)
      ab Zeile 6: SoC-Werte in % + Ladeleistungen je Fahrzeug
    """
    vehicle_profiles: list[VehicleProfile] = []

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";")

        _brands_row = next(reader, None)

        model_row = next(reader, None)
        if not model_row or len(model_row) < 2:
            return []
        model_names = [m.strip() for m in model_row[1:]]

        class_row = next(reader, None)
        if not class_row or len(class_row) < 2:
            return []
        vehicle_classes = [c.strip() if c.strip() != "" else "PKW" for c in class_row[1:]]

        capacity_row = next(reader, None)
        if not capacity_row or len(capacity_row) < 2:
            return []
        raw_capacities = capacity_row[1:]

        _soc_header_row = next(reader, None)

        capacities_kwh: list[float] = []
        for val in raw_capacities:
            val = (val or "").strip()
            if val == "":
                capacities_kwh.append(np.nan)
                continue
            try:
                cap = parse_number_de_or_en(val)
                capacities_kwh.append(cap if cap > 0 else np.nan)
            except ValueError:
                capacities_kwh.append(np.nan)

        num_vehicles = min(len(model_names), len(vehicle_classes), len(capacities_kwh))
        model_names = model_names[:num_vehicles]
        vehicle_classes = vehicle_classes[:num_vehicles]
        capacities_kwh = capacities_kwh[:num_vehicles]

        soc_lists: list[list[float]] = [[] for _ in range(num_vehicles)]
        power_lists: list[list[float]] = [[] for _ in range(num_vehicles)]

        for row in reader:
            if not row:
                continue

            soc_str = (row[0] or "").strip()
            if soc_str == "":
                continue

            try:
                soc_val_percent = parse_number_de_or_en(soc_str)
            except ValueError:
                continue

            soc_val = soc_val_percent / 100.0

            for idx in range(num_vehicles):
                if idx + 1 >= len(row):
                    continue
                cell = (row[idx + 1] or "").strip()
                if cell == "":
                    continue
                try:
                    power_kw = parse_number_de_or_en(cell)
                except ValueError:
                    continue

                soc_lists[idx].append(soc_val)
                power_lists[idx].append(power_kw)

        for i in range(num_vehicles):
            name = model_names[i]
            vclass = vehicle_classes[i]
            cap = capacities_kwh[i]

            if np.isnan(cap) or len(soc_lists[i]) == 0:
                continue

            soc_grid = np.array(soc_lists[i], dtype=float)
            power_grid = np.array(power_lists[i], dtype=float)

            sort_idx = np.argsort(soc_grid)
            soc_grid = soc_grid[sort_idx]
            power_grid = power_grid[sort_idx]

            vehicle_profiles.append(
                VehicleProfile(
                    name=name,
                    battery_capacity_kwh=float(cap),
                    vehicle_class=vclass,
                    soc_grid=soc_grid,
                    power_grid_kw=power_grid,
                )
            )

    return vehicle_profiles


def vehicle_power_at_soc(session: dict[str, Any]) -> float:
    """
    Fahrzeugspezifische maximale Ladeleistung (kW) für aktuellen SoC,
    basierend auf der in der Session gespeicherten Ladekurve.
    """
    soc_grid = session["soc_grid"]
    power_grid = session["power_grid_kw"]

    soc_arrival = session["soc_arrival"]
    delivered_energy = session.get("delivered_energy_kwh", 0.0)
    capacity = session["battery_capacity_kwh"]

    current_soc = soc_arrival + delivered_energy / capacity
    current_soc = min(current_soc, session["soc_target"])

    power_kw = float(np.interp(current_soc, soc_grid, power_grid))
    return max(power_kw, 0.0)


# =============================================================================
# 2) Szenario laden (YAML) + Pfadkontext (NEU)
# =============================================================================

def load_scenario(path: str) -> dict[str, Any]:
    """
    Liest eine YAML-Szenariodatei ein und gibt sie als Dictionary zurück.
    Merkt sich zusätzlich den YAML-Ordner für portable relative Pfade.
    """
    with open(path, "r", encoding="utf-8") as file:
        scenario = yaml.safe_load(file)

    scenario["_scenario_dir"] = str(Path(path).resolve().parent)
    return scenario


# =============================================================================
# 3) Hilfsfunktionen: Ranges, Feiertage, Zeitindex
# =============================================================================

def sample_from_range(value_definition: Any) -> float:
    """
    Verarbeitet Werte aus der YAML, die entweder als Skalar oder als Liste
    [min, max] (bzw. [value]) angegeben sind, und gibt einen float zurück.
    """
    if isinstance(value_definition, (list, tuple)):
        if len(value_definition) == 1:
            return float(value_definition[0])
        if len(value_definition) == 2:
            lower_bound, upper_bound = value_definition
            return float(np.random.uniform(lower_bound, upper_bound))
        raise ValueError(f"Ungültiges Range-Format: {value_definition}")

    return float(value_definition)


def parse_holiday_dates_from_scenario(
    scenario: dict,
    simulation_start_datetime: datetime,
) -> set[date]:
    """
    Erzeugt Feiertage als Set[date] aus der YAML.

    Unterstützte Varianten (kombinierbar):
      A) Manuelle Liste:
         holidays:
           dates: ["2025-01-01", "2025-12-25", ...]
      B) Automatisch nach Land + Bundesland:
         holidays:
           country: "DE"
           subdivision: "BY"
    """
    holidays_cfg = scenario.get("holidays", {}) or {}
    holiday_dates: set[date] = set()

    country = holidays_cfg.get("country", None)
    subdivision = holidays_cfg.get("subdivision", None)

    if country:
        try:
            import holidays as holidays_lib
        except ImportError as e:
            raise ImportError(
                "Für automatische Feiertage bitte das Paket 'holidays' installieren: pip install holidays"
            ) from e

        horizon_days = int(scenario.get("simulation_horizon_days", 1))
        simulation_end_datetime = simulation_start_datetime + timedelta(days=horizon_days)

        years = list(range(simulation_start_datetime.year, simulation_end_datetime.year + 1))
        hol = holidays_lib.country_holidays(country, subdiv=subdivision, years=years)
        holiday_dates |= set(hol.keys())

    dates_list = holidays_cfg.get("dates", None)
    if isinstance(dates_list, list):
        for date_string in dates_list:
            if isinstance(date_string, str) and date_string.strip():
                holiday_dates.add(datetime.fromisoformat(date_string).date())

    return holiday_dates


def determine_day_type_with_holidays(
    current_datetime: datetime,
    holiday_dates: set[date],
) -> str:
    """
    Tagtyp inkl. Feiertage:
      - 'sunday_holiday' (Feiertag oder Sonntag)
      - 'saturday'
      - 'working_day'
    """
    current_date = current_datetime.date()

    if current_date in holiday_dates:
        return "sunday_holiday"

    weekday_index = current_datetime.weekday()  # Mo=0 ... So=6
    if weekday_index == 6:
        return "sunday_holiday"
    if weekday_index == 5:
        return "saturday"
    return "working_day"


def create_time_index(scenario: dict, start_datetime: datetime | None = None) -> list[datetime]:
    """
    Erzeugt eine Liste von Zeitstempeln über den im Szenario definierten
    Simulationshorizont.
    """
    if start_datetime is not None:
        simulation_start_datetime = start_datetime
    elif "start_datetime" in scenario:
        simulation_start_datetime = datetime.fromisoformat(scenario["start_datetime"])
    else:
        now = datetime.now()
        simulation_start_datetime = now.replace(hour=0, minute=0, second=0, microsecond=0)

    time_resolution_min = scenario["time_resolution_min"]
    simulation_horizon_days = scenario["simulation_horizon_days"]

    total_minutes_in_simulation = simulation_horizon_days * 24 * 60
    number_of_time_steps = int(total_minutes_in_simulation / time_resolution_min)

    time_step_delta = timedelta(minutes=time_resolution_min)
    time_index = [
        simulation_start_datetime + step_index * time_step_delta
        for step_index in range(number_of_time_steps)
    ]
    return time_index


# =============================================================================
# 4) Zufallsverteilungen / Mischungen
# =============================================================================

def sample_mixture(
    number_of_samples: int,
    mixture_components: list[dict[str, Any]],
    max_value: float | None = None,
    unit_description: str = "generic",
) -> np.ndarray:
    """
    Erzeugt Stichproben aus einer Mischung von Verteilungen.
    Unterstützte Verteilungen pro Komponente:
      - lognormal, normal, beta, uniform
    Optional:
      - shift_minutes: wird nach dem Sampling addiert (z.B. Arrival Times).
    """
    if number_of_samples <= 0:
        return np.array([])

    component_weights = np.array([component["weight"] for component in mixture_components], dtype=float)
    component_weights = component_weights / component_weights.sum()

    chosen_component_indices = np.random.choice(
        len(mixture_components),
        size=number_of_samples,
        p=component_weights,
    )

    sampled_values = np.zeros(number_of_samples, dtype=float)

    for sample_index, component_index in enumerate(chosen_component_indices):
        component = mixture_components[component_index]
        dist_type = component.get("distribution", "lognormal").lower()

        if dist_type == "lognormal":
            mu_value = float(component["mu"])
            sigma_value = float(component["sigma"])
            value = np.random.lognormal(mean=mu_value, sigma=sigma_value)

        elif dist_type == "normal":
            mu_value = float(component["mu"])
            sigma_value = float(component["sigma"])
            value = np.random.normal(loc=mu_value, scale=sigma_value)

        elif dist_type == "beta":
            alpha = float(component["alpha"])
            beta_param = float(component["beta"])
            value = np.random.beta(a=alpha, b=beta_param)

        elif dist_type == "uniform":
            low = float(component["low"])
            high = float(component["high"])
            value = np.random.uniform(low, high)

        else:
            raise ValueError(f"Unbekannte Verteilung: {dist_type}")

        if "shift_minutes" in component and component["shift_minutes"] is not None:
            value = value * 60.0 + float(component["shift_minutes"])

        sampled_values[sample_index] = value

    if max_value is not None:
        sampled_values = np.minimum(sampled_values, max_value)

    return sampled_values


def realize_mixture_components(
    component_templates: list[dict[str, Any]],
    allow_shift: bool = False,
) -> list[dict[str, Any]]:
    """
    Nimmt Komponenten-Templates aus YAML und erzeugt konkrete Komponenten
    für die Mischung (inkl. Ziehen der Ranges mit sample_from_range).
    """
    realized: list[dict[str, Any]] = []

    for component_template in component_templates:
        dist_type = component_template.get("distribution", "lognormal")
        component: dict[str, Any] = {
            "distribution": dist_type,
            "weight": sample_from_range(component_template.get("weight", 1.0)),
        }

        if dist_type in ("lognormal", "normal"):
            component["mu"] = sample_from_range(component_template["mu"])
            component["sigma"] = sample_from_range(component_template["sigma"])
        elif dist_type == "beta":
            component["alpha"] = sample_from_range(component_template["alpha"])
            component["beta"] = sample_from_range(component_template["beta"])
        elif dist_type == "uniform":
            component["low"] = sample_from_range(component_template["low"])
            component["high"] = sample_from_range(component_template["high"])
        else:
            raise ValueError(f"Unbekannte Verteilung in YAML: {dist_type}")

        if allow_shift:
            component["shift_minutes"] = component_template.get("shift_minutes", None)

        realized.append(component)

    return realized


# =============================================================================
# 5) Fahrzeugwahl nach Standortgewichtung
# =============================================================================

def choose_vehicle_profile(
    vehicle_profiles: list[VehicleProfile],
    scenario: dict[str, Any],
) -> VehicleProfile:
    """
    Wählt ein Fahrzeugprofil aus der geladenen Flotte aus.
    Optional kann pro Standort in der YAML ein fleet_mix gesetzt werden.

    Falls fleet_mix fehlt oder ungültig ist, wird gleichverteilt gewählt.
    """
    fleet_mix = scenario.get("vehicles", {}).get("fleet_mix", None)

    if not fleet_mix:
        return np.random.choice(vehicle_profiles)

    selectable = [vp for vp in vehicle_profiles if vp.vehicle_class in fleet_mix]
    if not selectable:
        return np.random.choice(vehicle_profiles)

    weights = np.array([float(fleet_mix[vp.vehicle_class]) for vp in selectable], dtype=float)
    if weights.sum() <= 0.0:
        return np.random.choice(selectable)

    probs = weights / weights.sum()
    return np.random.choice(selectable, p=probs)


# =============================================================================
# 6) Session-Generierung (Ankunft, Parkdauer, SoC, Tages-Sessions)
# =============================================================================

def sample_arrival_times_for_day(
    scenario: dict,
    day_start_datetime: datetime,
    holiday_dates: set[date],
) -> list[datetime]:
    """
    Generiert alle Ankunftszeitpunkte für einen Tag als datetime-Objekte.
    """
    day_type = determine_day_type_with_holidays(day_start_datetime, holiday_dates)

    number_of_chargers = scenario["site"]["number_chargers"]
    expected_sessions_per_charger_range = scenario["site"]["expected_sessions_per_charger_per_day"]
    expected_sessions_per_charger = sample_from_range(expected_sessions_per_charger_range)

    weekday_weight_range = scenario["arrival_time_distribution"]["weekday_weight"][day_type]
    weekday_weight = sample_from_range(weekday_weight_range)

    number_of_sessions_today = int(number_of_chargers * expected_sessions_per_charger * weekday_weight)
    if number_of_sessions_today <= 0:
        return []

    templates = scenario["arrival_time_distribution"]["components_per_weekday"][day_type]
    mixture_components = realize_mixture_components(templates, allow_shift=True)
    if not mixture_components:
        return []

    sampled_minutes = sample_mixture(
        number_of_samples=number_of_sessions_today,
        mixture_components=mixture_components,
        max_value=None,
        unit_description="minutes",
    )

    sampled_minutes = np.maximum(sampled_minutes, 0.0)
    sampled_minutes = np.minimum(sampled_minutes, 24.0 * 60.0 - 1.0)

    arrivals = [day_start_datetime + timedelta(minutes=float(m)) for m in sampled_minutes]
    arrivals.sort()
    return arrivals


def sample_parking_durations(scenario: dict, number_of_sessions: int) -> np.ndarray:
    """
    Generiert Parkdauern in Minuten für die übergebene Anzahl an Sessions.
    """
    cfg = scenario["parking_duration_distribution"]
    max_minutes = cfg["max_duration_minutes"]
    min_minutes = cfg.get("min_duration_minutes", 10.0)

    templates = cfg["components"]
    mixture_components = realize_mixture_components(templates, allow_shift=False)

    durations = sample_mixture(
        number_of_samples=number_of_sessions,
        mixture_components=mixture_components,
        max_value=max_minutes,
        unit_description="minutes",
    )

    durations = np.clip(durations, min_minutes, max_minutes)
    return durations


def sample_soc_upon_arrival(scenario: dict, number_of_sessions: int) -> np.ndarray:
    """
    Generiert SoC-Werte (0..1) bei Ankunft für mehrere Ladesessions.
    """
    cfg = scenario["soc_at_arrival_distribution"]
    max_soc = cfg["max_soc"]

    templates = cfg["components"]
    mixture_components = realize_mixture_components(templates, allow_shift=False)

    soc_values = sample_mixture(
        number_of_samples=number_of_sessions,
        mixture_components=mixture_components,
        max_value=max_soc,
        unit_description="soc_fraction",
    )

    return np.maximum(soc_values, 0.0)


def build_charging_sessions_for_day(
    scenario: dict,
    day_start_datetime: datetime,
    vehicle_profiles: list[VehicleProfile],
    holiday_dates: set[date],
) -> list[dict[str, Any]]:
    """
    Erzeugt für einen Tag eine Liste von Ladesessions mit allen relevanten Parametern.
    Jetzt mit zufälliger Fahrzeugwahl und fahrzeugspezifischen Ladekurven.
    """
    arrivals = sample_arrival_times_for_day(scenario, day_start_datetime, holiday_dates)
    n = len(arrivals)
    if n == 0:
        return []

    parking_minutes = sample_parking_durations(scenario, n)
    soc_arrivals = sample_soc_upon_arrival(scenario, n)
    target_soc = scenario["vehicles"]["soc_target"]

    allow_cross_day = scenario["site"].get("allow_cross_day_charging", True)

    sessions: list[dict[str, Any]] = []

    for i in range(n):
        arrival_time = arrivals[i]
        raw_departure = arrival_time + timedelta(minutes=float(parking_minutes[i]))

        if allow_cross_day:
            departure_time = raw_departure
        else:
            end_of_day = day_start_datetime + timedelta(days=1)
            departure_time = min(raw_departure, end_of_day)

        soc_at_arrival = float(soc_arrivals[i])

        vehicle_profile = choose_vehicle_profile(vehicle_profiles, scenario)
        battery_capacity_kwh = float(vehicle_profile.battery_capacity_kwh)

        required_energy_kwh = max(target_soc - soc_at_arrival, 0.0) * battery_capacity_kwh
        if required_energy_kwh <= 0.0:
            continue

        max_vehicle_power_kw = float(vehicle_profile.power_grid_kw.max())

        sessions.append(
            {
                "arrival_time": arrival_time,
                "departure_time": departure_time,
                "soc_arrival": soc_at_arrival,
                "soc_target": target_soc,
                "battery_capacity_kwh": battery_capacity_kwh,
                "energy_required_kwh": required_energy_kwh,
                "delivered_energy_kwh": 0.0,
                "max_charging_power_kw": max_vehicle_power_kw,
                "vehicle_name": vehicle_profile.name,
                "vehicle_class": vehicle_profile.vehicle_class,
                "soc_grid": vehicle_profile.soc_grid,
                "power_grid_kw": vehicle_profile.power_grid_kw,
            }
        )

    return sessions


# =============================================================================
# 7) Lademanagement-Strategien (NEU: market / generation)
# =============================================================================

CSV_DT_FORMAT = "%d.%m.%Y %H:%M"


def read_strategy_series_from_csv_first_col_time(
    csv_path: str,
    value_col_1_based: int,
    delimiter: str = ";",
) -> dict[datetime, float]:
    """
    Liest ein externes Strategie-Signal (Preis oder Erzeugung) aus CSV:

    Annahmen (bewusst robust für unterschiedliche Header):
      - Zeitstempel steht IMMER in der 1. Spalte (1-basiert: 1)
      - Signalwert steht in der Spalte value_col_1_based (1-basiert: 1=erste Spalte)
      - Header-Bezeichnungen können variieren (werden automatisch übersprungen)

    Erwartetes Zeitformat der 1. Spalte:
      "01.12.2025 00:15"  (Format: %d.%m.%Y %H:%M)

    WICHTIG (1-basiert):
      - value_col_1_based=1 wäre die Zeitspalte (nicht sinnvoll für Signal)
      - typischerweise value_col_1_based=3 (3. Spalte)
    """
    if not isinstance(value_col_1_based, int) or value_col_1_based < 2:
        raise ValueError(
            "value_col_1_based muss eine ganze Zahl >= 2 sein "
            "(1 = Zeitspalte, 2..n = Signalspalte)."
        )

    data: dict[datetime, float] = {}

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=delimiter)

        for row in reader:
            if not row:
                continue
            if len(row) < value_col_1_based:
                continue

            t_raw = (row[0] or "").strip()  # 1. Spalte (Zeit) -> Index 0
            v_raw = (row[value_col_1_based - 1] or "").strip()  # 1-basiert -> Index

            if not t_raw or not v_raw or v_raw == "-":
                continue

            try:
                ts = datetime.strptime(t_raw, CSV_DT_FORMAT)
            except ValueError:
                # Header/sonstige Zeilen ohne Datum automatisch überspringen
                continue

            try:
                val = parse_number_de_or_en(v_raw)
            except ValueError:
                continue

            data[ts] = float(val)

    if not data:
        raise ValueError(f"Keine gültigen Datenzeilen im Strategie-CSV gefunden: {csv_path}")

    return data


def floor_to_resolution(dt: datetime, resolution_min: int) -> datetime:
    """Rundet auf den Start des aktuellen Zeitslots (z.B. 15-min) ab."""
    discard = dt.minute % resolution_min
    return dt.replace(minute=dt.minute - discard, second=0, microsecond=0)


def lookup_signal(strategy_map: dict[datetime, float], ts: datetime, resolution_min: int) -> float | None:
    """Findet den Signalwert für den (abgerundeten) Zeitslot."""
    return strategy_map.get(floor_to_resolution(ts, resolution_min), None)


def compute_multiplier_from_signal(
    signal_value: float,
    q_low: float,
    q_high: float,
    min_mult: float,
    max_mult: float,
    mode: str,      # "market" oder "generation"
    gamma: float = 1.0,
) -> float:
    """
    Mappt ein Signal (Preis oder Erzeugung) auf einen Multiplikator für die
    maximal verfügbare Standortleistung.

    - market: niedriger Preis => hoher Multiplikator (invertiert)
    - generation: hohe Erzeugung => hoher Multiplikator
    """
    if q_high <= q_low:
        return max_mult

    x = (signal_value - q_low) / (q_high - q_low)
    x = float(np.clip(x, 0.0, 1.0))

    mode = (mode or "immediate").lower()
    if mode == "market":
        x = 1.0 - x
    elif mode == "generation":
        pass
    else:
        return 1.0

    x = x ** float(gamma)
    return float(min_mult + (max_mult - min_mult) * x)


# =============================================================================
# 8) Hauptsimulation / Orchestrierung der Lastgangberechnung
# =============================================================================

def simulate_load_profile(scenario: dict, start_datetime: datetime | None = None):
    """
    Führt die Lastgang-Simulation durch.

    Unterstützte Lademanagementstrategien:
      - immediate  : lädt (unter Restriktionen) sofort nach Ankunft
      - market     : skaliert Standortleistung hoch, wenn Preis niedrig ist
      - generation : skaliert Standortleistung hoch, wenn Erzeugung hoch ist

    Hinweis:
      Das Strategie-Signal kommt aus einer CSV (z.B. Day-Ahead), dabei gilt:
        - Zeitstempel steht in der 1. Spalte
        - Signalwert-Spalte wird über YAML 'site.strategy_value_col' gewählt (1-basiert: 3 = 3. Spalte)
    """

    # -------------------------------------------------------------------
    # 1) Zeitindex erzeugen (Simulationszeitachse)
    # -------------------------------------------------------------------
    time_index = create_time_index(scenario, start_datetime)

    # -------------------------------------------------------------------
    # 2) Feiertage 1x ableiten (Bundesland, Jahr(e) aus Start + Horizont)
    # -------------------------------------------------------------------
    if time_index:
        simulation_start_datetime = time_index[0]
    else:
        if start_datetime is not None:
            simulation_start_datetime = start_datetime
        elif "start_datetime" in scenario:
            simulation_start_datetime = datetime.fromisoformat(scenario["start_datetime"])
        else:
            simulation_start_datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    holiday_dates = parse_holiday_dates_from_scenario(
        scenario=scenario,
        simulation_start_datetime=simulation_start_datetime,
    )

    # -------------------------------------------------------------------
    # 3) Fahrzeugflotte inkl. Ladekurven aus CSV laden (Pfad relativ zur YAML)
    # -------------------------------------------------------------------
    vehicle_csv_path = resolve_path_relative_to_scenario(scenario, scenario["vehicles"]["vehicle_curve_csv"])
    vehicle_profiles = load_vehicle_profiles_from_csv(vehicle_csv_path)

    # -------------------------------------------------------------------
    # 4) Strategie-Initialisierung (market / generation / immediate)
    # -------------------------------------------------------------------
    site_cfg = scenario.get("site", {}) or {}
    charging_strategy = (site_cfg.get("charging_strategy") or "immediate").lower()

    # Strategie-Defaults (bewusst im Code, damit YAML schlank bleibt)
    STRATEGY_RESOLUTION_MIN = 15
    STRATEGY_MIN_MULTIPLIER = 0.20
    STRATEGY_MAX_MULTIPLIER = 1.00
    STRATEGY_Q_LOW = 0.20
    STRATEGY_Q_HIGH = 0.80
    STRATEGY_GAMMA = 1.0

    strategy_map: dict[datetime, float] | None = None
    q_low_val: float | None = None
    q_high_val: float | None = None

    # -------------------------------------------------------------------
    # 4a) Nur bei market/generation: Signal-CSV laden und Quantile bestimmen
    # -------------------------------------------------------------------
    if charging_strategy in ("market", "generation"):
        strategy_csv = site_cfg.get("strategy_csv", None)
        if not strategy_csv:
            raise ValueError(
                "Für charging_strategy 'market' oder 'generation' muss in der YAML 'site.strategy_csv' gesetzt sein."
            )

        # Spaltennummer kommt aus YAML (1-basiert; 3 = dritte Spalte)
        col_1_based = site_cfg.get("strategy_value_col", None)
        if col_1_based is None or not isinstance(col_1_based, int) or col_1_based < 2:
            raise ValueError(
                "Bitte 'site.strategy_value_col' in der YAML als ganze Zahl >= 2 setzen "
                "(1-basiert: 1=Zeitspalte, 3=dritte Spalte=Signal)."
            )

        strategy_csv_path = resolve_path_relative_to_scenario(scenario, strategy_csv)

        # Strategie-Signal einlesen (Zeitstempel immer in Spalte 1)
        strategy_map = read_strategy_series_from_csv_first_col_time(
            csv_path=strategy_csv_path,
            value_col_1_based=int(col_1_based),   # <<< 1-basiert bleibt 1-basiert
            delimiter=";",
        )

        values = np.array(list(strategy_map.values()), dtype=float)
        q_low_val = float(np.quantile(values, STRATEGY_Q_LOW))
        q_high_val = float(np.quantile(values, STRATEGY_Q_HIGH))

        # -------------------------------------------------------------------
        # 4b) Plausibilitätsprüfung: Zeitliche Abdeckung Strategie-CSV
        # -------------------------------------------------------------------
        if time_index and strategy_map:
            csv_start = min(strategy_map.keys())
            csv_end = max(strategy_map.keys())

            sim_start = time_index[0]
            sim_end = time_index[-1]

            if csv_end < sim_start or csv_start > sim_end:
                warnings.warn(
                    "⚠️ Strategie-CSV liegt zeitlich vollständig außerhalb des "
                    "Simulationszeitraums.\n"
                    "→ charging_strategy wirkt NICHT (Fallback ≈ immediate).\n"
                    f"CSV-Zeitraum: {csv_start} bis {csv_end}\n"
                    f"Simulation: {sim_start} bis {sim_end}",
                    UserWarning,
                )
            elif csv_start > sim_start or csv_end < sim_end:
                warnings.warn(
                    "⚠️ Strategie-CSV deckt den Simulationszeitraum nur TEILWEISE ab.\n"
                    "→ charging_strategy wirkt nur in überlappenden Zeitbereichen.\n"
                    f"CSV-Zeitraum: {csv_start} bis {csv_end}\n"
                    f"Simulation: {sim_start} bis {sim_end}",
                    UserWarning,
                )

    # -------------------------------------------------------------------
    # 5) Initialisierung: Simulationsparameter & Ergebniscontainer
    # -------------------------------------------------------------------
    time_resolution_min = scenario["time_resolution_min"]
    time_step_hours = time_resolution_min / 60.0

    n_steps = len(time_index)
    load_profile_kw = np.zeros(n_steps, dtype=float)

    grid_limit_p_avb_kw = scenario["site"]["grid_limit_p_avb_kw"]
    rated_power_kw = scenario["site"]["rated_power_kw"]
    number_of_chargers = scenario["site"]["number_chargers"]
    charger_efficiency = scenario["site"]["charger_efficiency"]

    charging_count_series: list[int] = []
    all_charging_sessions: list[dict[str, Any]] = []

    # -------------------------------------------------------------------
    # 6) Tagesweise Ladesessions für den gesamten Horizont erzeugen
    # -------------------------------------------------------------------
    if time_index:
        first_day_start = time_index[0].replace(hour=0, minute=0, second=0, microsecond=0)
        horizon_days = int(scenario["simulation_horizon_days"])

        for day_offset in range(horizon_days):
            day_start = first_day_start + timedelta(days=day_offset)
            all_charging_sessions.extend(
                build_charging_sessions_for_day(
                    scenario=scenario,
                    day_start_datetime=day_start,
                    vehicle_profiles=vehicle_profiles,
                    holiday_dates=holiday_dates,
                )
            )

    # -------------------------------------------------------------------
    # 7) Zeitschrittweise Simulation: Sessions auswählen & Leistung/Energie zuweisen
    # -------------------------------------------------------------------
    for i, ts in enumerate(time_index):

        present_sessions = [
            s for s in all_charging_sessions
            if s["arrival_time"] <= ts < s["departure_time"]
            and s["energy_required_kwh"] > 0.0
        ]

        if not present_sessions:
            load_profile_kw[i] = 0.0
            charging_count_series.append(0)
            continue

        present_sessions.sort(key=lambda s: (s["departure_time"], s["arrival_time"]))
        charging_sessions = present_sessions[:number_of_chargers]
        charging_count_series.append(len(charging_sessions))

        # -------------------------------------------------------------------
        # 7a) Standortleistung (immediate) + optionaler Strategie-Multiplikator
        # -------------------------------------------------------------------
        base_max_site_power_kw = min(
            grid_limit_p_avb_kw,
            len(charging_sessions) * rated_power_kw,
        )

        multiplier = 1.0
        if strategy_map is not None and q_low_val is not None and q_high_val is not None:
            sig = lookup_signal(strategy_map, ts, STRATEGY_RESOLUTION_MIN)
            if sig is not None:
                multiplier = compute_multiplier_from_signal(
                    signal_value=sig,
                    q_low=q_low_val,
                    q_high=q_high_val,
                    min_mult=STRATEGY_MIN_MULTIPLIER,
                    max_mult=STRATEGY_MAX_MULTIPLIER,
                    mode=charging_strategy,
                    gamma=STRATEGY_GAMMA,
                )

        max_site_power_kw = base_max_site_power_kw * multiplier
        available_power_per_session_kw = max_site_power_kw / len(charging_sessions)

        total_power_kw = 0.0

        # -------------------------------------------------------------------
        # 7b) Pro Session: fahrzeugspezifische Leistung + Energiebedarf beachten
        # -------------------------------------------------------------------
        for s in charging_sessions:
            vehicle_power_limit_kw = vehicle_power_at_soc(s)

            allowed_power_kw = min(
                available_power_per_session_kw,
                rated_power_kw,
                vehicle_power_limit_kw,
                s["max_charging_power_kw"],
            )

            possible_energy_kwh = allowed_power_kw * time_step_hours * charger_efficiency
            energy_needed = s["energy_required_kwh"]

            if possible_energy_kwh >= energy_needed:
                energy_delivered = energy_needed
                s["energy_required_kwh"] = 0.0
                actual_power_kw = energy_delivered / (time_step_hours * charger_efficiency)
            else:
                energy_delivered = possible_energy_kwh
                s["energy_required_kwh"] -= possible_energy_kwh
                actual_power_kw = allowed_power_kw

            s["delivered_energy_kwh"] += energy_delivered
            total_power_kw += actual_power_kw

        load_profile_kw[i] = total_power_kw

    # -------------------------------------------------------------------
    # 8) Strategie-Status (NEU): ACTIVE / PARTIAL / INACTIVE / IMMEDIATE
    # -------------------------------------------------------------------
    strategy_status = "IMMEDIATE"
    if charging_strategy in ("market", "generation") and strategy_map and time_index:
        csv_start = min(strategy_map.keys())
        csv_end = max(strategy_map.keys())
        sim_start = time_index[0]
        sim_end = time_index[-1]

        if csv_end < sim_start or csv_start > sim_end:
            strategy_status = "INACTIVE"
        elif csv_start > sim_start or csv_end < sim_end:
            strategy_status = "PARTIAL"
        else:
            strategy_status = "ACTIVE"

    # -------------------------------------------------------------------
    # 9) Rückgabe der Simulationsergebnisse (erweitert)
    # -------------------------------------------------------------------
    return (
        time_index,
        load_profile_kw,
        all_charging_sessions,
        charging_count_series,
        holiday_dates,
        charging_strategy,
        strategy_status,
    )
