import yaml
import numpy as np
from pathlib import Path
import csv  # CSV-Parsing (Fahrzeugkurven + Markt/Erzeugungssignal)
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Any


# =============================================================================
# 0) Projekt-/Pfad-Utilities
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
# 0b) Robustes Zahlen-Parsing
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
    return float(s)


# =============================================================================
# 0c) HTML-Statusausgabe (optional im Notebook)
# =============================================================================

def show_strategy_status_html(strategy: str, status: str) -> None:
    """
    Zeigt im Jupyter Notebook eine farbige Statuszeile (HTML).
    Fallback: normaler print, wenn IPython nicht verfügbar ist.

    status: "ACTIVE" | "INACTIVE" | "IMMEDIATE"
    """
    status = (status or "IMMEDIATE").upper()
    strategy = (strategy or "immediate").upper()

    color_map = {
        "ACTIVE": "green",
        "INACTIVE": "red",
        "IMMEDIATE": "gray",
    }
    emoji_map = {
        "ACTIVE": "🟢",
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
    vehicle_class: str
    soc_grid: np.ndarray
    power_grid_kw: np.ndarray


def load_vehicle_profiles_from_csv(path: str) -> list[VehicleProfile]:
    """
    CSV mit Fahrzeugen und SoC-abhängigen Ladekurven laden.

    Struktur (Delimiter ';'):
      Zeile 1: Hersteller (ignoriert)
      Zeile 2: Modellnamen
      Zeile 3: Fahrzeugklasse
      Zeile 4: Kapazität (kWh)
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
    Fahrzeugspezifische maximale Ladeleistung (kW) für aktuellen SoC
    (Interpolation aus SoC-/Power-Stützstellen).
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
# 2) Szenario laden (YAML) + Pfadkontext
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
    Verarbeitet YAML-Werte als Skalar oder als Liste [min, max] bzw. [value].
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
    Feiertage aus YAML als Set[date].

    Unterstützt:
      A) manuell: holidays.dates
      B) automatisch: holidays.country + holidays.subdivision
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
    return [simulation_start_datetime + step_index * time_step_delta for step_index in range(number_of_time_steps)]


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
    Unterstützte Verteilungen pro Komponente: lognormal, normal, beta, uniform.
    Optional: shift_minutes (bei Arrival Times).
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

    Falls fleet_mix fehlt/ungültig: gleichverteilt.
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

    return np.clip(durations, min_minutes, max_minutes)


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

                # -----------------------------------------------------------------
                # OPTIMIERUNG: Slot-basierte Präferenzen (werden nach Sessionbau befüllt)
                # -----------------------------------------------------------------
                "preferred_slot_indices": [],  # Liste von Indizes in time_index, sortiert nach Signalqualität
                "preferred_ptr": 0,            # Pointer auf nächsten bevorzugten Slot (>= aktuellem Schritt)
            }
        )

    return sessions


# =============================================================================
# 7) Strategie-Signal aus CSV (Market / Generation)
# =============================================================================

CSV_DT_FORMATS = ("%d.%m.%Y %H:%M", "%d.%m.%y %H:%M")


def read_strategy_series_from_csv_first_col_time(
    csv_path: str,
    value_col_1_based: int,
    delimiter: str = ";",
) -> dict[datetime, float]:
    """
    Liest ein externes Strategie-Signal (Preis oder Erzeugung) aus CSV:

    Annahmen:
      - Zeitstempel steht IMMER in der 1. Spalte
      - Signalwert steht in Spalte value_col_1_based (1-basiert, >=2)
      - Header-Zeilen werden übersprungen
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

            t_raw = (row[0] or "").strip()
            v_raw = (row[value_col_1_based - 1] or "").strip()

            if not t_raw or not v_raw or v_raw == "-":
                continue

            # -------------------------------
            # Zeitstempel robust parsen (2- oder 4-stelliges Jahr)
            # -------------------------------
            ts = None
            for fmt in CSV_DT_FORMATS:
                try:
                    ts = datetime.strptime(t_raw, fmt)
                    break
                except ValueError:
                    pass

            if ts is None:
                # Header/sonstige Zeilen ohne Datum automatisch überspringen
                continue

            # -------------------------------
            # Wert robust parsen
            # -------------------------------
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


# =============================================================================
# 7b) OPTIMIERUNG: Harte Validierung (Abbruch wenn CSV != Simulationshorizont)
# =============================================================================

def assert_strategy_csv_covers_simulation(
    strategy_map: dict[datetime, float],
    time_index: list[datetime],
    strategy_resolution_min: int,
    charging_strategy: str,
    strategy_csv_path: str,
) -> None:
    """
    Bricht ab, wenn die Strategie-CSV den Simulationszeitraum NICHT vollständig abdeckt
    oder wenn innerhalb des Zeitraums Zeitstempel fehlen.

    Ziel: Keine langen Rechenzeiten durch stillen Fallback auf "immediate".
    """
    if not time_index:
        raise ValueError("Simulationszeitachse ist leer – kann Strategie-CSV nicht prüfen.")

    # Erwartete Zeitpunkte auf Strategie-Auflösung (gefloort)
    expected_ts = [floor_to_resolution(t, strategy_resolution_min) for t in time_index]
    expected_set = set(expected_ts)

    csv_start = min(strategy_map.keys())
    csv_end = max(strategy_map.keys())

    sim_start = min(expected_set)
    sim_end = max(expected_set)

    # 1) Zeitraum muss vollständig enthalten sein
    if csv_start > sim_start or csv_end < sim_end:
        # Wie viele volle Tage sind ab sim_start durch die CSV abgedeckt?
        covered_minutes = int((csv_end - sim_start).total_seconds() / 60)
        covered_days = max(0.0, covered_minutes / (24 * 60))

        # Vorschlag: horizon_days auf floor(covered_days) setzen
        # Kalendertage zählen (wenn CSV bis Tagesende reicht, zählt der Tag voll)
        last_day = csv_end.date()
        first_day = sim_start.date()
        suggested_days = (last_day - first_day).days + 1


        raise ValueError(
            "❌ Abbruch: Strategie-CSV deckt den Simulationszeitraum nicht vollständig ab.\n"
            f"Strategie: {charging_strategy}\n"
            f"CSV: {strategy_csv_path}\n"
            f"CSV-Zeitraum: {csv_start} bis {csv_end}\n"
            f"Simulation:   {sim_start} bis {sim_end}\n"
            f"→ CSV deckt ab Start nur ca. {covered_days:.2f} Tage ab.\n"
            f"✅ Vorschlag: simulation_horizon_days auf {suggested_days} setzen (oder CSV erweitern).\n"
        )

    # 2) Lückencheck: jeder benötigte Timestamp muss vorhanden sein
    missing = sorted([t for t in expected_set if t not in strategy_map])
    if missing:
        preview = "\n".join([f"- {t}" for t in missing[:10]])
        raise ValueError(
            "❌ Abbruch: Strategie-CSV hat fehlende Zeitstempel innerhalb des Simulationszeitraums.\n"
            f"Strategie: {charging_strategy}\n"
            f"CSV: {strategy_csv_path}\n"
            f"Fehlende Timestamps: {len(missing)} (erste 10):\n{preview}\n"
            "Bitte CSV-Auflösung/Zeitraum prüfen (z.B. 15-min Raster, Sommer-/Winterzeit, fehlende Zeilen)."
        )


# =============================================================================
# 8) OPTIMIERUNG: Slot-basierte Heuristik (Best Slot innerhalb der Standzeit)
# =============================================================================

def _slack_minutes_for_session(
    s: dict[str, Any],
    ts: datetime,
    rated_power_kw: float,
    charger_efficiency: float,
) -> float:
    """
    Slack-Minuten (Puffer) der Session am Zeitpunkt ts.

    slack = verbleibende Standzeit - benötigte Ladezeit (bei max. physikalischer Session-Leistung)

    Hinweis: Netz-/NAP-Engpässe sind hier NICHT enthalten.
    """
    remaining_seconds = (s["departure_time"] - ts).total_seconds()
    if remaining_seconds <= 0:
        return -1e9

    remaining_hours = remaining_seconds / 3600.0

    vehicle_limit_kw = vehicle_power_at_soc(s)
    p_max_kw = max(0.0, min(rated_power_kw, vehicle_limit_kw, float(s["max_charging_power_kw"])))

    if p_max_kw <= 1e-9:
        return -1e9

    e_rest = float(s["energy_required_kwh"])
    needed_hours = e_rest / (p_max_kw * charger_efficiency)

    return (remaining_hours - needed_hours) * 60.0


def _build_preferred_slots_for_all_sessions(
    all_sessions: list[dict[str, Any]],
    time_index: list[datetime],
    time_to_idx: dict[datetime, int],
    charging_strategy: str,
    strategy_map: dict[datetime, float] | None,
    strategy_resolution_min: int,
) -> None:
    """
    Erzeugt pro Session eine Liste bevorzugter Zeitslots (Indices in time_index),
    sortiert nach:
      - market: günstigster Preis zuerst (aufsteigend)
      - generation: höchste Erzeugung zuerst (absteigend)

    Fehlende Signalwerte (None) werden als "sehr schlecht" behandelt und ans Ende sortiert.
    """
    if charging_strategy not in ("market", "generation"):
        return
    if not strategy_map or not time_index:
        return

    bad_market = 1e30          # fehlender Preis -> extrem schlecht
    bad_generation_score = 1e30  # wir sortieren generation über -signal, fehlend -> schlecht

    for s in all_sessions:
        if float(s.get("energy_required_kwh", 0.0)) <= 0.0:
            continue

        # Session-Zeitraum auf Strategie-Auflösung runden
        a = floor_to_resolution(s["arrival_time"], strategy_resolution_min)
        d = floor_to_resolution(s["departure_time"], strategy_resolution_min)

        # Robust: auf vorhandene time_index-Slots ziehen
        while a not in time_to_idx and a < time_index[-1]:
            a += timedelta(minutes=strategy_resolution_min)
        while d not in time_to_idx and d > time_index[0]:
            d -= timedelta(minutes=strategy_resolution_min)

        if a not in time_to_idx or d not in time_to_idx:
            s["preferred_slot_indices"] = []
            s["preferred_ptr"] = 0
            continue

        start_idx = time_to_idx[a]
        end_idx = time_to_idx[d]
        if end_idx <= start_idx:
            s["preferred_slot_indices"] = []
            s["preferred_ptr"] = 0
            continue

        idxs = list(range(start_idx, end_idx))

        scored: list[tuple[float, int]] = []
        for idx in idxs:
            t = time_index[idx]
            sig = lookup_signal(strategy_map, t, strategy_resolution_min)

            if sig is None:
                # Missing signal -> ans Ende
                score = bad_market if charging_strategy == "market" else bad_generation_score
            else:
                if charging_strategy == "market":
                    # niedriger Preis ist besser
                    score = float(sig)
                else:
                    # höher ist besser -> wir sortieren aufsteigend nach (-sig)
                    score = -float(sig)

            scored.append((score, idx))

        scored.sort(key=lambda x: x[0])
        s["preferred_slot_indices"] = [idx for _, idx in scored]
        s["preferred_ptr"] = 0


# =============================================================================
# 9) Hauptsimulation / Orchestrierung der Lastgangberechnung
# =============================================================================

def simulate_load_profile(scenario: dict, start_datetime: datetime | None = None):
    """
    Führt die Lastgang-Simulation durch.

    Unterstützte Lademanagementstrategien:
      - immediate  : lädt (unter Restriktionen) sofort nach Ankunft
      - market     : OPTIMIERUNG (slotbasiert): lädt bevorzugt zu günstigsten Zeitpunkten innerhalb der Standzeit
      - generation : OPTIMIERUNG (slotbasiert): lädt bevorzugt zu erzeugungsstärksten Zeitpunkten innerhalb der Standzeit

    WICHTIG:
      - Für market/generation wird die CSV-Abdeckung strikt geprüft:
        passt die CSV nicht EXAKT zum Simulationshorizont -> ABRUCH (keine Fallback-Simulation).
      - strategy_status nur: IMMEDIATE, ACTIVE, INACTIVE
    """

    # -------------------------------------------------------------------
    # 1) Zeitindex erzeugen (Simulationszeitachse)
    # -------------------------------------------------------------------
    time_index = create_time_index(scenario, start_datetime)

    # -------------------------------------------------------------------
    # 2) Feiertage 1x ableiten
    # -------------------------------------------------------------------
    simulation_start_datetime = time_index[0] if time_index else (
        start_datetime if start_datetime is not None else
        datetime.fromisoformat(scenario["start_datetime"]) if "start_datetime" in scenario else
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    )

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

    # CSV/Signal-Auflösung (typisch 15min)
    STRATEGY_RESOLUTION_MIN = 15

    strategy_map: dict[datetime, float] | None = None
    strategy_csv_path: str | None = None

    # -------------------------------------------------------------------
    # 4a) Nur bei market/generation: Signal-CSV laden + HARTE VALIDIERUNG
    # -------------------------------------------------------------------
    if charging_strategy in ("market", "generation"):
        strategy_csv = site_cfg.get("strategy_csv", None)
        if not strategy_csv:
            raise ValueError(
                "❌ Abbruch: Für charging_strategy 'market' oder 'generation' muss in der YAML 'site.strategy_csv' gesetzt sein."
            )

        col_1_based = site_cfg.get("strategy_value_col", None)
        if col_1_based is None or not isinstance(col_1_based, int) or col_1_based < 2:
            raise ValueError(
                "❌ Abbruch: Bitte 'site.strategy_value_col' in der YAML als ganze Zahl >= 2 setzen "
                "(1-basiert: 1=Zeitspalte, 3=dritte Spalte=Signal)."
            )

        strategy_csv_path = resolve_path_relative_to_scenario(scenario, strategy_csv)

        strategy_map = read_strategy_series_from_csv_first_col_time(
            csv_path=strategy_csv_path,
            value_col_1_based=int(col_1_based),
            delimiter=";",
        )

        # OPTIMIERUNG: Abbruch, wenn CSV nicht exakt passt (kein PARTIAL/kein Fallback)
        assert_strategy_csv_covers_simulation(
            strategy_map=strategy_map,
            time_index=time_index,
            strategy_resolution_min=STRATEGY_RESOLUTION_MIN,
            charging_strategy=charging_strategy,
            strategy_csv_path=strategy_csv_path,
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

    # OPTIMIERUNG: Notfall-Puffer (Minuten) – lädt unabhängig vom Slot, wenn kritisch
    # (Retail/Highway mit immediate können strategy_params leer lassen; wird dann ignoriert)
    strategy_params = site_cfg.get("strategy_params", {}) or {}
    emergency_slack_minutes = float(strategy_params.get("emergency_slack_minutes", 60.0))

    charging_count_series: list[int] = []
    all_charging_sessions: list[dict[str, Any]] = []

    # -------------------------------------------------------------------
    # 6) Tagesweise Ladesessions erzeugen
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
    # OPTIMIERUNG: Präferenzlisten (best time slots) je Session vorberechnen
    # -------------------------------------------------------------------
    time_to_idx = {t: idx for idx, t in enumerate(time_index)} if time_index else {}
    _build_preferred_slots_for_all_sessions(
        all_sessions=all_charging_sessions,
        time_index=time_index,
        time_to_idx=time_to_idx,
        charging_strategy=charging_strategy,
        strategy_map=strategy_map,
        strategy_resolution_min=STRATEGY_RESOLUTION_MIN,
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

        # ---------------------------------------------------------------
        # OPTIMIERUNG: Auswahl der zu ladenden Sessions
        #   - immediate: Earliest departure first
        #   - market/generation: lade nur in bevorzugten Slots, außer "emergency slack"
        # ---------------------------------------------------------------
        charging_sessions: list[dict[str, Any]] = []

        if charging_strategy == "immediate":
            # Immediate bleibt wie gehabt
            present_sessions.sort(key=lambda s: (s["departure_time"], s["arrival_time"]))
            charging_sessions = present_sessions[:number_of_chargers]

        else:
            # market/generation (strategy_map ist hier garantiert vorhanden, sonst wäre abgebrochen)
            emergency: list[dict[str, Any]] = []
            slot_candidates: list[dict[str, Any]] = []

            for s in present_sessions:
                slack_m = _slack_minutes_for_session(
                    s=s,
                    ts=ts,
                    rated_power_kw=rated_power_kw,
                    charger_efficiency=charger_efficiency,
                )
                s["_slack_minutes"] = slack_m  # nur intern

                # Notfall: unabhängig vom Slot-Wunsch laden
                if slack_m <= emergency_slack_minutes:
                    emergency.append(s)
                    continue

                # Slot-Wunsch prüfen: (nächster bevorzugter Slot == current i)
                pref = s.get("preferred_slot_indices", [])
                ptr = int(s.get("preferred_ptr", 0))

                while ptr < len(pref) and pref[ptr] < i:
                    ptr += 1
                s["preferred_ptr"] = ptr

                if ptr < len(pref) and pref[ptr] == i:
                    slot_candidates.append(s)

            # Kandidaten: emergency zuerst, dann Slot-Kandidaten
            seen = set()
            candidates: list[dict[str, Any]] = []

            for s in emergency:
                sid = id(s)
                if sid not in seen:
                    candidates.append(s)
                    seen.add(sid)

            for s in slot_candidates:
                sid = id(s)
                if sid not in seen:
                    candidates.append(s)
                    seen.add(sid)

            if candidates:
                # Kritischste zuerst: kleiner Slack, dann frühere Abfahrt
                candidates.sort(key=lambda s: (s.get("_slack_minutes", 1e9), s["departure_time"], s["arrival_time"]))
                charging_sessions = candidates[:number_of_chargers]
            else:
                charging_sessions = []

        charging_count_series.append(len(charging_sessions))

        if not charging_sessions:
            load_profile_kw[i] = 0.0
            continue

        # ---------------------------------------------------------------
        # 7a) Standortleistung / Limits (NAP, CP, Anzahl Charger)
        # ---------------------------------------------------------------
        max_site_power_kw = min(
            grid_limit_p_avb_kw,
            len(charging_sessions) * rated_power_kw,
        )
        available_power_per_session_kw = max_site_power_kw / len(charging_sessions)

        total_power_kw = 0.0

        # ---------------------------------------------------------------
        # 7b) Pro Session: fahrzeugspezifische Leistung + Energiebedarf beachten
        # ---------------------------------------------------------------
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
    # 8) Strategie-Status: ACTIVE / INACTIVE / IMMEDIATE (kein PARTIAL)
    # -------------------------------------------------------------------
    if charging_strategy == "immediate":
        strategy_status = "IMMEDIATE"
    elif charging_strategy in ("market", "generation"):
        # Wenn wir bis hier kommen, war die CSV-Abdeckung korrekt (sonst Abbruch)
        strategy_status = "ACTIVE" if strategy_map else "INACTIVE"
    else:
        # Unbekannte Strategie -> defensiv abbrechen (anstatt stiller Fallback)
        raise ValueError(f"❌ Abbruch: Unbekannte charging_strategy='{charging_strategy}'")

    # -------------------------------------------------------------------
    # 9) Rückgabe der Simulationsergebnisse
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
