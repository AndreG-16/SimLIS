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
    Ermittelt die fahrzeugspezifische maximale Ladeleistung (kW) für den aktuellen SoC.

    Hintergrund:
      - Viele Fahrzeuge reduzieren die Ladeleistung bei hohem SoC (Tapering).
      - Daher ist die maximal mögliche Ladeleistung nicht konstant, sondern SoC-abhängig.

    Output:
      - Maximale Ladeleistung in kW, die das Fahrzeug im aktuellen Zeitschritt physikalisch zulässt.
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
    """Verarbeitet YAML-Werte als Skalar oder als Liste [min, max] bzw. [value]."""
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
    Liest Feiertage aus YAML als Set[date].

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
                "Für automatische Feiertage wird das Paket 'holidays' benötigt: pip install holidays"
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
    Ermittelt den Tagtyp inkl. Feiertage:
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
    """Erzeugt eine Liste von Zeitstempeln über den im Szenario definierten Simulationshorizont."""
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
            # Arrival Times: mu wird als Stunden interpretiert (z.B. 8.5h),
            # danach umgerechnet auf Minuten (mu*60 + shift_minutes).
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
    """Generiert alle Ankunftszeitpunkte für einen Tag als datetime-Objekte."""
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
    """Generiert Parkdauern in Minuten für die übergebene Anzahl an Sessions."""
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
    """Generiert SoC-Werte (0..1) bei Ankunft für mehrere Ladesessions."""
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
    Erzeugt für einen Tag eine Liste von Ladesessions (Fahrzeug-Ankunft bis Abfahrt).

    Jede Session beschreibt ein Fahrzeug, das für eine gewisse Zeit am Standort steht
    und innerhalb dieser Standzeit eine Energiemenge nachladen möchte.

    Wichtiger Hinweis:
      - In dieser Funktion wird noch nicht entschieden, wann genau geladen wird.
      - Die zeitliche Entscheidung erfolgt später im Zeitschrittloop der Simulation.
      - Für market/generation werden später pro Session Slot-Präferenzen berechnet.
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

                # Restenergie, die bis zur Abfahrt geladen werden soll; wird später schrittweise reduziert.
                "energy_required_kwh": required_energy_kwh,

                # Bereits geladene Energie; wird im Simulationsloop schrittweise erhöht.
                "delivered_energy_kwh": 0.0,

                # Physikalische/Hardware-Grenzen
                "max_charging_power_kw": max_vehicle_power_kw,
                "vehicle_name": vehicle_profile.name,
                "vehicle_class": vehicle_profile.vehicle_class,
                "soc_grid": vehicle_profile.soc_grid,
                "power_grid_kw": vehicle_profile.power_grid_kw,

                # Slot-basierte Präferenzen (werden nach Sessionbau befüllt)
                "preferred_slot_indices": [],
                "preferred_ptr": 0,
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
    Liest ein externes Strategie-Signal aus einer CSV-Datei (Preis oder Erzeugung).

    Annahmen:
      - Zeitstempel steht in der 1. Spalte
      - Signalwert steht in der Spalte value_col_1_based (1-basiert, >=2)
      - Header-Zeilen werden automatisch übersprungen
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

            ts = None
            for fmt in CSV_DT_FORMATS:
                try:
                    ts = datetime.strptime(t_raw, fmt)
                    break
                except ValueError:
                    pass

            if ts is None:
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
    """Rundet einen Timestamp auf den Start des aktuellen Zeitslots (z.B. 15-min) ab."""
    discard = dt.minute % resolution_min
    return dt.replace(minute=dt.minute - discard, second=0, microsecond=0)


def lookup_signal(strategy_map: dict[datetime, float], ts: datetime, resolution_min: int) -> float | None:
    """Findet den Signalwert für den (abgerundeten) Zeitslot."""
    return strategy_map.get(floor_to_resolution(ts, resolution_min), None)


# =============================================================================
# 7b) Harte Validierung (Abbruch wenn CSV != Simulationshorizont)
# =============================================================================

def assert_strategy_csv_covers_simulation(
    strategy_map: dict[datetime, float],
    time_index: list[datetime],
    strategy_resolution_min: int,
    charging_strategy: str,
    strategy_csv_path: str,
) -> None:
    """
    Bricht ab, wenn die Strategie-CSV den Simulationszeitraum nicht vollständig abdeckt
    oder wenn innerhalb des Zeitraums Zeitstempel fehlen.

    Ziel:
      - Kein stiller Fallback auf "immediate"
      - Keine langen Rechenzeiten bei falschen CSV-Zeiträumen
    """
    if not time_index:
        raise ValueError("Simulationszeitachse ist leer – Strategie-CSV kann nicht geprüft werden.")

    expected_ts = [floor_to_resolution(t, strategy_resolution_min) for t in time_index]
    expected_set = set(expected_ts)

    csv_start = min(strategy_map.keys())
    csv_end = max(strategy_map.keys())

    sim_start = min(expected_set)
    sim_end = max(expected_set)

    if csv_start > sim_start or csv_end < sim_end:
        covered_minutes = int((csv_end - sim_start).total_seconds() / 60)
        covered_days = max(0.0, covered_minutes / (24 * 60))

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

    missing = sorted([t for t in expected_set if t not in strategy_map])
    if missing:
        preview = "\n".join([f"- {t}" for t in missing[:10]])
        raise ValueError(
            "❌ Abbruch: Strategie-CSV hat fehlende Zeitstempel innerhalb des Simulationszeitraums.\n"
            f"Strategie: {charging_strategy}\n"
            f"CSV: {strategy_csv_path}\n"
            f"Fehlende Timestamps: {len(missing)} (erste 10):\n{preview}\n"
            "CSV-Auflösung/Zeitraum prüfen (z.B. 15-min Raster, Sommer-/Winterzeit, fehlende Zeilen)."
        )


# =============================================================================
# 7c) Normalisierung der CSV-Werte (Market: €/kWh, Generation: kW)
# =============================================================================

def convert_strategy_value_to_internal(
    charging_strategy: str,
    raw_value: float,
    strategy_unit: str,
    step_hours: float,
) -> float:
    """
    Normalisiert Strategie-CSV-Werte auf interne Einheiten:

    - generation: internes Signal = kW
        * kW  -> kW
        * kWh -> kW  (kWh pro Zeitschritt / step_hours)
        * MWh -> kW  (MWh pro Zeitschritt * 1000 / step_hours)

    - market: internes Signal = €/kWh
        * €/MWh -> €/kWh ( /1000 )
        * €/kWh -> €/kWh
    """
    unit = (strategy_unit or "").strip()

    if charging_strategy == "generation":
        if unit == "kW":
            return float(raw_value)
        if unit == "kWh":
            return float(raw_value) / float(step_hours)
        if unit == "MWh":
            return float(raw_value) * 1000.0 / float(step_hours)
        raise ValueError(
            f"❌ Abbruch: Unbekannte strategy_unit='{strategy_unit}' für generation. Erlaubt: 'kW', 'kWh', 'MWh'."
        )

    if charging_strategy == "market":
        if unit == "€/kWh":
            return float(raw_value)
        if unit == "€/MWh":
            return float(raw_value) / 1000.0
        raise ValueError(
            f"❌ Abbruch: Unbekannte strategy_unit='{strategy_unit}' für market. Erlaubt: '€/kWh', '€/MWh'."
        )

    return float(raw_value)


# =============================================================================
# 7d) Reporting/Notebook-Helper: Strategie-Signal als aligned Zeitreihe
# =============================================================================

def build_strategy_signal_series(
    scenario: dict[str, Any],
    timestamps: list[datetime],
    charging_strategy: str,
    normalize_to_internal: bool = True,
    strategy_resolution_min: int = 15,
) -> tuple[np.ndarray | None, str | None]:
    """
    Baut eine Signal-Zeitreihe aligned auf 'timestamps' (für Plot/Reporting).

    Rückgabe:
      - series: np.ndarray der Länge len(timestamps), np.nan wenn kein Wert gefunden
      - y_label: passende Achsenbeschriftung
    """
    strat = (charging_strategy or "immediate").lower()
    if strat not in ("market", "generation"):
        return None, None

    if not timestamps:
        return None, None

    site_cfg = scenario.get("site", {}) or {}

    strategy_unit = str(site_cfg.get("strategy_unit", "") or "").strip()
    if not strategy_unit:
        raise ValueError(
            "❌ Abbruch: Für market/generation muss 'site.strategy_unit' gesetzt sein "
            "(market: '€/MWh' oder '€/kWh' | generation: 'MWh' oder 'kWh' oder 'kW')."
        )

    strategy_csv = site_cfg.get("strategy_csv", None)
    col_1_based = site_cfg.get("strategy_value_col", None)
    if not strategy_csv or not isinstance(col_1_based, int) or col_1_based < 2:
        raise ValueError("❌ Abbruch: 'site.strategy_csv' oder 'site.strategy_value_col' fehlt/ungültig.")

    csv_path = resolve_path_relative_to_scenario(scenario, str(strategy_csv))

    strategy_map = read_strategy_series_from_csv_first_col_time(
        csv_path=csv_path,
        value_col_1_based=int(col_1_based),
        delimiter=";",
    )

    step_hours = strategy_resolution_min / 60.0

    series = np.full(len(timestamps), np.nan, dtype=float)
    for i, ts in enumerate(timestamps):
        v = lookup_signal(strategy_map, ts, strategy_resolution_min)
        if v is None:
            continue

        if normalize_to_internal:
            series[i] = convert_strategy_value_to_internal(
                charging_strategy=strat,
                raw_value=float(v),
                strategy_unit=strategy_unit,
                step_hours=step_hours,
            )
        else:
            series[i] = float(v)

    if normalize_to_internal:
        y_label = "Preis [€/kWh]" if strat == "market" else "Erzeugung [kW]"
    else:
        y_label = f"{strat.upper()} [{strategy_unit}]"

    return series, y_label


# =============================================================================
# 7e) Optional: Basislast aus CSV (für Generation)
# =============================================================================

def build_base_load_series(
    scenario: dict[str, Any],
    timestamps: list[datetime],
    base_load_resolution_min: int = 15,
) -> np.ndarray | None:
    """
    Es wird eine Basislast-Zeitreihe aligned auf 'timestamps' erzeugt.

    Priorität:
      1) site.base_load_csv (wenn gesetzt)  -> Zeitreihe aus CSV (intern kW)
      2) site.base_load_kw  (wenn gesetzt)  -> konstante Grundlast (intern kW)
      3) None                               -> keine Basislast

    CSV-Annahmen (falls genutzt):
      - Spalte 1: Zeitstempel
      - Spalte site.base_load_value_col: Basislastwert
      - Einheit site.base_load_unit: 'kW' | 'kWh' | 'MWh'
    """
    if not timestamps:
        return None

    site_cfg = scenario.get("site", {}) or {}

    # ------------------------------------------------------------
    # 1) CSV hat Vorrang, falls vorhanden
    # ------------------------------------------------------------
    base_load_csv = site_cfg.get("base_load_csv", None)
    if base_load_csv:
        base_load_col = site_cfg.get("base_load_value_col", None)
        base_load_unit = str(site_cfg.get("base_load_unit", "") or "").strip()

        if not isinstance(base_load_col, int) or base_load_col < 2:
            raise ValueError("❌ Abbruch: 'site.base_load_value_col' fehlt/ungültig (muss int >= 2 sein).")
        if base_load_unit not in ("kW", "kWh", "MWh"):
            raise ValueError("❌ Abbruch: 'site.base_load_unit' muss 'kW', 'kWh' oder 'MWh' sein.")

        csv_path = resolve_path_relative_to_scenario(scenario, str(base_load_csv))
        base_map = read_strategy_series_from_csv_first_col_time(
            csv_path=csv_path,
            value_col_1_based=int(base_load_col),
            delimiter=";",
        )

        step_hours = base_load_resolution_min / 60.0

        series_kw = np.full(len(timestamps), np.nan, dtype=float)
        for i, ts in enumerate(timestamps):
            v = lookup_signal(base_map, ts, base_load_resolution_min)
            if v is None:
                continue

            if base_load_unit == "kW":
                series_kw[i] = float(v)
            elif base_load_unit == "kWh":
                series_kw[i] = float(v) / step_hours
            else:  # "MWh"
                series_kw[i] = float(v) * 1000.0 / step_hours

        return series_kw

    # ------------------------------------------------------------
    # 2) Konstante Grundlast, falls gesetzt
    # ------------------------------------------------------------
    base_load_kw = site_cfg.get("base_load_kw", None)
    if base_load_kw is not None:
        return np.full(len(timestamps), float(base_load_kw), dtype=float)

    # ------------------------------------------------------------
    # 3) Keine Basislast konfiguriert
    # ------------------------------------------------------------
    return None


# =============================================================================
# 8) Slot-basierte Heuristik (Best Slot innerhalb der Standzeit)
# =============================================================================

def _slack_minutes_for_session(
    s: dict[str, Any],
    ts: datetime,
    rated_power_kw: float,
    charger_efficiency: float,
) -> float:
    """
    Berechnet den zeitlichen Puffer (Slack) einer Session im Zeitpunkt ts.

    Definition:
      slack = verbleibende Standzeit - benötigte Ladezeit

    Dabei wird die benötigte Ladezeit so berechnet, als ob das Fahrzeug ab jetzt
    dauerhaft mit der maximal physikalisch möglichen Leistung laden würde.

    Interpretation:
      - Slack groß  -> viel Spielraum, Laden kann verschoben werden.
      - Slack klein -> zeitkritisch, Laden sollte sofort beginnen.
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
    strategy_unit: str,
    step_hours: float,
) -> None:
    """
    Erstellt pro Session eine sortierte Wunschliste an Slots innerhalb der Standzeit.

    Sortierkriterium:
      - market: günstigste Preise zuerst (€/kWh aufsteigend)
      - generation: höchste Erzeugung zuerst (kW absteigend)

    Ergebnis:
      - s["preferred_slot_indices"] enthält Indizes im time_index.
      - s["preferred_ptr"] zeigt auf den nächsten noch relevanten Slot >= aktuellem Zeitschritt.
    """
    if charging_strategy not in ("market", "generation"):
        return
    if not strategy_map or not time_index:
        return

    for s in all_sessions:
        if float(s.get("energy_required_kwh", 0.0)) <= 0.0:
            continue

        a = floor_to_resolution(s["arrival_time"], strategy_resolution_min)
        d = floor_to_resolution(s["departure_time"], strategy_resolution_min)

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
            sig_raw = lookup_signal(strategy_map, t, strategy_resolution_min)

            if sig_raw is None:
                score = 1e30
            else:
                sig_internal = convert_strategy_value_to_internal(
                    charging_strategy=charging_strategy,
                    raw_value=float(sig_raw),
                    strategy_unit=strategy_unit,
                    step_hours=step_hours,
                )

                if charging_strategy == "market":
                    score = float(sig_internal)       # kleiner = besser
                else:
                    score = -float(sig_internal)      # größer = besser

            scored.append((score, idx))

        scored.sort(key=lambda x: x[0])
        s["preferred_slot_indices"] = [idx for _, idx in scored]
        s["preferred_ptr"] = 0


# =============================================================================
# 8b) GENERATION – PV-Reservation + Grid-Only-If-Necessary (Rolling Horizon)
# =============================================================================

def _session_power_cap_kw(s: dict[str, Any], rated_power_kw: float) -> float:
    """
    Gibt die maximale Ladeleistung (kW) zurück, die diese Session physikalisch zulässt.
    Es wird bewusst eine Obergrenze verwendet (ohne Zukunfts-SoC-Tapering zu simulieren),
    damit die PV-Planung nicht unnötig pessimistisch ist.
    """
    return max(
        0.0,
        min(
            float(rated_power_kw),
            float(s.get("max_charging_power_kw", rated_power_kw)),
        ),
    )


def _idx_of_last_step_before_departure(
    departure_time: datetime,
    time_index: list[datetime],
    time_resolution_min: int,
) -> int:
    """
    Findet den letzten Zeitschritt-Index, der noch innerhalb der Standzeit liegt
    (ts < departure_time). Dieser Index wird als Ende der Planungsfenstergrenze genutzt.
    """
    if not time_index:
        return 0

    # Die Standzeit wird auf das Simulationsraster abgebildet.
    d = floor_to_resolution(departure_time, time_resolution_min)

    # Falls d nicht exakt in time_index ist, wird auf den letzten vorhandenen Slot zurückgegangen.
    # (dies ist robust gegen Rundungsartefakte / nicht exakt passende Timestamps)
    time_to_idx = {t: i for i, t in enumerate(time_index)}
    while d not in time_to_idx and d > time_index[0]:
        d -= timedelta(minutes=time_resolution_min)

    if d in time_to_idx:
        return int(time_to_idx[d])

    # Fallback: bis zum Ende planen
    return len(time_index) - 1


def _plan_pv_commitments_for_present_sessions(
    present_sessions: list[dict[str, Any]],
    i_now: int,
    pv_surplus_series_kw: np.ndarray,
    time_index: list[datetime],
    time_resolution_min: int,
    time_step_hours: float,
    charger_efficiency: float,
    rated_power_kw: float,
) -> tuple[np.ndarray, set[int]]:
    """
    Rolling-Horizon PV-Planung (Reservation):

    Es wird pro Zeitschritt eine „PV-Commitment“-Zeitreihe gebaut, die beschreibt,
    wie viel PV-Überschuss (kW) in zukünftigen Slots bereits „verplant“ ist.

    Ziel:
      - Wenn eine Session ihren Restbedarf vollständig aus (verfügbarem) PV-Überschuss
        innerhalb der Reststandzeit decken könnte, wird Grid für diese Session gesperrt.
      - Wenn PV nicht ausreicht, wird die Session als „grid-needed“ markiert.

    Heuristik:
      - Sessions werden nach frühester Abfahrt priorisiert (EDF), damit enge Deadlines
        zuerst PV reservieren dürfen.
      - PV-Slots werden nach höchstem verfügbarem PV-Überschuss gewählt (damit PV-Spitzen
        genutzt werden), bei gleichen Werten eher früher.

    Rückgabe:
      - pv_commit_kw: np.ndarray (kW) gleicher Länge wie pv_surplus_series_kw,
                      enthält verplante PV pro Slot.
      - grid_needed_session_ids: set[int] der id(s)-Werte, die trotz PV-Reservation
                                 nicht vollständig versorgt werden können.
    """
    n = len(pv_surplus_series_kw)
    pv_commit_kw = np.zeros(n, dtype=float)

    if not present_sessions or i_now >= n:
        return pv_commit_kw, set()

    # Verfügbare PV ab „jetzt“ ist die Überschusszeitreihe (kW)
    pv_available_kw = np.array(pv_surplus_series_kw, dtype=float)
    pv_available_kw[:i_now] = 0.0  # Vergangenheit irrelevant

    # Sessions nach Deadline (Abfahrt), dann Ankunft
    present_sorted = sorted(
        present_sessions,
        key=lambda s: (s["departure_time"], s["arrival_time"]),
    )

    grid_needed_session_ids: set[int] = set()

    for s in present_sorted:
        e_need_kwh = float(s.get("energy_required_kwh", 0.0))
        if e_need_kwh <= 1e-9:
            continue

        cap_kw = _session_power_cap_kw(s, rated_power_kw)
        if cap_kw <= 1e-9:
            # Kann physikalisch nicht laden -> muss (wenn überhaupt) als grid-needed gelten
            grid_needed_session_ids.add(id(s))
            continue

        # Fenster: von jetzt bis (letzter Slot vor Abfahrt)
        end_idx = _idx_of_last_step_before_departure(
            departure_time=s["departure_time"],
            time_index=time_index,
            time_resolution_min=time_resolution_min,
        )
        start_idx = i_now
        if end_idx <= start_idx:
            grid_needed_session_ids.add(id(s))
            continue

        # Kandidatenslots im Fenster sortieren:
        # - zuerst hohe PV-Verfügbarkeit, bei Gleichstand früher
        slot_indices = list(range(start_idx, min(end_idx + 1, n)))
        slot_indices.sort(key=lambda j: (-pv_available_kw[j], j))

        e_remaining_kwh = e_need_kwh

        for j in slot_indices:
            if e_remaining_kwh <= 1e-9:
                break

            avail_kw = float(pv_available_kw[j])
            if avail_kw <= 1e-9:
                continue

            # In diesem Slot kann die Session höchstens cap_kw nutzen.
            # Zusätzlich wird nur so viel reserviert, wie zur Deckung der Restenergie nötig ist.
            max_kw_for_energy = e_remaining_kwh / (time_step_hours * charger_efficiency)
            take_kw = min(avail_kw, cap_kw, max_kw_for_energy)

            if take_kw <= 1e-9:
                continue

            pv_commit_kw[j] += take_kw
            pv_available_kw[j] -= take_kw

            e_remaining_kwh -= take_kw * time_step_hours * charger_efficiency

        # Wenn nach Reservation noch Restbedarf bleibt -> PV reicht nicht -> Grid wäre nötig
        if e_remaining_kwh > 1e-6:
            grid_needed_session_ids.add(id(s))

    return pv_commit_kw, grid_needed_session_ids


# =============================================================================
# 9) Hauptsimulation / Orchestrierung der Lastgangberechnung
# =============================================================================

def simulate_load_profile(
    scenario: dict,
    start_datetime: datetime | None = None,
    record_debug: bool = False,
):
    """
    Führt die Lastgang-Simulation durch.

    Strategien:
      - immediate  : lädt sofort nach Ankunft (Deadline-getrieben)
      - market     : lädt bevorzugt in günstigen Slots (Kostenminimierung)
      - generation : lädt bevorzugt bei hoher Eigenerzeugung (Eigenverbrauchsmaximierung)

    Fixe Heuristik-Parameter (bewusst NICHT in YAML, um YAML schlank zu halten):
      - MARKET_TOP_K_SLOTS:
          * Für market gilt ein Preisqualitätsfilter: ein Slot ist zulässig, wenn er zu den K besten
            (günstigsten) Slots innerhalb der Standzeit gehört, die aus Sicht des aktuellen Zeitschritts
            noch in der Zukunft liegen.
          * Dadurch wird nicht nur die nächstgünstige Viertelstunde genutzt, sondern generell günstige Slots.
      - EMERGENCY_SLACK_MINUTES:
          * Unabhängig von Preis/Erzeugung wird geladen, wenn ein Fahrzeug zeitkritisch wird und sonst
            das Ziel-SoC voraussichtlich nicht erreicht wird.

    Physikalische Rahmenbedingungen:
      - Der Netzanschlusspunkt (NAP) begrenzt den Import aus dem öffentlichen Netz.
      - Lokale Erzeugung (z.B. PV) reduziert den Netzbezug, erhöht aber nicht das Importlimit.
      - Die Gesamtleistung ist zusätzlich durch Ladepunktleistung und Fahrzeuglimit begrenzt.
    """
    # ------------------------------------------------------------
    # Fixe Heuristik-Parameter (nicht in YAML)
    # ------------------------------------------------------------
    EMERGENCY_SLACK_MINUTES = 60.0
    MARKET_TOP_K_SLOTS = 4

    # ------------------------------------------------------------
    # 1) Zeitindex erzeugen (Simulationszeitachse)
    # ------------------------------------------------------------
    time_index = create_time_index(scenario, start_datetime)

    # ------------------------------------------------------------
    # 2) Feiertage einmal ableiten
    # ------------------------------------------------------------
    simulation_start_datetime = time_index[0] if time_index else (
        start_datetime if start_datetime is not None else
        datetime.fromisoformat(scenario["start_datetime"]) if "start_datetime" in scenario else
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    )

    holiday_dates = parse_holiday_dates_from_scenario(
        scenario=scenario,
        simulation_start_datetime=simulation_start_datetime,
    )

    # ------------------------------------------------------------
    # 3) Fahrzeugflotte inkl. Ladekurven aus CSV laden
    # ------------------------------------------------------------
    vehicle_csv_path = resolve_path_relative_to_scenario(scenario, scenario["vehicles"]["vehicle_curve_csv"])
    vehicle_profiles = load_vehicle_profiles_from_csv(vehicle_csv_path)

    # ------------------------------------------------------------
    # 4) Strategie-Initialisierung
    # ------------------------------------------------------------
    site_cfg = scenario.get("site", {}) or {}
    charging_strategy = (site_cfg.get("charging_strategy") or "immediate").lower()

    STRATEGY_RESOLUTION_MIN = 15
    strategy_step_hours = STRATEGY_RESOLUTION_MIN / 60.0

    strategy_map: dict[datetime, float] | None = None
    strategy_csv_path: str | None = None

    strategy_unit = str(site_cfg.get("strategy_unit", "") or "").strip()

    if charging_strategy in ("market", "generation"):
        if not strategy_unit:
            raise ValueError(
                "❌ Abbruch: Für charging_strategy 'market' oder 'generation' muss 'site.strategy_unit' gesetzt sein "
                "(market: '€/MWh' oder '€/kWh' | generation: 'MWh' oder 'kWh' oder 'kW')."
            )

        strategy_csv = site_cfg.get("strategy_csv", None)
        if not strategy_csv:
            raise ValueError(
                "❌ Abbruch: Für charging_strategy 'market' oder 'generation' muss in der YAML 'site.strategy_csv' gesetzt sein."
            )

        col_1_based = site_cfg.get("strategy_value_col", None)
        if col_1_based is None or not isinstance(col_1_based, int) or col_1_based < 2:
            raise ValueError(
                "❌ Abbruch: 'site.strategy_value_col' muss eine ganze Zahl >= 2 sein (1=Zeitspalte)."
            )

        strategy_csv_path = resolve_path_relative_to_scenario(scenario, strategy_csv)

        strategy_map = read_strategy_series_from_csv_first_col_time(
            csv_path=strategy_csv_path,
            value_col_1_based=int(col_1_based),
            delimiter=";",
        )

        assert_strategy_csv_covers_simulation(
            strategy_map=strategy_map,
            time_index=time_index,
            strategy_resolution_min=STRATEGY_RESOLUTION_MIN,
            charging_strategy=charging_strategy,
            strategy_csv_path=strategy_csv_path,
        )

    # ------------------------------------------------------------
    # 5) Initialisierung: Parameter & Ergebniscontainer
    # ------------------------------------------------------------
    time_resolution_min = scenario["time_resolution_min"]
    time_step_hours = time_resolution_min / 60.0

    n_steps = len(time_index)
    load_profile_kw = np.zeros(n_steps, dtype=float)

    # grid_limit_p_avb_kw ist hier als Importlimit aus dem öffentlichen Netz interpretiert (NAP-Importlimit).
    # Lokale Erzeugung (PV) reduziert den Netzbezug, erhöht aber nicht das Importlimit.
    grid_limit_p_avb_kw = scenario["site"]["grid_limit_p_avb_kw"]
    rated_power_kw = scenario["site"]["rated_power_kw"]
    number_of_chargers = scenario["site"]["number_chargers"]
    charger_efficiency = scenario["site"]["charger_efficiency"]

    # -------------------------------------------------------------------
    # 5b) Basislast (nicht-flexible Last) vorbereiten (relevant für generation)
    # -------------------------------------------------------------------
    # Es wird eine Zeitreihe base_load_series[t] (kW) aufgebaut:
    #   - aus CSV (site.base_load_csv) oder
    #   - konstant (site.base_load_kw)
    # Wenn beides fehlt, wird base_load_series = None und Basislast = 0 kW angenommen.
    #
    # Die Basislast wird später von der lokalen Erzeugung abgezogen:
    #   PV_Überschuss = max(0, PV - Basislast)
    base_load_series = None
    if charging_strategy == "generation":
        base_load_series = build_base_load_series(
            scenario=scenario,
            timestamps=time_index,
            base_load_resolution_min=STRATEGY_RESOLUTION_MIN,
        )

    def _base_load_kw_at(i: int) -> float:
        """Gibt die Basislast (kW) für den Zeitschritt i zurück (NaN/None -> 0)."""
        if base_load_series is None:
            return 0.0
        v = float(base_load_series[i])
        return 0.0 if np.isnan(v) else max(0.0, v)

    def _generation_kw_at(ts: datetime) -> float:
        """
        Gibt die lokale Erzeugung (kW) für den Zeitschritt ts zurück.
        Fehlende/NaN-Werte werden als 0 interpretiert.
        """
        if charging_strategy != "generation":
            return 0.0
        if not strategy_map:
            return 0.0

        raw = lookup_signal(strategy_map, ts, STRATEGY_RESOLUTION_MIN)
        if raw is None:
            return 0.0

        return max(
            0.0,
            float(
                convert_strategy_value_to_internal(
                    charging_strategy="generation",
                    raw_value=float(raw),
                    strategy_unit=strategy_unit,
                    step_hours=strategy_step_hours,
                )
            ),
        )

    # -------------------------------------------------------------------
    # 5b+) PV-Überschuss-Zeitreihe (kW) vorberechnen (nur generation)
    # -------------------------------------------------------------------
    # Diese Zeitreihe ermöglicht eine Rolling-Horizon PV-Reservation:
    # - Fahrzeuge sollen kein Grid nutzen, wenn sie ihren Bedarf innerhalb der Standzeit
    #   aus zukünftigem PV-Überschuss decken könnten.
    pv_surplus_series_kw = None
    if charging_strategy == "generation":
        pv_surplus_series_kw = np.zeros(len(time_index), dtype=float)
        for ii, tts in enumerate(time_index):
            pv_kw = _generation_kw_at(tts)
            base_kw = _base_load_kw_at(ii)
            pv_surplus_series_kw[ii] = max(0.0, pv_kw - base_kw)


    # -------------------------------------------------------------------
    # 5c) Strategieparameter (zentral definiert, im ganzen Loop verfügbar)
    # -------------------------------------------------------------------
    # emergency_slack_minutes definiert, ab welchem zeitlichen Puffer (Slack)
    # eine Session als kritisch gilt und unabhängig von PV/Preis laden darf.
    strategy_params = site_cfg.get("strategy_params", {}) or {}
    emergency_slack_minutes = float(strategy_params.get("emergency_slack_minutes", 60.0))

    charging_count_series: list[int] = []
    all_charging_sessions: list[dict[str, Any]] = []


    # ------------------------------------------------------------
    # 5d) Debug-Log (optional): protokolliert pro Zeitschritt, 
    # welche Sessions laden und ob dabei Netzbezug entsteht.
    # ------------------------------------------------------------
    debug_rows: list[dict[str, Any]] = []
   

    # ------------------------------------------------------------
    # 6) Tagesweise Ladesessions erzeugen
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 6b) Präferenzlisten je Session vorberechnen
    # ------------------------------------------------------------
    time_to_idx = {t: idx for idx, t in enumerate(time_index)} if time_index else {}
    _build_preferred_slots_for_all_sessions(
        all_sessions=all_charging_sessions,
        time_index=time_index,
        time_to_idx=time_to_idx,
        charging_strategy=charging_strategy,
        strategy_map=strategy_map,
        strategy_resolution_min=STRATEGY_RESOLUTION_MIN,
        strategy_unit=strategy_unit,
        step_hours=strategy_step_hours,
    )

    # ------------------------------------------------------------
    # 7) Zeitschrittweise Simulation
    # ------------------------------------------------------------
    for i, ts in enumerate(time_index):

        # Anwesende Sessions mit Restbedarf
        present_sessions = [
            s for s in all_charging_sessions
            if s["arrival_time"] <= ts < s["departure_time"]
            and s["energy_required_kwh"] > 0.0
        ]

        if not present_sessions:
            load_profile_kw[i] = 0.0
            charging_count_series.append(0)
            continue

        # --------------------------------------------------------
        # 7.1) Auswahl der zu ladenden Sessions
        # --------------------------------------------------------
        charging_sessions: list[dict[str, Any]] = []

        if charging_strategy == "immediate":
            # Bei immediate werden die Sessions mit frühester Abfahrt priorisiert.
            present_sessions.sort(key=lambda s: (s["departure_time"], s["arrival_time"]))
            charging_sessions = present_sessions[:number_of_chargers]

        else:
            # Für market/generation werden Kandidatenlisten aufgebaut:
            #   A) emergency: zeitkritisch => es wird unabhängig vom Preis/PV geladen.
            #   B) slot_candidates: es wird im "gewünschten" Slot geladen.
            #
            # Erweiterung (generation):
            #   Wenn PV-Überschuss > 0 vorliegt und keine Notfall-Session aktiv ist,
            #   werden anwesende Sessions als Kandidaten zugelassen, auch wenn es nicht exakt
            #   ein Wunschslot ist. Dadurch wird PV-Überschuss besser genutzt und späterer
            #   Netzbezug vermieden bzw. reduziert.

            emergency: list[dict[str, Any]] = []
            slot_candidates: list[dict[str, Any]] = []

            # PV-Überschuss im aktuellen Zeitschritt (nur relevant für generation)
            pv_surplus_kw = 0.0
        if charging_strategy == "generation":
            # ---------------------------------------------------------------
            # GENERATION (kombiniert): PV-Reservation + Grid-Only-If-Necessary
            # ---------------------------------------------------------------
            # Die Logik arbeitet in 3 Stufen:
            # (1) Rolling-Horizon PV-Reservation (Planung):
            #     Es wird geprüft, welche Sessions ihren Restbedarf vollständig aus
            #     zukünftigem PV-Überschuss decken könnten. Für diese Sessions wird
            #     Grid gesperrt.
            #
            # (2) Auswahl aktiver Sessions (Charger-Limit):
            #     Sessions, die PV nicht vollständig abdecken kann, werden als
            #     "grid-needed" priorisiert, damit Ziel-SoC nicht verfehlt wird.
            #
            # (3) Leistungsverteilung:
            #     - PV wird zuerst verteilt (PV-only Sessions zuerst).
            #     - Grid wird ausschließlich für grid-needed Sessions genutzt.
            #
            # Hinweis:
            # - Diese Heuristik ist Rolling-Horizon: die PV-Planung bezieht sich
            #   immer auf „jetzt bis Abfahrt“ und wird pro Zeitschritt neu bewertet.

            # (A) PV-Überschuss im aktuellen Zeitschritt (kW)
            base_kw_now = _base_load_kw_at(i)
            gen_kw_now = _generation_kw_at(ts)
            pv_surplus_kw_now = max(0.0, gen_kw_now - base_kw_now)

            # (B) Rolling-Horizon PV-Reservation + Grid-Feasibility
            #     Es wird auf Basis pv_surplus_series_kw entschieden, ob PV in der Restzeit reichen könnte.
            if pv_surplus_series_kw is None:
                pv_surplus_series_kw = np.zeros(len(time_index), dtype=float)

            pv_commit_kw, grid_needed_ids = _plan_pv_commitments_for_present_sessions(
                present_sessions=present_sessions,
                i_now=i,
                pv_surplus_series_kw=pv_surplus_series_kw,
                time_index=time_index,
                time_resolution_min=time_resolution_min,
                time_step_hours=time_step_hours,
                charger_efficiency=charger_efficiency,
                rated_power_kw=rated_power_kw,
            )

            # Sessions markieren: Grid-Status + Slack
            pv_only_candidates: list[dict[str, Any]] = []
            grid_needed_candidates: list[dict[str, Any]] = []

            for s in present_sessions:
                slack_m = _slack_minutes_for_session(
                    s=s,
                    ts=ts,
                    rated_power_kw=rated_power_kw,
                    charger_efficiency=charger_efficiency,
                )
                s["_slack_minutes"] = float(slack_m)

                is_grid_needed = (id(s) in grid_needed_ids)
                s["_grid_needed"] = bool(is_grid_needed)
                s["_grid_forbidden"] = bool(not is_grid_needed)

                if is_grid_needed:
                    grid_needed_candidates.append(s)
                else:
                    pv_only_candidates.append(s)

            # (C) Charger-Auswahl:
            #     - grid-needed zuerst (Ziel-SoC darf nicht verfehlt werden)
            #     - dann PV-only (nutzt PV-Überschuss, vermeidet Grid)
            grid_needed_candidates.sort(key=lambda s: (s.get("_slack_minutes", 1e9), s["departure_time"], s["arrival_time"]))
            pv_only_candidates.sort(key=lambda s: (s.get("_slack_minutes", 1e9), s["departure_time"], s["arrival_time"]))

            charging_sessions: list[dict[str, Any]] = []
            charging_sessions.extend(grid_needed_candidates[:number_of_chargers])

            remaining_slots = number_of_chargers - len(charging_sessions)
            if remaining_slots > 0:
                charging_sessions.extend(pv_only_candidates[:remaining_slots])

            charging_count_series.append(len(charging_sessions))

            if not charging_sessions:
                load_profile_kw[i] = 0.0
                continue

            # (D) Budgets:
            #     - Chargerbudget ist physikalische Abgabeleistung (kW)
            #     - Gridbudget ist Importlimit (kW), darf nur für grid-needed Sessions genutzt werden
            charger_budget_kw = len(charging_sessions) * rated_power_kw
            grid_budget_kw = max(0.0, float(grid_limit_p_avb_kw))
            pv_budget_kw = float(pv_surplus_kw_now)

            # Gruppen trennen
            grid_needed_sessions = [s for s in charging_sessions if bool(s.get("_grid_needed", False))]
            pv_only_sessions = [s for s in charging_sessions if not bool(s.get("_grid_needed", False))]

            total_power_kw = 0.0

            # -----------------------------------------------------------
            # (E) PV-only Sessions zuerst: dürfen ausschließlich PV nutzen
            # -----------------------------------------------------------
            if pv_only_sessions and pv_budget_kw > 1e-9 and charger_budget_kw > 1e-9:
                pv_only_budget_kw = min(pv_budget_kw, charger_budget_kw)
                per_session_kw = pv_only_budget_kw / len(pv_only_sessions)

                pv_only_used_kw = 0.0

                for s in pv_only_sessions:
                    vehicle_power_limit_kw = vehicle_power_at_soc(s)

                    allowed_power_kw = min(
                        per_session_kw,
                        rated_power_kw,
                        vehicle_power_limit_kw,
                        float(s.get("max_charging_power_kw", rated_power_kw)),
                    )

                    possible_energy_kwh = allowed_power_kw * time_step_hours * charger_efficiency
                    energy_needed = float(s["energy_required_kwh"])

                    if possible_energy_kwh >= energy_needed:
                        energy_delivered = energy_needed
                        s["energy_required_kwh"] = 0.0
                        actual_power_kw = energy_delivered / (time_step_hours * charger_efficiency)
                    else:
                        energy_delivered = possible_energy_kwh
                        s["energy_required_kwh"] = energy_needed - possible_energy_kwh
                        actual_power_kw = allowed_power_kw

                    s["delivered_energy_kwh"] += energy_delivered
                    s["_actual_power_kw"] = float(actual_power_kw)

                    pv_only_used_kw += actual_power_kw
                    total_power_kw += actual_power_kw

                # Budgets reduzieren
                pv_budget_kw = max(0.0, pv_budget_kw - pv_only_used_kw)
                charger_budget_kw = max(0.0, charger_budget_kw - pv_only_used_kw)

            # -----------------------------------------------------------------
            # (F) Grid-needed Sessions: dürfen PV + Grid nutzen (aber nur hier)
            # -----------------------------------------------------------------
            # Diese Sessions werden nur dann als grid-needed klassifiziert, wenn
            # PV-Reservation zeigt, dass PV innerhalb der Reststandzeit nicht reicht.
            if grid_needed_sessions and charger_budget_kw > 1e-9:
                usable_budget_kw = min(
                    charger_budget_kw,
                    pv_budget_kw + grid_budget_kw,
                )
                per_session_kw = usable_budget_kw / len(grid_needed_sessions)

                grid_needed_used_kw = 0.0

                for s in grid_needed_sessions:
                    vehicle_power_limit_kw = vehicle_power_at_soc(s)

                    allowed_power_kw = min(
                        per_session_kw,
                        rated_power_kw,
                        vehicle_power_limit_kw,
                        float(s.get("max_charging_power_kw", rated_power_kw)),
                    )

                    possible_energy_kwh = allowed_power_kw * time_step_hours * charger_efficiency
                    energy_needed = float(s["energy_required_kwh"])

                    if possible_energy_kwh >= energy_needed:
                        energy_delivered = energy_needed
                        s["energy_required_kwh"] = 0.0
                        actual_power_kw = energy_delivered / (time_step_hours * charger_efficiency)
                    else:
                        energy_delivered = possible_energy_kwh
                        s["energy_required_kwh"] = energy_needed - possible_energy_kwh
                        actual_power_kw = allowed_power_kw

                    s["delivered_energy_kwh"] += energy_delivered
                    s["_actual_power_kw"] = float(actual_power_kw)

                    grid_needed_used_kw += actual_power_kw
                    total_power_kw += actual_power_kw

                # PV wird zuerst verwendet, Rest kommt aus Grid (nur in dieser Gruppe)
                pv_used_by_grid_needed_kw = min(pv_budget_kw, grid_needed_used_kw)
                pv_budget_kw = max(0.0, pv_budget_kw - pv_used_by_grid_needed_kw)

                # Grid-Import wäre der Restanteil, wird aber hier nur budgettechnisch gekappt:
                grid_used_kw = max(0.0, grid_needed_used_kw - pv_used_by_grid_needed_kw)
                grid_used_kw = min(grid_used_kw, grid_budget_kw)

                # Budgets reduzieren (physikalisch relevant)
                charger_budget_kw = max(0.0, charger_budget_kw - grid_needed_used_kw)

            load_profile_kw[i] = float(total_power_kw)

        else:
            # ---------------------------------------------------------------
            # Fallback für immediate/market: klassische Fair-Share-Logik
            # ---------------------------------------------------------------
            max_site_power_kw = min(
                grid_limit_p_avb_kw,
                len(charging_sessions) * rated_power_kw,
            )
            available_power_per_session_kw = max_site_power_kw / len(charging_sessions)

            total_power_kw = 0.0

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
                s["_actual_power_kw"] = float(actual_power_kw)

                total_power_kw += actual_power_kw

            load_profile_kw[i] = total_power_kw


        # ---------------------------------------------------------------
        # Debug: Netzbezug je Zeitschritt + Session-Details loggen
        # ---------------------------------------------------------------
        if record_debug:
            # PV- und Basislast nur für generation relevant
            if charging_strategy == "generation":
                pv_kw = _generation_kw_at(ts)
                base_kw = _base_load_kw_at(i)
                pv_surplus_kw = max(0.0, pv_kw - base_kw)
            else:
                pv_kw = 0.0
                base_kw = 0.0
                pv_surplus_kw = 0.0

            has_any_emergency_this_step = any(
                float(s.get("_slack_minutes", 1e9)) <= emergency_slack_minutes
                for s in charging_sessions
            )

            # Standort-Netzimport (nur generation physikalisch sinnvoll interpretiert)
            if charging_strategy == "generation":
                site_grid_import_kw = max(0.0, total_power_kw - pv_surplus_kw)
            else:
                site_grid_import_kw = max(0.0, total_power_kw)

            # Summe der Leistung der Emergency-Sessions (für proportionalen Netzanteil)
            emergency_power_kw = sum(
                float(s.get("_actual_power_kw", 0.0))
                for s in charging_sessions
                if float(s.get("_slack_minutes", 1e9)) <= emergency_slack_minutes
            )

            for s in charging_sessions:
                arrival_time = s["arrival_time"]
                departure_time = s["departure_time"]
                parking_hours = (departure_time - arrival_time).total_seconds() / 3600.0
                slack_minutes = float(s.get("_slack_minutes", np.nan))
                is_emergency = bool(slack_minutes <= emergency_slack_minutes)

                # -------------------------------------------------------
                # Netzanteile pro Session (2 Sichten!)
                #
                # (1) physical: Attribution des Standort-Netzimports proportional zur Session-Leistung.
                #     Das beantwortet: "Welche Fahrzeuge haben in einem Import-Zeitschritt geladen?"
                #
                # (2) policy: Netzimport wird nur Emergency-Sessions zugeschrieben (Regel-Logik).
                #     Das beantwortet: "Welche Fahrzeuge mussten laut Policy aus dem Netz laden dürfen?"
                # -------------------------------------------------------
                session_power_kw = float(s.get("_actual_power_kw", 0.0))

                grid_import_kw_session_physical = 0.0
                if charging_strategy == "generation" and total_power_kw > 1e-9:
                    grid_import_kw_session_physical = site_grid_import_kw * (session_power_kw / float(total_power_kw))

                grid_import_kw_session_policy = 0.0
                if charging_strategy == "generation" and is_emergency and emergency_power_kw > 1e-9:
                    grid_import_kw_session_policy = site_grid_import_kw * (session_power_kw / emergency_power_kw)

                debug_rows.append(
                    {
                        "ts": ts,
                        "vehicle_name": s.get("vehicle_name", ""),
                        "vehicle_class": s.get("vehicle_class", ""),
                        "arrival_time": arrival_time,
                        "departure_time": departure_time,
                        "parking_hours": parking_hours,
                        "slack_minutes": slack_minutes,
                        "is_emergency": is_emergency,
                        "pv_kw": float(pv_kw),
                        "base_kw": float(base_kw),
                        "pv_surplus_kw": float(pv_surplus_kw),
                        "site_total_power_kw": float(total_power_kw),
                        "grid_import_kw_site": float(site_grid_import_kw),
                        "grid_import_kw_session_physical": float(grid_import_kw_session_physical),
                        "grid_import_kw_session_policy": float(grid_import_kw_session_policy),
                        "has_any_emergency_this_step": bool(has_any_emergency_this_step),
                    }
                )



    # ------------------------------------------------------------
    # 8) Strategie-Status
    # ------------------------------------------------------------
    if charging_strategy == "immediate":
        strategy_status = "IMMEDIATE"
    elif charging_strategy in ("market", "generation"):
        strategy_status = "ACTIVE" if strategy_map else "INACTIVE"
    else:
        raise ValueError(f"❌ Abbruch: Unbekannte charging_strategy='{charging_strategy}'")

    # ------------------------------------------------------------
    # 9) Rückgabe
    # ------------------------------------------------------------
    return (
        time_index,
        load_profile_kw,
        all_charging_sessions,
        charging_count_series,
        holiday_dates,
        charging_strategy,
        strategy_status,
        debug_rows if record_debug else None,
    )
