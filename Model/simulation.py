import yaml
import numpy as np
from pathlib import Path
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Any


# =============================================================================
# Modulüberblick (Erläuterung in 3. Person)
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
# 0) Projekt-/Pfad-Utilities
# =============================================================================

def resolve_path_relative_to_scenario(scenario: dict[str, Any], p: str) -> str:
    """
    Diese Funktion löst Dateipfade robust auf.

    Sie sorgt dafür, dass:
      - absolute Pfade unverändert bleiben,
      - relative Pfade relativ zum YAML-Ordner interpretiert werden.

    Dadurch können Szenario-YAMLs portable gehalten werden,
    ohne dass der Nutzer absolute Pfade hardcoden muss.
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
    Diese Funktion parst numerische Strings robust für deutsches und englisches Zahlenformat.

    Unterstützt:
      - deutsches Format:  "1.234,56"
      - englisches Format: "1,234.56" oder "90.91"
      - deutsches Dezimalkomma ohne Tausender: "90,91"
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
            # deutsches Tausenderformat: 1.234,56
            s = s.replace(".", "").replace(",", ".")
            return float(s)
        else:
            # englisches Tausenderformat: 1,234.56
            s = s.replace(",", "")
            return float(s)

    if has_comma and not has_dot:
        # deutsches Dezimalformat: 90,91
        s = s.replace(",", ".")
        return float(s)

    return float(s)


# =============================================================================
# 0c) HTML-Statusausgabe (optional im Notebook)
# =============================================================================

def show_strategy_status_html(strategy: str, status: str) -> None:
    """
    Diese Funktion zeigt im Notebook eine farbige Statuszeile an.

    Sie visualisiert:
      - welche Strategie aktiv ist,
      - ob die Strategie aktiv (Signal geladen) oder inaktiv ist.

    Falls IPython nicht verfügbar ist, fällt sie auf print zurück.
    """
    status = (status or "IMMEDIATE").upper()
    strategy = (strategy or "immediate").upper()

    color_map = {"ACTIVE": "green", "INACTIVE": "red", "IMMEDIATE": "gray"}
    emoji_map = {"ACTIVE": "🟢", "INACTIVE": "🔴", "IMMEDIATE": "⚪"}

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
    """
    Diese Datenklasse repräsentiert ein Fahrzeugprofil (Ladekurve + Kapazität).
    """
    name: str
    battery_capacity_kwh: float
    vehicle_class: str
    soc_grid: np.ndarray
    power_grid_kw: np.ndarray


def load_vehicle_profiles_from_csv(path: str) -> list[VehicleProfile]:
    """
    Diese Funktion lädt Fahrzeugprofile aus einer CSV-Datei (Delimiter ';').

    Erwartete Struktur:
      Zeile 1: Hersteller (ignoriert)
      Zeile 2: Modellnamen
      Zeile 3: Fahrzeugklasse
      Zeile 4: Kapazität (kWh)
      Zeile 5: "SoC [%]" Header
      ab Zeile 6: SoC in % + Ladeleistungen je Fahrzeug
    """
    vehicle_profiles: list[VehicleProfile] = []

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";")

        _ = next(reader, None)  # Hersteller

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

        _ = next(reader, None)  # SoC Header

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
                    name=model_names[i],
                    battery_capacity_kwh=float(cap),
                    vehicle_class=vehicle_classes[i],
                    soc_grid=soc_grid,
                    power_grid_kw=power_grid,
                )
            )

    return vehicle_profiles


def vehicle_power_at_soc(session: dict[str, Any]) -> float:
    """
    Diese Funktion berechnet die maximal mögliche Ladeleistung (kW) eines Fahrzeugs
    bei aktuellem SoC per Interpolation der Ladekurve.
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
    Diese Funktion lädt ein YAML-Szenario und ergänzt den internen Kontextpfad.
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
    Diese Funktion interpretiert YAML-Werte als Skalar oder Bereich [min, max].
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
    Diese Funktion berechnet Feiertage für den Simulationszeitraum.
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
    Diese Funktion klassifiziert einen Zeitpunkt als working_day/saturday/sunday_holiday.
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
    Diese Funktion erzeugt die Simulationszeitachse als Liste von datetime.
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
    Diese Funktion zieht Samples aus einer Mischverteilung.
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
            value = np.random.lognormal(mean=float(component["mu"]), sigma=float(component["sigma"]))
        elif dist_type == "normal":
            value = np.random.normal(loc=float(component["mu"]), scale=float(component["sigma"]))
        elif dist_type == "beta":
            value = np.random.beta(a=float(component["alpha"]), b=float(component["beta"]))
        elif dist_type == "uniform":
            value = np.random.uniform(float(component["low"]), float(component["high"]))
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
    Diese Funktion realisiert stochastische YAML-Templates (Ranges -> konkrete Werte).
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
    Diese Funktion wählt ein Fahrzeugprofil aus der Flotte.
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
# 6) Grundmodell: Session-Generierung (Ankunft, Standzeit, SoC, Energiebedarf)
# =============================================================================

def sample_arrival_times_for_day(
    scenario: dict,
    day_start_datetime: datetime,
    holiday_dates: set[date],
) -> list[datetime]:
    """
    Diese Funktion erzeugt Ankunftszeiten für einen Tag.
    """
    day_type = determine_day_type_with_holidays(day_start_datetime, holiday_dates)

    number_of_chargers = scenario["site"]["number_chargers"]
    expected_sessions_per_charger = sample_from_range(
        scenario["site"]["expected_sessions_per_charger_per_day"]
    )

    weekday_weight = sample_from_range(
        scenario["arrival_time_distribution"]["weekday_weight"][day_type]
    )

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
    Diese Funktion zieht Parkdauern (in Minuten) aus einer Mischverteilung.
    """
    cfg = scenario["parking_duration_distribution"]
    max_minutes = cfg["max_duration_minutes"]
    min_minutes = cfg.get("min_duration_minutes", 10.0)

    mixture_components = realize_mixture_components(cfg["components"], allow_shift=False)

    durations = sample_mixture(
        number_of_samples=number_of_sessions,
        mixture_components=mixture_components,
        max_value=max_minutes,
        unit_description="minutes",
    )

    return np.clip(durations, min_minutes, max_minutes)


def sample_soc_upon_arrival(scenario: dict, number_of_sessions: int) -> np.ndarray:
    """
    Diese Funktion zieht SoC-Werte bei Ankunft aus einer Mischverteilung.
    """
    cfg = scenario["soc_at_arrival_distribution"]
    max_soc = cfg["max_soc"]

    mixture_components = realize_mixture_components(cfg["components"], allow_shift=False)

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
    Diese Funktion baut Session-Objekte (Ankunft/Abfahrt + Fahrzeug + Energiebedarf).
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

        sessions.append(
            {
                "session_id": f"{day_start_datetime.date()}_{i}",
                "arrival_time": arrival_time,
                "departure_time": departure_time,

                "soc_arrival": soc_at_arrival,
                "soc_target": target_soc,
                "battery_capacity_kwh": battery_capacity_kwh,

                "energy_required_kwh": float(required_energy_kwh),
                "delivered_energy_kwh": 0.0,

                "max_charging_power_kw": float(vehicle_profile.power_grid_kw.max()),
                "vehicle_name": vehicle_profile.name,
                "vehicle_class": vehicle_profile.vehicle_class,
                "soc_grid": vehicle_profile.soc_grid,
                "power_grid_kw": vehicle_profile.power_grid_kw,

                # Management-Bookkeeping:
                "_plug_in_time": None,
                "_charger_id": None,
                "_rejected": False,

                # Market/Geneation-Fallback Präferenzen:
                "preferred_slot_indices": [],
                "preferred_ptr": 0,
                "preferred_market_slot_indices": [],
                "preferred_market_ptr": 0,
            }
        )

    return sessions


def build_base_model(
    scenario: dict[str, Any],
    start_datetime: datetime | None = None,
) -> tuple[list[datetime], list[dict[str, Any]], set[date], list[VehicleProfile]]:
    """
    Diese Funktion erzeugt das Grundmodell der Simulation.

    Sie übernimmt explizit die Aufgaben:
      - Ermittlung Anzahl Ladevorgänge (Session-Count aus Distributionen),
      - Ermittlung Start Ladevorgang (Arrival-Time-Verteilung),
      - Ermittlung Standzeit (Parking-Duration),
      - Ermittlung Start-SOC & benötigte Energie,
      - Ermittlung Flotte (Vehicle Mix + Ladekurven).

    Rückgabe:
      - time_index
      - all_sessions (ungepluggt, später per Arrival-Policy belegt)
      - holiday_dates
      - vehicle_profiles (für Transparenz/Debug)
    """
    time_index = create_time_index(scenario, start_datetime)

    simulation_start_datetime = time_index[0] if time_index else (
        start_datetime if start_datetime is not None else
        datetime.fromisoformat(scenario["start_datetime"]) if "start_datetime" in scenario else
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    )

    holiday_dates = parse_holiday_dates_from_scenario(
        scenario=scenario,
        simulation_start_datetime=simulation_start_datetime,
    )

    vehicle_csv_path = resolve_path_relative_to_scenario(scenario, scenario["vehicles"]["vehicle_curve_csv"])
    vehicle_profiles = load_vehicle_profiles_from_csv(vehicle_csv_path)
    if not vehicle_profiles:
        raise ValueError("❌ Abbruch: Keine gültigen Fahrzeugprofile aus CSV geladen.")

    all_sessions: list[dict[str, Any]] = []
    if time_index:
        first_day_start = time_index[0].replace(hour=0, minute=0, second=0, microsecond=0)
        horizon_days = int(scenario["simulation_horizon_days"])
        for day_offset in range(horizon_days):
            day_start = first_day_start + timedelta(days=day_offset)
            all_sessions.extend(
                build_charging_sessions_for_day(
                    scenario=scenario,
                    day_start_datetime=day_start,
                    vehicle_profiles=vehicle_profiles,
                    holiday_dates=holiday_dates,
                )
            )

    all_sessions.sort(key=lambda s: s["arrival_time"])
    return time_index, all_sessions, holiday_dates, vehicle_profiles


# =============================================================================
# 7) Strategie-Signale (Market / Generation) aus CSV
# =============================================================================

CSV_DT_FORMATS = ("%d.%m.%Y %H:%M", "%d.%m.%y %H:%M")


def read_strategy_series_from_csv_first_col_time(
    csv_path: str,
    value_col_1_based: int,
    delimiter: str = ";",
) -> dict[datetime, float]:
    """
    Diese Funktion liest eine Zeitreihe aus einer CSV-Datei:
      - Spalte 1: Timestamp
      - value_col_1_based: Wertespalte (1-basiert)
    """
    if not isinstance(value_col_1_based, int) or value_col_1_based < 2:
        raise ValueError("value_col_1_based muss int >= 2 sein.")

    data: dict[datetime, float] = {}

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=delimiter)

        for row in reader:
            if not row or len(row) < value_col_1_based:
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
        raise ValueError(f"Keine gültigen Datenzeilen im CSV gefunden: {csv_path}")

    return data


def floor_to_resolution(dt: datetime, resolution_min: int) -> datetime:
    """
    Diese Funktion rundet einen Zeitstempel auf das Raster der Auflösung ab.
    """
    discard = dt.minute % resolution_min
    return dt.replace(minute=dt.minute - discard, second=0, microsecond=0)


def lookup_signal(strategy_map: dict[datetime, float], ts: datetime, resolution_min: int) -> float | None:
    """
    Diese Funktion sucht einen Signalwert, indem ts vorher auf das Raster abgerundet wird.
    """
    return strategy_map.get(floor_to_resolution(ts, resolution_min), None)


def assert_strategy_csv_covers_simulation(
    strategy_map: dict[datetime, float],
    time_index: list[datetime],
    strategy_resolution_min: int,
    charging_strategy: str,
    strategy_csv_path: str,
) -> None:
    """
    Diese Funktion prüft, ob ein Strategie-CSV den kompletten Simulationszeitraum abdeckt.
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
        raise ValueError(
            "❌ Abbruch: Strategie-CSV deckt den Simulationszeitraum nicht vollständig ab.\n"
            f"Strategie: {charging_strategy}\n"
            f"CSV: {strategy_csv_path}\n"
            f"CSV-Zeitraum: {csv_start} bis {csv_end}\n"
            f"Simulation:   {sim_start} bis {sim_end}\n"
        )

    missing = sorted([t for t in expected_set if t not in strategy_map])
    if missing:
        preview = "\n".join([f"- {t}" for t in missing[:10]])
        raise ValueError(
            "❌ Abbruch: Strategie-CSV hat fehlende Zeitstempel innerhalb des Simulationszeitraums.\n"
            f"Strategie: {charging_strategy}\n"
            f"CSV: {strategy_csv_path}\n"
            f"Fehlende Timestamps: {len(missing)} (erste 10):\n{preview}\n"
        )


def convert_strategy_value_to_internal(
    charging_strategy: str,
    raw_value: float,
    strategy_unit: str,
    step_hours: float,
) -> float:
    """
    Diese Funktion normalisiert CSV-Rohwerte auf interne Einheiten:
      - generation: kW
      - market: €/kWh
    """
    unit = (strategy_unit or "").strip()

    if charging_strategy == "generation":
        if unit == "kW":
            return float(raw_value)
        if unit == "kWh":
            return float(raw_value) / float(step_hours)
        if unit == "MWh":
            return float(raw_value) * 1000.0 / float(step_hours)
        raise ValueError("Unbekannte generation strategy_unit (erlaubt: kW, kWh, MWh).")

    if charging_strategy == "market":
        if unit == "€/kWh":
            return float(raw_value)
        if unit == "€/MWh":
            return float(raw_value) / 1000.0
        raise ValueError("Unbekannte market strategy_unit (erlaubt: €/kWh, €/MWh).")

    return float(raw_value)


def build_base_load_series(
    scenario: dict[str, Any],
    timestamps: list[datetime],
    base_load_resolution_min: int = 15,
) -> np.ndarray | None:
    """
    Diese Funktion baut eine Basislast-Zeitreihe (kW) für den Standort.

    Priorität:
      1) base_load_csv (wenn gesetzt)
      2) base_load_kw (konstant)
      3) sonst: None
    """
    if not timestamps:
        return None

    site_cfg = scenario.get("site", {}) or {}

    base_load_csv = site_cfg.get("base_load_csv", None)
    if base_load_csv:
        base_load_col = site_cfg.get("base_load_value_col", None)
        base_load_unit = str(site_cfg.get("base_load_unit", "") or "").strip()

        if not isinstance(base_load_col, int) or base_load_col < 2:
            raise ValueError("'site.base_load_value_col' fehlt/ungültig (int >= 2).")
        if base_load_unit not in ("kW", "kWh", "MWh"):
            raise ValueError("'site.base_load_unit' muss kW, kWh oder MWh sein.")

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
            else:
                series_kw[i] = float(v) * 1000.0 / step_hours

        return series_kw

    base_load_kw = site_cfg.get("base_load_kw", None)
    if base_load_kw is not None:
        return np.full(len(timestamps), float(base_load_kw), dtype=float)

    return None


# =============================================================================
# 8) Slack & (Market) Slot-Präferenzen
# =============================================================================

def _slack_minutes_for_session(
    s: dict[str, Any],
    ts: datetime,
    rated_power_kw: float,
    charger_efficiency: float,
) -> float:
    """
    Diese Funktion berechnet Slack in Minuten:
      Slack = Restzeit - benötigte Ladezeit bei maximal möglicher Leistung.
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


def _build_preferred_slots_for_market(
    all_sessions: list[dict[str, Any]],
    time_index: list[datetime],
    time_to_idx: dict[datetime, int],
    market_map: dict[datetime, float],
    market_resolution_min: int,
    market_unit: str,
    step_hours: float,
) -> None:
    """
    Diese Funktion baut pro Session eine nach günstigen Preisen sortierte Slot-Liste.

    Konservative Variante A:
      - Es wird nur eine Präferenzreihenfolge erstellt.
      - Bei "verpassten" Slots wird später einfach zum nächstbesten Slot
        (in der Preis-Rangfolge) gewechselt, jedoch nie in der Vergangenheit.
    """
    if not market_map or not time_index:
        return

    for s in all_sessions:
        if float(s.get("energy_required_kwh", 0.0)) <= 0.0:
            continue

        a = floor_to_resolution(s["arrival_time"], market_resolution_min)
        d = floor_to_resolution(s["departure_time"], market_resolution_min)

        while a not in time_to_idx and a < time_index[-1]:
            a += timedelta(minutes=market_resolution_min)
        while d not in time_to_idx and d > time_index[0]:
            d -= timedelta(minutes=market_resolution_min)

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
            raw = lookup_signal(market_map, t, market_resolution_min)
            if raw is None:
                score = 1e30
            else:
                score = float(
                    convert_strategy_value_to_internal(
                        charging_strategy="market",
                        raw_value=float(raw),
                        strategy_unit=market_unit,
                        step_hours=step_hours,
                    )
                )
            scored.append((score, idx))

        scored.sort(key=lambda x: x[0])
        s["preferred_slot_indices"] = [idx for _, idx in scored]
        s["preferred_ptr"] = 0


# =============================================================================
# 9) Power Allocation: Fair-Share (Water-Filling)
# =============================================================================

def _session_step_power_cap_kw(
    s: dict[str, Any],
    rated_power_kw: float,
    time_step_hours: float,
    charger_efficiency: float,
) -> float:
    """
    Diese Funktion bestimmt das effektive Power-Cap einer Session in diesem Zeitschritt.
    """
    if float(s.get("energy_required_kwh", 0.0)) <= 1e-9:
        return 0.0

    vehicle_limit_kw = vehicle_power_at_soc(s)
    hw_limit_kw = float(s.get("max_charging_power_kw", rated_power_kw))
    cap_kw = max(0.0, min(rated_power_kw, vehicle_limit_kw, hw_limit_kw))

    # Zusätzlich wird verhindert, dass mehr Leistung zugeteilt wird als für die Restenergie nötig ist.
    e_need = float(s.get("energy_required_kwh", 0.0))
    p_need_for_full_step = e_need / (time_step_hours * charger_efficiency)
    cap_kw = min(cap_kw, max(0.0, p_need_for_full_step))

    return max(0.0, cap_kw)


def allocate_power_water_filling(
    sessions: list[dict[str, Any]],
    total_budget_kw: float,
    rated_power_kw: float,
    time_step_hours: float,
    charger_efficiency: float,
) -> dict[int, float]:
    """
    Diese Funktion verteilt eine Standort-Leistungsbudget (kW) fair auf Sessions (Water-Filling).

    Fair-Share bedeutet:
      - Jede aktive Session erhält zunächst gleich viel.
      - Falls eine Session ihr Cap erreicht, wird der Rest umverteilt.
      - Das Verfahren wiederholt sich, bis Budget verbraucht ist oder alle Caps erreicht sind.

    Rückgabe:
      - Mapping id(session) -> zugewiesene Leistung (kW)
    """
    alloc: dict[int, float] = {}
    if not sessions or total_budget_kw <= 1e-9:
        return alloc

    caps = {id(s): _session_step_power_cap_kw(s, rated_power_kw, time_step_hours, charger_efficiency) for s in sessions}
    active = [s for s in sessions if caps.get(id(s), 0.0) > 1e-9]
    if not active:
        return alloc

    remaining_budget = float(total_budget_kw)
    remaining = {id(s): float(caps[id(s)]) for s in active}
    current = {id(s): 0.0 for s in active}

    # Iteratives Water-Filling
    for _ in range(50):
        active_ids = [sid for sid, cap in remaining.items() if cap > 1e-9]
        if not active_ids or remaining_budget <= 1e-9:
            break

        per = remaining_budget / float(len(active_ids))
        used_this_round = 0.0

        for sid in active_ids:
            take = min(per, remaining[sid])
            if take <= 1e-12:
                continue
            current[sid] += take
            remaining[sid] -= take
            used_this_round += take

        if used_this_round <= 1e-9:
            break
        remaining_budget = max(0.0, remaining_budget - used_this_round)

    for sid, p in current.items():
        if p > 1e-9:
            alloc[sid] = float(p)

    return alloc


def apply_energy_update(
    ts: datetime,
    sessions: list[dict[str, Any]],
    power_alloc_kw: dict[int, float],
    time_step_hours: float,
    charger_efficiency: float,
    mode_label: str,
) -> float:
    """
    Diese Funktion überführt eine Leistungszuteilung in Energiezuwachs und aktualisiert Sessions.

    Zusätzlich protokolliert sie pro Session, in welchem Modus Energie geliefert wurde.
    Dadurch kann im Notebook später gezählt werden, wie viele Sessions mit
    generation/market/immediate geladen wurden.

    Rückgabe:
      - total_power_kw (Summe der tatsächlich gefahrenen Ladeleistung)
    """
    total_power_kw = 0.0
    mode = (mode_label or "").strip().lower()

    for s in sessions:
        p = float(power_alloc_kw.get(id(s), 0.0))
        s["_actual_power_kw"] = float(p)

        if p <= 1e-9:
            continue

        possible_energy_kwh = p * time_step_hours * charger_efficiency
        e_need = float(s.get("energy_required_kwh", 0.0))

        if possible_energy_kwh >= e_need:
            e_del = e_need
            s["energy_required_kwh"] = 0.0
            if "finished_charging_time" not in s:
                s["finished_charging_time"] = ts

            p = e_del / (time_step_hours * charger_efficiency) if e_del > 0 else 0.0
            s["_actual_power_kw"] = float(p)
        else:
            e_del = possible_energy_kwh
            s["energy_required_kwh"] = e_need - possible_energy_kwh

        s["delivered_energy_kwh"] = float(s.get("delivered_energy_kwh", 0.0)) + float(e_del)
        total_power_kw += float(p)

        # --- NEU: Mode-Tracking pro Session ---
        if "_energy_by_mode_kwh" not in s or not isinstance(s["_energy_by_mode_kwh"], dict):
            s["_energy_by_mode_kwh"] = {}
        s["_energy_by_mode_kwh"][mode] = float(s["_energy_by_mode_kwh"].get(mode, 0.0)) + float(e_del)

        if "_modes_used" not in s or not isinstance(s["_modes_used"], set):
            s["_modes_used"] = set()
        s["_modes_used"].add(mode)

    return float(total_power_kw)



# =============================================================================
# 10) Lademanagement: Plug-In Policy (FCFS, drive_off)
# =============================================================================

def assign_chargers_drive_off_fcfs(
    ts: datetime,
    chargers: list[dict[str, Any] | None],
    all_sessions: list[dict[str, Any]],
    next_arrival_idx: int,
) -> int:
    """
    Diese Funktion verarbeitet Ankünfte nach FCFS (drive_off).

    Regel:
      - Wenn ein Ladepunkt frei ist, wird die Session eingesteckt.
      - Wenn nicht, wird die Session abgewiesen (drive_off).
      - Einmal eingesteckt bleibt das Fahrzeug bis zur Abfahrt eingesteckt.
    """
    while next_arrival_idx < len(all_sessions) and all_sessions[next_arrival_idx]["arrival_time"] <= ts:
        s = all_sessions[next_arrival_idx]
        next_arrival_idx += 1

        if float(s.get("energy_required_kwh", 0.0)) <= 1e-9:
            continue

        s["_rejected"] = False
        free_c = None
        for c in range(len(chargers)):
            if chargers[c] is None:
                free_c = c
                break

        if free_c is not None:
            chargers[free_c] = s
            s["_charger_id"] = free_c
            s["_plug_in_time"] = ts
            continue

        s["_rejected"] = True
        s["_rejection_time"] = ts
        s["_no_charge_reason"] = "no_free_charger_at_arrival"

    return next_arrival_idx


def release_departed_sessions(
    ts: datetime,
    chargers: list[dict[str, Any] | None],
) -> None:
    """
    Diese Funktion gibt Ladepunkte frei, wenn die Session abgefahren ist.
    """
    for c in range(len(chargers)):
        s = chargers[c]
        if s is None:
            continue
        departed = not (s["arrival_time"] <= ts < s["departure_time"])
        if departed:
            chargers[c] = None


def get_present_plugged_sessions(
    ts: datetime,
    chargers: list[dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """
    Diese Funktion liefert alle eingesteckten Sessions, die aktuell anwesend sind und noch Energiebedarf haben.
    """
    plugged_sessions = [s for s in chargers if s is not None]
    present = [
        s for s in plugged_sessions
        if s["arrival_time"] <= ts < s["departure_time"]
        and float(s.get("energy_required_kwh", 0.0)) > 1e-9
    ]
    for s in present:
        s["_actual_power_kw"] = 0.0
    return present


# =============================================================================
# 11) Lademanagement: Immediate
# =============================================================================

def run_step_immediate(
    ts: datetime,
    i: int,
    present_sessions: list[dict[str, Any]],
    grid_limit_p_avb_kw: float,
    rated_power_kw: float,
    time_step_hours: float,
    charger_efficiency: float,
) -> float:
    """
    Diese Funktion implementiert immediate nach FCFS-Plug-In + Fair-Share.

    Regel:
      - Alle anwesenden, eingesteckten Sessions werden berücksichtigt.
      - Standort-Budget = grid_limit_p_avb_kw (import-limit, ohne PV).
      - Verteilung per Water-Filling.
    """
    if not present_sessions:
        return 0.0

    total_budget_kw = max(0.0, float(grid_limit_p_avb_kw))
    alloc = allocate_power_water_filling(
        sessions=present_sessions,
        total_budget_kw=total_budget_kw,
        rated_power_kw=rated_power_kw,
        time_step_hours=time_step_hours,
        charger_efficiency=charger_efficiency,
    )
    total_power_kw = apply_energy_update(
        ts=ts,
        sessions=present_sessions,
        power_alloc_kw=alloc,
        time_step_hours=time_step_hours,
        charger_efficiency=charger_efficiency,
        mode_label="immediate",
    )

    return float(total_power_kw)


# =============================================================================
# 12) Lademanagement: Market (konservativ, vorausschauend via Slot-Präferenzen)
# =============================================================================

def select_sessions_market_or_fallback_to_immediate(
    ts: datetime,
    i: int,
    present_sessions: list[dict[str, Any]],
    rated_power_kw: float,
    charger_efficiency: float,
    emergency_slack_minutes: float,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Diese Funktion wählt Sessions für Market in einem Zeitschritt aus.

    Konservative Variante A:
      - Es wird bevorzugt in den Slot geladen, der aktuell als "nächster" in der Preis-Priorität liegt.
      - Wenn Slots verpasst wurden, wird nur in die Zukunft gesprungen (ptr überspringt idx < i).
      - Wenn kein Vehicle für Market "dran" ist, wird auf Immediate zurückgefallen.

    Rückgabe:
      - selected_sessions
      - did_fallback_to_immediate (True/False)
    """
    if not present_sessions:
        return [], False

    emergency: list[dict[str, Any]] = []
    slot_due: list[dict[str, Any]] = []

    # Slack berechnen (für Emergency + Priorisierung)
    for s in present_sessions:
        slack_m = _slack_minutes_for_session(s, ts, rated_power_kw, charger_efficiency)
        s["_slack_minutes"] = float(slack_m)
        if slack_m <= emergency_slack_minutes:
            emergency.append(s)
            continue

        pref = s.get("preferred_slot_indices", [])
        ptr = int(s.get("preferred_ptr", 0))

        # niemals in Vergangenheit planen
        while ptr < len(pref) and pref[ptr] < i:
            ptr += 1
        s["preferred_ptr"] = ptr

        # Slot ist "due", wenn der aktuelle Index der nächste bevorzugte ist
        if pref and ptr < len(pref) and pref[ptr] == i:
            slot_due.append(s)

    candidates: list[dict[str, Any]] = []
    seen = set()

    for s in emergency:
        sid = id(s)
        if sid not in seen:
            candidates.append(s)
            seen.add(sid)
    for s in slot_due:
        sid = id(s)
        if sid not in seen:
            candidates.append(s)
            seen.add(sid)

    if not candidates:
        # WICHTIG: Market-Fallback zu Immediate (so gewünscht)
        return present_sessions, True

    candidates.sort(key=lambda s: (s.get("_slack_minutes", 1e9), s["departure_time"], s["arrival_time"]))
    return candidates, False


def run_step_market(
    ts: datetime,
    i: int,
    present_sessions: list[dict[str, Any]],
    grid_limit_p_avb_kw: float,
    rated_power_kw: float,
    time_step_hours: float,
    charger_efficiency: float,
    emergency_slack_minutes: float,
) -> tuple[float, bool]:
    """
    Diese Funktion führt einen Market-Zeitschritt aus.

    Ablauf:
      1) Auswahl der Sessions, die jetzt laden sollen (Market-Slots + Emergency).
      2) Falls niemand dran ist: Fallback zu Immediate (für alle Present).
      3) Fair-Share (Water-Filling) auf dem Grid-Budget.
    """
    if not present_sessions:
        return 0.0, False

    selected, fell_back = select_sessions_market_or_fallback_to_immediate(
        ts=ts,
        i=i,
        present_sessions=present_sessions,
        rated_power_kw=rated_power_kw,
        charger_efficiency=charger_efficiency,
        emergency_slack_minutes=emergency_slack_minutes,
    )

    total_budget_kw = max(0.0, float(grid_limit_p_avb_kw))
    alloc = allocate_power_water_filling(
        sessions=selected,
        total_budget_kw=total_budget_kw,
        rated_power_kw=rated_power_kw,
        time_step_hours=time_step_hours,
        charger_efficiency=charger_efficiency,
    )
    mode_for_energy = "immediate" if fell_back else "market"
    total_power_kw = apply_energy_update(
        ts=ts,
        sessions=selected,
        power_alloc_kw=alloc,
        time_step_hours=time_step_hours,
        charger_efficiency=charger_efficiency,
        mode_label=mode_for_energy,
    )

    return float(total_power_kw), bool(fell_back)


# =============================================================================
# 13) Lademanagement: Generation (PV) + Fallback PV -> Market -> Immediate
# =============================================================================

def compute_pv_surplus_kw(
    ts: datetime,
    i: int,
    generation_map: dict[datetime, float],
    generation_unit: str,
    base_load_series_kw: np.ndarray | None,
    strategy_resolution_min: int,
    step_hours_strategy: float,
) -> float:
    """
    Diese Funktion berechnet den PV-Überschuss (kW) als:
      pv_surplus = PV_generation - base_load
    """
    raw = lookup_signal(generation_map, ts, strategy_resolution_min)
    pv_kw = 0.0
    if raw is not None:
        pv_kw = max(
            0.0,
            float(
                convert_strategy_value_to_internal(
                    charging_strategy="generation",
                    raw_value=float(raw),
                    strategy_unit=str(generation_unit),
                    step_hours=step_hours_strategy,
                )
            ),
        )

    base_kw = 0.0
    if base_load_series_kw is not None:
        v = float(base_load_series_kw[i])
        base_kw = 0.0 if np.isnan(v) else max(0.0, v)

    return max(0.0, pv_kw - base_kw)


def run_step_generation_with_fallbacks(
    ts: datetime,
    i: int,
    present_sessions: list[dict[str, Any]],
    pv_surplus_kw_now: float,
    grid_limit_p_avb_kw: float,
    rated_power_kw: float,
    time_step_hours: float,
    charger_efficiency: float,
    emergency_slack_minutes: float,
    # Market fallback signals:
    market_enabled: bool,
    # Für Market-Fallback wird dasselbe Auswahlverhalten wie in "market" genutzt.
) -> tuple[float, str, bool]:
    """
    Diese Funktion führt einen Generation-Zeitschritt aus (PV zuerst, dann Fallbacks).

    Physik/Limit-Definition (wie gewünscht):
      - grid_limit_p_avb_kw ist ein Import-Limit.
      - PV kommt zusätzlich und reduziert den Grid-Import physikalisch.
      - Damit gilt: grid_import = max(0, total_power - pv_surplus_kw_now) <= grid_limit_p_avb_kw
      - also: total_power <= pv_surplus_kw_now + grid_limit_p_avb_kw
      - zusätzlich gelten Charger-Limits pro Session.

    Fallback-Logik:
      1) Zuerst PV-Allocation (priorisiert nach Slack)
      2) Wenn PV nicht reicht und (Market aktiv), wird Market-Fallback angewendet
      3) Wenn Market niemanden lädt (oder Market nicht aktiv), Fallback zu Immediate

    Rückgabe:
      - total_power_kw
      - mode_label: "PV" | "MARKET_FALLBACK" | "IMMEDIATE_FALLBACK"
      - did_fallback_to_immediate
    """
    if not present_sessions:
        return 0.0, "PV", False

    # --- Slack berechnen, da PV bei Mangel nach Slack priorisiert ---
    for s in present_sessions:
        s["_slack_minutes"] = float(_slack_minutes_for_session(s, ts, rated_power_kw, charger_efficiency))

    present_sorted = sorted(
        present_sessions,
        key=lambda s: (s.get("_slack_minutes", 1e9), s["departure_time"], s["arrival_time"])
    )

    # --- 1) PV-only Allocation (greedy nach Slack, um PV maximal "sicher" zu nutzen) ---
    pv_remaining = float(max(0.0, pv_surplus_kw_now))
    pv_alloc: dict[int, float] = {}

    for s in present_sorted:
        if pv_remaining <= 1e-9:
            break
        cap = _session_step_power_cap_kw(s, rated_power_kw, time_step_hours, charger_efficiency)
        if cap <= 1e-9:
            continue
        take = min(cap, pv_remaining)
        if take > 1e-9:
            pv_alloc[id(s)] = float(take)
            pv_remaining -= float(take)

    # PV-Leistung anwenden (nur auf Sessions, die PV bekommen)
    pv_sessions = [s for s in present_sorted if id(s) in pv_alloc]
    pv_power_kw = apply_energy_update(
        ts=ts,
        sessions=pv_sessions,
        power_alloc_kw=pv_alloc,
        time_step_hours=time_step_hours,
        charger_efficiency=charger_efficiency,
        mode_label="generation",
    )


    # --- 2) Prüfen, ob noch Bedarf besteht und ob Grid-Fallback nötig ist ---
    still_need = [s for s in present_sessions if float(s.get("energy_required_kwh", 0.0)) > 1e-9]

    if not still_need:
        return float(pv_power_kw), "PV", False

    # Maximal erlaubte zusätzliche Gesamtleistung aufgrund Import-Limit:
    # grid_import = max(0, total_power - pv_surplus_kw_now) <= grid_limit
    # => total_power <= pv_surplus_kw_now + grid_limit
    total_power_upper = float(max(0.0, pv_surplus_kw_now + float(grid_limit_p_avb_kw)))

    # In dieser Stufe ist bereits pv_power_kw gefahren.
    remaining_total_power_budget = max(0.0, total_power_upper - float(pv_power_kw))
    if remaining_total_power_budget <= 1e-9:
        # Keine Grid-Leistung möglich (Import-Limit bereits ausgeschöpft)
        return float(pv_power_kw), "PV", False

    # --- 3) Market-Fallback oder Immediate-Fallback ---
    if market_enabled:
        # Market-Fallback wird wie Market im Schritt gewählt.
        selected, fell_back = select_sessions_market_or_fallback_to_immediate(
            ts=ts,
            i=i,
            present_sessions=still_need,
            rated_power_kw=rated_power_kw,
            charger_efficiency=charger_efficiency,
            emergency_slack_minutes=emergency_slack_minutes,
        )

        # Hier wird nur das verbleibende Import-abhängige Budget verteilt.
        alloc = allocate_power_water_filling(
            sessions=selected,
            total_budget_kw=remaining_total_power_budget,
            rated_power_kw=rated_power_kw,
            time_step_hours=time_step_hours,
            charger_efficiency=charger_efficiency,
        )
        mode_for_energy = "immediate" if fell_back else "market"
        add_power = apply_energy_update(
            ts=ts,
            sessions=selected,
            power_alloc_kw=alloc,
            time_step_hours=time_step_hours,
            charger_efficiency=charger_efficiency,
            mode_label=mode_for_energy,
        )


        mode = "MARKET_FALLBACK"
        if fell_back:
            mode = "IMMEDIATE_FALLBACK"

        return float(pv_power_kw + add_power), mode, bool(fell_back)

    # Wenn Market nicht aktiv ist: Immediate-Fallback
    alloc = allocate_power_water_filling(
        sessions=still_need,
        total_budget_kw=remaining_total_power_budget,
        rated_power_kw=rated_power_kw,
        time_step_hours=time_step_hours,
        charger_efficiency=charger_efficiency,
    )
    add_power = apply_energy_update(
        ts=ts,
        sessions=still_need,
        power_alloc_kw=alloc,
        time_step_hours=time_step_hours,
        charger_efficiency=charger_efficiency,
        mode_label="immediate",
    )

    return float(pv_power_kw + add_power), "IMMEDIATE_FALLBACK", True


# =============================================================================
# 14) Hauptsimulation (sauber getrennt: Grundmodell vs. Lademanagement)
# =============================================================================

def simulate_load_profile(
    scenario: dict,
    start_datetime: datetime | None = None,
    record_debug: bool = False,
):
    """
    Diese Funktion führt die Lastgangsimulation aus.

    Trennung:
      - Teil A: Grundmodell erzeugen (Sessions/Flotte/Bedarf)
      - Teil B: Lademanagementstrategie anwenden (immediate/market/generation inkl. Fallbacks)

    Rückgabe:
      - time_index
      - load_profile_kw
      - all_charging_sessions (inkl. Ergebnis)
      - charging_count_series
      - holiday_dates
      - charging_strategy
      - strategy_status
      - debug_rows (optional)
    """

    # ------------------------------------------------------------
    # A) Grundmodell
    # ------------------------------------------------------------
    time_index, all_charging_sessions, holiday_dates, _vehicle_profiles = build_base_model(
        scenario=scenario,
        start_datetime=start_datetime,
    )

    if not time_index:
        return ([], np.array([]), [], [], holiday_dates, "immediate", "IMMEDIATE", [] if record_debug else None)

    # ------------------------------------------------------------
    # B) Strategie-Initialisierung (Signale laden)
    # ------------------------------------------------------------
    site_cfg = scenario.get("site", {}) or {}
    charging_strategy = (site_cfg.get("charging_strategy") or "immediate").lower()

    STRATEGY_RESOLUTION_MIN = 15
    strategy_step_hours = STRATEGY_RESOLUTION_MIN / 60.0

    generation_map: dict[datetime, float] | None = None
    generation_unit: str | None = None

    market_map: dict[datetime, float] | None = None
    market_unit: str | None = None

    if charging_strategy == "market":
        market_unit = str(site_cfg.get("market_strategy_unit", "") or "").strip()
        if market_unit not in ("€/MWh", "€/kWh"):
            raise ValueError("'site.market_strategy_unit' muss '€/MWh' oder '€/kWh' sein.")

        market_csv = site_cfg.get("market_strategy_csv", None)
        market_col = site_cfg.get("market_strategy_value_col", None)
        if not market_csv or not isinstance(market_col, int) or market_col < 2:
            raise ValueError("Für market müssen 'market_strategy_csv' und 'market_strategy_value_col'(>=2) gesetzt sein.")

        market_csv_path = resolve_path_relative_to_scenario(scenario, str(market_csv))
        market_map = read_strategy_series_from_csv_first_col_time(market_csv_path, int(market_col), ";")
        assert_strategy_csv_covers_simulation(
            strategy_map=market_map,
            time_index=time_index,
            strategy_resolution_min=STRATEGY_RESOLUTION_MIN,
            charging_strategy="market",
            strategy_csv_path=market_csv_path,
        )

    elif charging_strategy == "generation":
        generation_unit = str(site_cfg.get("generation_strategy_unit", "") or "").strip()
        if generation_unit not in ("kW", "kWh", "MWh"):
            raise ValueError("'site.generation_strategy_unit' muss 'kW', 'kWh' oder 'MWh' sein.")

        gen_csv = site_cfg.get("generation_strategy_csv", None)
        gen_col = site_cfg.get("generation_strategy_value_col", None)
        if not gen_csv or not isinstance(gen_col, int) or gen_col < 2:
            raise ValueError("Für generation müssen 'generation_strategy_csv' und 'generation_strategy_value_col'(>=2) gesetzt sein.")

        gen_csv_path = resolve_path_relative_to_scenario(scenario, str(gen_csv))
        generation_map = read_strategy_series_from_csv_first_col_time(gen_csv_path, int(gen_col), ";")
        assert_strategy_csv_covers_simulation(
            strategy_map=generation_map,
            time_index=time_index,
            strategy_resolution_min=STRATEGY_RESOLUTION_MIN,
            charging_strategy="generation",
            strategy_csv_path=gen_csv_path,
        )

        # Market-Fallback für Generation ist in diesem Setup erlaubt/typisch:
        market_csv = site_cfg.get("market_strategy_csv", None)
        market_col = site_cfg.get("market_strategy_value_col", None)
        market_unit = str(site_cfg.get("market_strategy_unit", "") or "").strip()

        if market_csv and isinstance(market_col, int) and market_col >= 2 and market_unit in ("€/MWh", "€/kWh"):
            market_csv_path = resolve_path_relative_to_scenario(scenario, str(market_csv))
            market_map = read_strategy_series_from_csv_first_col_time(market_csv_path, int(market_col), ";")
            assert_strategy_csv_covers_simulation(
                strategy_map=market_map,
                time_index=time_index,
                strategy_resolution_min=STRATEGY_RESOLUTION_MIN,
                charging_strategy="market_fallback_for_generation",
                strategy_csv_path=market_csv_path,
            )
        else:
            market_map = None
            market_unit = None

    elif charging_strategy == "immediate":
        pass
    else:
        raise ValueError(f"Unbekannte charging_strategy='{charging_strategy}'")

    # ------------------------------------------------------------
    # C) Präferenzlisten für Market aufbauen (auch für PV->Market Fallback)
    # ------------------------------------------------------------
    time_to_idx = {t: idx for idx, t in enumerate(time_index)}
    if market_map is not None and market_unit is not None:
        _build_preferred_slots_for_market(
            all_sessions=all_charging_sessions,
            time_index=time_index,
            time_to_idx=time_to_idx,
            market_map=market_map,
            market_resolution_min=STRATEGY_RESOLUTION_MIN,
            market_unit=str(market_unit),
            step_hours=strategy_step_hours,
        )

    # ------------------------------------------------------------
    # D) Parameter & Ergebniscontainer
    # ------------------------------------------------------------
    time_resolution_min = scenario["time_resolution_min"]
    time_step_hours = time_resolution_min / 60.0
    n_steps = len(time_index)

    load_profile_kw = np.zeros(n_steps, dtype=float)
    charging_count_series: list[int] = []
    debug_rows: list[dict[str, Any]] = []

    grid_limit_p_avb_kw = float(scenario["site"]["grid_limit_p_avb_kw"])
    rated_power_kw = float(scenario["site"]["rated_power_kw"])
    number_of_chargers = int(scenario["site"]["number_chargers"])
    charger_efficiency = float(scenario["site"]["charger_efficiency"])

    strategy_params = site_cfg.get("strategy_params", {}) or {}
    emergency_slack_minutes = float(strategy_params.get("emergency_slack_minutes", 60.0))

    # Basislast nur für generation
    base_load_series_kw = None
    if charging_strategy == "generation":
        base_load_series_kw = build_base_load_series(
            scenario=scenario,
            timestamps=time_index,
            base_load_resolution_min=STRATEGY_RESOLUTION_MIN,
        )

    # ------------------------------------------------------------
    # E) Plug-In Policy (FCFS, drive_off)
    # ------------------------------------------------------------
    chargers: list[dict[str, Any] | None] = [None] * number_of_chargers
    next_arrival_idx = 0

    # ------------------------------------------------------------
    # F) Zeitschritt-Simulation
    # ------------------------------------------------------------
    for i, ts in enumerate(time_index):
        release_departed_sessions(ts, chargers)
        next_arrival_idx = assign_chargers_drive_off_fcfs(
            ts=ts,
            chargers=chargers,
            all_sessions=all_charging_sessions,
            next_arrival_idx=next_arrival_idx,
        )

        present_sessions = get_present_plugged_sessions(ts, chargers)
        charging_count_series.append(len(present_sessions))

        if not present_sessions:
            load_profile_kw[i] = 0.0
            continue

        # --- Strategien anwenden ---
        if charging_strategy == "immediate":
            total_power_kw = run_step_immediate(
                ts=ts,
                i=i,
                present_sessions=present_sessions,
                grid_limit_p_avb_kw=grid_limit_p_avb_kw,
                rated_power_kw=rated_power_kw,
                time_step_hours=time_step_hours,
                charger_efficiency=charger_efficiency,
            )
            load_profile_kw[i] = float(total_power_kw)

            if record_debug:
                debug_rows.append(
                    {
                        "ts": ts,
                        "mode": "IMMEDIATE",
                        "site_total_power_kw": float(total_power_kw),
                        "grid_import_kw_site": float(max(0.0, total_power_kw)),
                    }
                )
            continue

        if charging_strategy == "market":
            total_power_kw, fell_back = run_step_market(
                ts=ts,
                i=i,
                present_sessions=present_sessions,
                grid_limit_p_avb_kw=grid_limit_p_avb_kw,
                rated_power_kw=rated_power_kw,
                time_step_hours=time_step_hours,
                charger_efficiency=charger_efficiency,
                emergency_slack_minutes=emergency_slack_minutes,
            )
            load_profile_kw[i] = float(total_power_kw)

            if record_debug:
                debug_rows.append(
                    {
                        "ts": ts,
                        "mode": "MARKET" if not fell_back else "MARKET->IMMEDIATE",
                        "site_total_power_kw": float(total_power_kw),
                        "grid_import_kw_site": float(max(0.0, total_power_kw)),
                    }
                )
            continue

        # generation
        if charging_strategy == "generation":
            if generation_map is None or generation_unit is None:
                # Wenn PV-Signal nicht verfügbar ist: als Immediate behandeln
                total_power_kw = run_step_immediate(
                    ts=ts,
                    i=i,
                    present_sessions=present_sessions,
                    grid_limit_p_avb_kw=grid_limit_p_avb_kw,
                    rated_power_kw=rated_power_kw,
                    time_step_hours=time_step_hours,
                    charger_efficiency=charger_efficiency,
                )
                load_profile_kw[i] = float(total_power_kw)

                if record_debug:
                    debug_rows.append(
                        {
                            "ts": ts,
                            "mode": "GENERATION_MISSING_SIGNAL->IMMEDIATE",
                            "pv_surplus_kw": 0.0,
                            "site_total_power_kw": float(total_power_kw),
                            "grid_import_kw_site": float(max(0.0, total_power_kw)),
                        }
                    )
                continue

            pv_surplus_kw_now = compute_pv_surplus_kw(
                ts=ts,
                i=i,
                generation_map=generation_map,
                generation_unit=str(generation_unit),
                base_load_series_kw=base_load_series_kw,
                strategy_resolution_min=STRATEGY_RESOLUTION_MIN,
                step_hours_strategy=strategy_step_hours,
            )

            total_power_kw, mode_label, fell_back_immediate = run_step_generation_with_fallbacks(
                ts=ts,
                i=i,
                present_sessions=present_sessions,
                pv_surplus_kw_now=pv_surplus_kw_now,
                grid_limit_p_avb_kw=grid_limit_p_avb_kw,
                rated_power_kw=rated_power_kw,
                time_step_hours=time_step_hours,
                charger_efficiency=charger_efficiency,
                emergency_slack_minutes=emergency_slack_minutes,
                market_enabled=(market_map is not None and market_unit is not None),
            )
            load_profile_kw[i] = float(total_power_kw)

            # Physikalischer Grid-Import:
            grid_import_kw_site = max(0.0, float(total_power_kw) - float(pv_surplus_kw_now))

            if record_debug:
                debug_rows.append(
                    {
                        "ts": ts,
                        "mode": mode_label,
                        "pv_surplus_kw": float(pv_surplus_kw_now),
                        "site_total_power_kw": float(total_power_kw),
                        "grid_import_kw_site": float(grid_import_kw_site),
                        "fell_back_immediate": bool(fell_back_immediate),
                    }
                )

            continue

    # ------------------------------------------------------------
    # G) Strategie-Status
    # ------------------------------------------------------------
    if charging_strategy == "immediate":
        strategy_status = "IMMEDIATE"
    elif charging_strategy == "market":
        strategy_status = "ACTIVE" if market_map else "INACTIVE"
    elif charging_strategy == "generation":
        strategy_status = "ACTIVE" if generation_map else "INACTIVE"
    else:
        strategy_status = "INACTIVE"

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

# =============================================================================
# 15) KPI Helper / Reporting (Notebook)
# =============================================================================

def summarize_sessions_by_charging_mode(
    sessions: list[dict[str, Any]],
    eps_kwh: float = 1e-6,
) -> dict[str, Any]:
    """
    Diese Funktion zählt, wie viele Sessions mindestens einmal mit generation, market
    oder immediate geladen wurden.

    Wichtig:
      - Eine Session kann in mehreren Modi gezählt werden (z.B. PV + Market-Fallback).
      - Gezählt wird nur, wenn tatsächlich Energie > eps_kwh im jeweiligen Modus geliefert wurde.

    Rückgabe:
      - Dictionary mit Session-Anzahlen je Modus sowie Overlap-Informationen.
    """
    modes = ["generation", "market", "immediate"]

    counts = {m: 0 for m in modes}
    sessions_with_any = 0
    sessions_multi_mode = 0

    for s in sessions or []:
        energy_by_mode = s.get("_energy_by_mode_kwh", {}) or {}
        used_modes = [m for m in modes if float(energy_by_mode.get(m, 0.0)) > eps_kwh]

        if not used_modes:
            continue

        sessions_with_any += 1
        for m in used_modes:
            counts[m] += 1
        if len(used_modes) > 1:
            sessions_multi_mode += 1

    return {
        "sessions_with_any_charging": sessions_with_any,
        "sessions_charged_with_generation": counts["generation"],
        "sessions_charged_with_market": counts["market"],
        "sessions_charged_with_immediate": counts["immediate"],
        "sessions_with_multiple_modes": sessions_multi_mode,
    }


def summarize_energy_by_charging_mode(
    sessions: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Diese Funktion summiert die geladene Energie (kWh) getrennt nach Modus
    über alle Sessions.

    Voraussetzung:
      - apply_energy_update(...) hat pro Session _energy_by_mode_kwh geschrieben.
    """
    total = {"generation": 0.0, "market": 0.0, "immediate": 0.0}

    for s in sessions or []:
        energy_by_mode = s.get("_energy_by_mode_kwh", {}) or {}
        for mode in total.keys():
            total[mode] += float(energy_by_mode.get(mode, 0.0))

    return total

