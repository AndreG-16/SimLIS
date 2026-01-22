import yaml
import numpy as np
from pathlib import Path
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Any


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
# 2) Szenario laden (YAML) + Pfadkontext + Validierung der YAML
# =============================================================================

def load_scenario(path: str) -> dict[str, Any]:
    """
    Diese Funktion lädt ein YAML-Szenario und ergänzt den internen Kontextpfad.
    """
    with open(path, "r", encoding="utf-8") as file:
        scenario = yaml.safe_load(file)
    scenario["_scenario_dir"] = str(Path(path).resolve().parent)
    return scenario

def validate_scenario(scenario: dict[str, Any]) -> None:
    """
    Minimal robuste Szenario-Validierung.
    Wirft ValueError mit verständlicher Fehlermeldung, wenn etwas unplausibel ist.
    """
    if not isinstance(scenario, dict):
        raise ValueError("Szenario ist kein dict (YAML fehlerhaft?).")

    # --- Kernfelder ---
    if "time_resolution_min" not in scenario:
        raise ValueError("Fehlend: time_resolution_min")
    if "simulation_horizon_days" not in scenario:
        raise ValueError("Fehlend: simulation_horizon_days")
    if "site" not in scenario or not isinstance(scenario["site"], dict):
        raise ValueError("Fehlend/ungültig: site")

    tr = int(scenario["time_resolution_min"])
    if tr <= 0:
        raise ValueError("time_resolution_min muss > 0 sein.")
    if 60 % tr != 0:
        # nicht zwingend verboten, aber oft ein Zeichen für Misskonfiguration
        raise ValueError("time_resolution_min sollte ein Teiler von 60 sein (z.B. 5, 10, 15).")

    horizon = int(scenario["simulation_horizon_days"])
    if horizon <= 0:
        raise ValueError("simulation_horizon_days muss > 0 sein.")

    site = scenario["site"]
    n_ch = int(site.get("number_chargers", 0))
    if n_ch <= 0:
        raise ValueError("site.number_chargers muss >= 1 sein.")

    rated = float(site.get("rated_power_kw", -1))
    if rated <= 0:
        raise ValueError("site.rated_power_kw muss > 0 sein.")

    grid_lim = float(site.get("grid_limit_p_avb_kw", -1))
    if grid_lim <= 0:
        raise ValueError("site.grid_limit_p_avb_kw muss > 0 sein.")

    eff = float(site.get("charger_efficiency", -1))
    if not (0.0 < eff <= 1.0):
        raise ValueError("site.charger_efficiency muss in (0, 1] liegen.")

    # --- Strategy ---
    strat = str(site.get("charging_strategy", "immediate")).lower()
    if strat not in ("immediate", "market", "generation"):
        raise ValueError("site.charging_strategy muss 'immediate', 'market' oder 'generation' sein.")

    # --- Vehicles ---
    veh = scenario.get("vehicles", {})
    if "soc_target" not in veh:
        raise ValueError("vehicles.soc_target fehlt.")
    soc_target = float(veh["soc_target"])
    if not (0.0 < soc_target <= 1.0):
        raise ValueError("vehicles.soc_target muss in (0, 1] liegen.")

    if "vehicle_curve_csv" not in veh:
        raise ValueError("vehicles.vehicle_curve_csv fehlt.")

    # --- Distributions (minimal checks) ---
    if "arrival_time_distribution" not in scenario:
        raise ValueError("arrival_time_distribution fehlt.")
    if "parking_duration_distribution" not in scenario:
        raise ValueError("parking_duration_distribution fehlt.")
    if "soc_at_arrival_distribution" not in scenario:
        raise ValueError("soc_at_arrival_distribution fehlt.")

    # weekday keys check
    atd = scenario["arrival_time_distribution"]
    if atd.get("type") != "mixture":
        raise ValueError("arrival_time_distribution.type muss 'mixture' sein.")
    if "components_per_weekday" not in atd:
        raise ValueError("arrival_time_distribution.components_per_weekday fehlt.")
    for k in ("working_day", "saturday", "sunday_holiday"):
        if k not in atd["components_per_weekday"]:
            raise ValueError(f"arrival_time_distribution.components_per_weekday.{k} fehlt.")

    # SoC caps plausibility
    sad = scenario["soc_at_arrival_distribution"]
    max_soc = float(sad.get("max_soc", 1.0))
    if not (0.0 < max_soc <= 1.0):
        raise ValueError("soc_at_arrival_distribution.max_soc muss in (0, 1] liegen.")

    # Parkdauer plausibility
    pdd = scenario["parking_duration_distribution"]
    max_d = float(pdd.get("max_duration_minutes", 0))
    min_d = float(pdd.get("min_duration_minutes", 0))
    if max_d <= 0 or min_d <= 0 or max_d < min_d:
        raise ValueError("parking_duration_distribution min/max_duration_minutes ist unplausibel.")


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
    vehicle_profiles: list["VehicleProfile"],
    scenario: dict[str, Any],
) -> "VehicleProfile":
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

                # --- Market Planung via Slot-Reservierung ---
                "market_plan_kw_by_idx": {},           # idx -> reservierte kW
                "_market_planned_energy_kwh": 0.0,
                "_market_remaining_after_plan_kwh": float(required_energy_kwh),


                # Generation/PV Plan (Option 1: Event-basiert):
                "pv_plan_kw_by_idx": {},              # idx -> reservierte kW (nur PV)
                "_pv_planned_energy_kwh": 0.0,
                "_pv_remaining_after_plan_kwh": float(required_energy_kwh),
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

        if "_energy_by_mode_kwh" not in s or not isinstance(s["_energy_by_mode_kwh"], dict):
            s["_energy_by_mode_kwh"] = {}
        s["_energy_by_mode_kwh"][mode] = float(s["_energy_by_mode_kwh"].get(mode, 0.0)) + float(e_del)

        if "_modes_used" not in s or not isinstance(s["_modes_used"], set):
            s["_modes_used"] = set()
        s["_modes_used"].add(mode)

    return float(total_power_kw)


# =============================================================================
# 9b) PV-Planung: Reservierung über den Horizont (Option 1: event-basiert)
# =============================================================================

def _departure_index_exclusive(
    s: dict[str, Any],
    time_index: list[datetime],
    start_idx: int,
) -> int:
    """
    Diese Funktion bestimmt den exklusiven Endindex für die Anwesenheit einer Session.

    Definition:
      - Die Session gilt als anwesend, solange time_index[idx] < departure_time gilt.
      - Der Rückgabewert ist somit der erste Index, der nicht mehr im Fenster liegt.
    """
    dep_ts = s["departure_time"]
    j = start_idx
    n = len(time_index)
    while j < n and time_index[j] < dep_ts:
        j += 1
    return j


def unreserve_pv_plan_for_sessions(
    sessions: list[dict[str, Any]],
    pv_reserved_kw: np.ndarray,
    from_idx_inclusive: int = 0,
) -> None:
    """
    Diese Funktion entfernt (de-reserviert) die PV-Reservierungen der angegebenen Sessions aus pv_reserved_kw.

    Zweck:
      - Das event-basierte Replanning ersetzt alte Pläne vollständig.
      - Damit pv_reserved_kw konsistent bleibt, werden vorherige Reservierungen abgezogen.

    Parameter:
      - from_idx_inclusive erlaubt das selektive Entfernen (z.B. nur Zukunft ab i+1).
    """
    if pv_reserved_kw is None or len(pv_reserved_kw) == 0:
        return

    start = max(0, int(from_idx_inclusive))

    for s in sessions:
        plan = s.get("pv_plan_kw_by_idx", None)
        if not plan or not isinstance(plan, dict):
            s["pv_plan_kw_by_idx"] = {}
            continue

        for idx, p in plan.items():
            try:
                j = int(idx)
            except Exception:
                continue
            if j < start:
                continue
            if 0 <= j < len(pv_reserved_kw):
                pv_reserved_kw[j] = max(0.0, float(pv_reserved_kw[j]) - float(p))

        # Der Plan wird nach dem Unreserve geleert (Replanning schreibt neu).
        s["pv_plan_kw_by_idx"] = {k: v for k, v in (plan.items() if isinstance(plan, dict) else []) if int(k) < start}
        s["_pv_planned_energy_kwh"] = float(s.get("_pv_planned_energy_kwh", 0.0))


def plan_and_reserve_pv_for_session_from_now(
    s: dict[str, Any],
    now_idx: int,
    time_index: list[datetime],
    pv_available_kw: np.ndarray,
    pv_reserved_kw: np.ndarray,
    rated_power_kw: float,
    time_step_hours: float,
    charger_efficiency: float,
) -> None:
    """
    Diese Funktion plant und reserviert PV-Leistung (kW) für eine Session ab now_idx bis zur Abfahrt.

    Planungsprinzip:
      - Sobald PV in einem Zeitschritt verfügbar ist, wird sie genutzt (kein Preisziel).
      - Bei Knappheit entscheidet die Aufrufreihenfolge (Slack-Priorisierung im Replanner).
      - Es wird konservativ mit einem Plan-Cap gearbeitet (Rated/Hardware), ohne SoC-Forecast.

    Ergebnis:
      - s["pv_plan_kw_by_idx"][idx] enthält reservierte PV-kW für idx.
      - pv_reserved_kw[idx] wird entsprechend erhöht.
    """
    e_need = float(s.get("energy_required_kwh", 0.0))
    if e_need <= 1e-9:
        s["pv_plan_kw_by_idx"] = {}
        s["_pv_planned_energy_kwh"] = 0.0
        s["_pv_remaining_after_plan_kwh"] = 0.0
        return

    end_idx = _departure_index_exclusive(s, time_index, now_idx)
    if end_idx <= now_idx:
        s["pv_plan_kw_by_idx"] = {}
        s["_pv_planned_energy_kwh"] = 0.0
        s["_pv_remaining_after_plan_kwh"] = e_need
        return

    # Konservatives Plan-Cap (ohne SoC-Projektion)
    hw_limit_kw = float(s.get("max_charging_power_kw", rated_power_kw))
    plan_cap_kw = max(0.0, min(float(rated_power_kw), hw_limit_kw))

    pv_plan: dict[int, float] = {}
    e_left = float(e_need)

    for idx in range(now_idx, end_idx):
        pv_free = max(0.0, float(pv_available_kw[idx]) - float(pv_reserved_kw[idx]))
        if pv_free <= 1e-9:
            continue
        if plan_cap_kw <= 1e-9:
            break

        p_need = e_left / (time_step_hours * charger_efficiency)
        p_take = min(plan_cap_kw, pv_free, p_need)

        if p_take > 1e-9:
            pv_plan[idx] = float(p_take)
            pv_reserved_kw[idx] += float(p_take)
            e_left -= float(p_take) * time_step_hours * charger_efficiency
            if e_left <= 1e-6:
                break

    s["pv_plan_kw_by_idx"] = pv_plan
    planned = e_need - max(0.0, e_left)
    s["_pv_planned_energy_kwh"] = float(planned)
    s["_pv_remaining_after_plan_kwh"] = float(max(0.0, e_left))


def replan_pv_for_plugged_sessions_on_event(
    ts: datetime,
    now_idx: int,
    time_index: list[datetime],
    chargers: list[dict[str, Any] | None],
    pv_available_kw: np.ndarray,
    pv_reserved_kw: np.ndarray,
    rated_power_kw: float,
    time_step_hours: float,
    charger_efficiency: float,
) -> None:
    """
    Diese Funktion replanted PV-Pläne event-basiert (Option 1).

    Auslöser:
      - Plug-In eines neuen Fahrzeugs (FCFS),
      - Departure/Unplug,
      - Finish-Events oder Netznachladung, die Reservierungen freimachen sollte.

    Vorgehen:
      1) Alle aktuell eingesteckten Sessions sammeln.
      2) Alle bestehenden PV-Reservierungen ab now_idx aus pv_reserved_kw entfernen.
      3) Sessions nach Slack sortieren (kleiner Slack zuerst).
      4) In dieser Reihenfolge PV ab now_idx bis Abfahrt reservieren.

    Damit gilt:
      - Slack-Priorisierung beeinflusst ausschließlich die PV-Planung innerhalb der eingesteckten Sessions.
      - Die FCFS Plug-In Policy bleibt unangetastet.
    """
    plugged = [s for s in chargers if s is not None]
    if not plugged:
        return

    # Alte Reservierungen für die Zukunft ab now_idx entfernen
    unreserve_pv_plan_for_sessions(plugged, pv_reserved_kw, from_idx_inclusive=now_idx)

    for s in plugged:
        if not (s["arrival_time"] <= ts < s["departure_time"]):
            s["_slack_minutes"] = -1e9
        else:
            s["_slack_minutes"] = float(_slack_minutes_for_session(s, ts, rated_power_kw, charger_efficiency))

    plugged_sorted = sorted(
        plugged,
        key=lambda s: (s.get("_slack_minutes", 1e9), s["departure_time"], s["arrival_time"])
    )

    for s in plugged_sorted:
        if not (s["arrival_time"] <= ts < s["departure_time"]):
            continue
        if float(s.get("energy_required_kwh", 0.0)) <= 1e-9:
            s["pv_plan_kw_by_idx"] = {}
            s["_pv_planned_energy_kwh"] = 0.0
            s["_pv_remaining_after_plan_kwh"] = 0.0
            continue

        plan_and_reserve_pv_for_session_from_now(
            s=s,
            now_idx=now_idx,
            time_index=time_index,
            pv_available_kw=pv_available_kw,
            pv_reserved_kw=pv_reserved_kw,
            rated_power_kw=rated_power_kw,
            time_step_hours=time_step_hours,
            charger_efficiency=charger_efficiency,
        )


def clear_future_pv_plan_if_finished(
    sessions: list[dict[str, Any]],
    current_idx: int,
    pv_reserved_kw: np.ndarray,
) -> bool:
    """
    Diese Funktion entfernt zukünftige PV-Reservierungen für Sessions, die bereits fertig geladen sind.

    Hintergrund:
      - Bei event-basierter Planung kann eine Session durch PV oder Grid (Market/Immediate) schneller fertig werden.
      - Ohne Cleanup würden Reservierungen die PV-Verfügbarkeit künstlich reduzieren.

    Rückgabe:
      - True, wenn mindestens eine Reservierung entfernt wurde (Event für Replanning).
    """
    changed = False
    if pv_reserved_kw is None or len(pv_reserved_kw) == 0:
        return False

    start = max(0, int(current_idx) + 1)

    for s in sessions:
        if float(s.get("energy_required_kwh", 0.0)) > 1e-9:
            continue

        plan = s.get("pv_plan_kw_by_idx", {}) or {}
        if not isinstance(plan, dict) or not plan:
            continue

        # Zukunft ab start entfernen
        for idx, p in list(plan.items()):
            try:
                j = int(idx)
            except Exception:
                continue
            if j >= start and 0 <= j < len(pv_reserved_kw):
                pv_reserved_kw[j] = max(0.0, float(pv_reserved_kw[j]) - float(p))
                del plan[idx]
                changed = True

        s["pv_plan_kw_by_idx"] = plan

    return changed


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
) -> int:
    """
    Diese Funktion gibt Ladepunkte frei, wenn die Session abgefahren ist.

    Rückgabe:
      - Anzahl der freigegebenen Ladepunkte (n_departures).
    """
    n_departures = 0
    for c in range(len(chargers)):
        s = chargers[c]
        if s is None:
            continue
        departed = not (s["arrival_time"] <= ts < s["departure_time"])
        if departed:
            chargers[c] = None
            n_departures += 1
    return n_departures


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
# 12) Lademanagement: Market
# =============================================================================

def run_step_market(
    ts: datetime,
    i: int,
    present_sessions: list[dict[str, Any]],
    chargers: list[dict[str, Any] | None],
    time_index: list[datetime],
    market_map: dict[datetime, float],
    market_unit: str,
    market_reserved_kw: np.ndarray,
    grid_limit_p_avb_kw: float,
    rated_power_kw: float,
    time_step_hours: float,
    charger_efficiency: float,
    emergency_slack_minutes: float,
    strategy_resolution_min: int,
    strategy_step_hours: float,
) -> tuple[float, bool, str, bool]:
    """
    Market mit Slot-Reservierung.

    Ablauf pro Schritt:
      1) Geplante Market-Leistung ausführen (Plan pro Session/idx).
      2) Prüfen: wer ist kritisch (Slack <= emergency) ODER hatte gar keinen Plan.
      3) Wenn kritisch/no-plan -> Immediate-Fallback (nur für diese) innerhalb Restbudget.
      4) Optional Cleanup-Event: fertige Sessions geben zukünftige Reservierungen frei.

    Rückgabe:
      - total_power_kw
      - did_fallback_to_immediate
      - mode_label
      - did_finish_or_cleanup_event (damit Simulation replanen kann)
    """
    if not present_sessions:
        return 0.0, False, "MARKET_IDLE", False

    did_fallback = False
    mode_label = "MARKET_PLANNED"
    did_event = False

    # (A) Market-Plan ausführen
    total_power_kw = 0.0
    planned_alloc: dict[int, float] = {}

    for s in present_sessions:
        p_plan = float((s.get("market_plan_kw_by_idx", {}) or {}).get(i, 0.0))
        if p_plan <= 1e-9:
            continue

        p_phys = _session_step_power_cap_kw(s, rated_power_kw, time_step_hours, charger_efficiency)
        p = min(p_plan, p_phys)
        if p > 1e-9:
            planned_alloc[id(s)] = float(p)

    planned_sessions = [s for s in present_sessions if id(s) in planned_alloc]
    if planned_sessions:
        total_power_kw += apply_energy_update(
            ts=ts,
            sessions=planned_sessions,
            power_alloc_kw=planned_alloc,
            time_step_hours=time_step_hours,
            charger_efficiency=charger_efficiency,
            mode_label="market",
        )

    # (B) Critical/no-plan bestimmen
    fallback_candidates: list[dict[str, Any]] = []
    for s in present_sessions:
        slack = float(_slack_minutes_for_session(s, ts, rated_power_kw, charger_efficiency))
        s["_slack_minutes"] = slack

        no_plan = float(s.get("_market_planned_energy_kwh", 0.0)) <= 1e-9
        if no_plan or slack <= emergency_slack_minutes:
            fallback_candidates.append(s)

    # (C) Immediate fallback mit Restbudget
    remaining_budget = max(0.0, float(grid_limit_p_avb_kw) - float(total_power_kw))
    if fallback_candidates and remaining_budget > 1e-9:
        alloc = allocate_power_water_filling(
            sessions=fallback_candidates,
            total_budget_kw=remaining_budget,
            rated_power_kw=rated_power_kw,
            time_step_hours=time_step_hours,
            charger_efficiency=charger_efficiency,
        )
        add_power = apply_energy_update(
            ts=ts,
            sessions=fallback_candidates,
            power_alloc_kw=alloc,
            time_step_hours=time_step_hours,
            charger_efficiency=charger_efficiency,
            mode_label="immediate",
        )
        if add_power > 1e-9:
            did_fallback = True
            mode_label = "MARKET_PLANNED->IMMEDIATE"
            total_power_kw += float(add_power)

    # (D) Cleanup: fertige Sessions geben Reservierungen frei
    plugged = [s for s in chargers if s is not None]
    if clear_future_market_plan_if_finished(plugged, current_idx=i, market_reserved_kw=market_reserved_kw):
        did_event = True

    return float(total_power_kw), bool(did_fallback), str(mode_label), bool(did_event)


# =============================================================================
# 12b) Market-Planung: Slot-Reservierung (NEU)
# =============================================================================

def unreserve_market_plan_for_sessions(
    sessions: list[dict[str, Any]],
    market_reserved_kw: np.ndarray,
    from_idx_inclusive: int = 0,
) -> None:
    """
    Entfernt Market-Reservierungen der Sessions ab from_idx_inclusive aus market_reserved_kw.
    """
    if market_reserved_kw is None or len(market_reserved_kw) == 0:
        return

    start = max(0, int(from_idx_inclusive))

    for s in sessions:
        plan = s.get("market_plan_kw_by_idx", None)
        if not plan or not isinstance(plan, dict):
            s["market_plan_kw_by_idx"] = {}
            continue

        for idx, p in list(plan.items()):
            try:
                j = int(idx)
            except Exception:
                continue
            if j < start:
                continue
            if 0 <= j < len(market_reserved_kw):
                market_reserved_kw[j] = max(0.0, float(market_reserved_kw[j]) - float(p))
                del plan[idx]

        # Plan in der Vergangenheit behalten, Zukunft geleert
        s["market_plan_kw_by_idx"] = plan


def plan_and_reserve_market_for_session_from_now(
    s: dict[str, Any],
    now_idx: int,
    time_index: list[datetime],
    market_map: dict[datetime, float],
    market_resolution_min: int,
    market_unit: str,
    step_hours_strategy: float,
    market_reserved_kw: np.ndarray,
    grid_limit_p_avb_kw: float,
    rated_power_kw: float,
    time_step_hours: float,
    charger_efficiency: float,
) -> None:
    """
    Plant ab now_idx bis Abfahrt die günstigsten Slots und reserviert sie.
    Wenn Slots voll sind => nächstgünstiger Slot.
    Ergebnis steht in s["market_plan_kw_by_idx"].
    """
    e_need = float(s.get("energy_required_kwh", 0.0))
    if e_need <= 1e-9:
        s["market_plan_kw_by_idx"] = {}
        s["_market_planned_energy_kwh"] = 0.0
        s["_market_remaining_after_plan_kwh"] = 0.0
        return

    end_idx = _departure_index_exclusive(s, time_index, now_idx)
    if end_idx <= now_idx:
        s["market_plan_kw_by_idx"] = {}
        s["_market_planned_energy_kwh"] = 0.0
        s["_market_remaining_after_plan_kwh"] = e_need
        return

    # 1) Alle Kandidaten-Slots im Anwesenheitsfenster bewerten
    scored: list[tuple[float, int]] = []
    for idx in range(now_idx, end_idx):
        ts = time_index[idx]
        raw = lookup_signal(market_map, ts, market_resolution_min)
        if raw is None:
            continue
        price = float(
            convert_strategy_value_to_internal(
                charging_strategy="market",
                raw_value=float(raw),
                strategy_unit=str(market_unit),
                step_hours=float(step_hours_strategy),
            )
        )
        scored.append((price, idx))

    scored.sort(key=lambda x: x[0])

    plan: dict[int, float] = {}
    e_left = float(e_need)

    for _, idx in scored:
        if e_left <= 1e-9:
            break

        # freie Grid-Leistung im Slot (site budget)
        free_kw = max(0.0, float(grid_limit_p_avb_kw) - float(market_reserved_kw[idx]))
        if free_kw <= 1e-9:
            continue

        # Session-physikalisches Cap im Schritt (SoC-abhängig)
        p_cap = _session_step_power_cap_kw(s, rated_power_kw, time_step_hours, charger_efficiency)
        if p_cap <= 1e-9:
            break

        # benötigte Leistung um Restenergie zu decken
        p_need = e_left / (time_step_hours * charger_efficiency)
        p_take = min(p_cap, free_kw, p_need)

        if p_take > 1e-9:
            plan[idx] = float(p_take)
            market_reserved_kw[idx] += float(p_take)
            e_left -= float(p_take) * time_step_hours * charger_efficiency

    s["market_plan_kw_by_idx"] = plan
    planned = e_need - max(0.0, e_left)
    s["_market_planned_energy_kwh"] = float(planned)
    s["_market_remaining_after_plan_kwh"] = float(max(0.0, e_left))


def replan_market_on_event(
    ts: datetime,
    now_idx: int,
    time_index: list[datetime],
    chargers: list[dict[str, Any] | None],
    market_map: dict[datetime, float],
    market_resolution_min: int,
    market_unit: str,
    step_hours_strategy: float,
    market_reserved_kw: np.ndarray,
    grid_limit_p_avb_kw: float,
    rated_power_kw: float,
    time_step_hours: float,
    charger_efficiency: float,
) -> None:
    """
    Replanning-Event für Market:
      - alte Reservierungen ab now_idx entfernen
      - plugged Sessions nach Slack priorisieren
      - in Reihenfolge neu reservieren
    """
    plugged = [s for s in chargers if s is not None]
    if not plugged:
        return

    # alte Reservierungen für Zukunft entfernen
    unreserve_market_plan_for_sessions(plugged, market_reserved_kw, from_idx_inclusive=now_idx)

    # Slack priorisiert: kleiner Slack zuerst
    for s in plugged:
        if not (s["arrival_time"] <= ts < s["departure_time"]):
            s["_slack_minutes"] = -1e9
        else:
            s["_slack_minutes"] = float(_slack_minutes_for_session(s, ts, rated_power_kw, charger_efficiency))

    plugged_sorted = sorted(
        plugged,
        key=lambda s: (s.get("_slack_minutes", 1e9), s["departure_time"], s["arrival_time"])
    )

    for s in plugged_sorted:
        if not (s["arrival_time"] <= ts < s["departure_time"]):
            continue
        if float(s.get("energy_required_kwh", 0.0)) <= 1e-9:
            s["market_plan_kw_by_idx"] = {}
            s["_market_planned_energy_kwh"] = 0.0
            s["_market_remaining_after_plan_kwh"] = 0.0
            continue

        plan_and_reserve_market_for_session_from_now(
            s=s,
            now_idx=now_idx,
            time_index=time_index,
            market_map=market_map,
            market_resolution_min=market_resolution_min,
            market_unit=market_unit,
            step_hours_strategy=step_hours_strategy,
            market_reserved_kw=market_reserved_kw,
            grid_limit_p_avb_kw=grid_limit_p_avb_kw,
            rated_power_kw=rated_power_kw,
            time_step_hours=time_step_hours,
            charger_efficiency=charger_efficiency,
        )


def clear_future_market_plan_if_finished(
    sessions: list[dict[str, Any]],
    current_idx: int,
    market_reserved_kw: np.ndarray,
) -> bool:
    """
    Wenn Session fertig ist, werden zukünftige Market-Reservierungen freigegeben.
    Rückgabe: True wenn etwas geändert wurde (kann Replanning triggern).
    """
    changed = False
    if market_reserved_kw is None or len(market_reserved_kw) == 0:
        return False

    start = max(0, int(current_idx) + 1)

    for s in sessions:
        if float(s.get("energy_required_kwh", 0.0)) > 1e-9:
            continue

        plan = s.get("market_plan_kw_by_idx", {}) or {}
        if not isinstance(plan, dict) or not plan:
            continue

        for idx, p in list(plan.items()):
            try:
                j = int(idx)
            except Exception:
                continue
            if j >= start and 0 <= j < len(market_reserved_kw):
                market_reserved_kw[j] = max(0.0, float(market_reserved_kw[j]) - float(p))
                del plan[idx]
                changed = True

        s["market_plan_kw_by_idx"] = plan

    return changed

def _session_step_power_cap_kw_for_energy(
    s: dict[str, Any],
    rated_power_kw: float,
    time_step_hours: float,
    charger_efficiency: float,
    e_need_kwh: float,
) -> float:
    """
    Wie _session_step_power_cap_kw, aber begrenzt auf eine explizite Energierestmenge (kWh),
    z.B. Market-Fallback-Rest nach PV-Plan.
    """
    if float(e_need_kwh) <= 1e-9:
        return 0.0

    vehicle_limit_kw = vehicle_power_at_soc(s)
    hw_limit_kw = float(s.get("max_charging_power_kw", rated_power_kw))
    cap_kw = max(0.0, min(rated_power_kw, vehicle_limit_kw, hw_limit_kw))

    p_need_for_full_step = float(e_need_kwh) / (time_step_hours * charger_efficiency)
    cap_kw = min(cap_kw, max(0.0, p_need_for_full_step))

    return max(0.0, cap_kw)


def plan_and_reserve_market_fallback_energy_from_now(
    s: dict[str, Any],
    now_idx: int,
    time_index: list[datetime],
    market_map: dict[datetime, float],
    market_resolution_min: int,
    market_unit: str,
    step_hours_strategy: float,
    market_reserved_kw: np.ndarray,
    grid_limit_p_avb_kw: float,
    rated_power_kw: float,
    time_step_hours: float,
    charger_efficiency: float,
    fallback_energy_need_kwh: float,
) -> None:
    """
    Plant Market-Slots nur für fallback_energy_need_kwh (Restenergie, die PV nicht deckt).
    Schreibt in s["market_plan_kw_by_idx"] und reserviert in market_reserved_kw.
    """
    e_need = float(fallback_energy_need_kwh)
    if e_need <= 1e-9:
        s["market_plan_kw_by_idx"] = {}
        s["_market_planned_energy_kwh"] = 0.0
        s["_market_remaining_after_plan_kwh"] = 0.0
        return

    end_idx = _departure_index_exclusive(s, time_index, now_idx)
    if end_idx <= now_idx:
        s["market_plan_kw_by_idx"] = {}
        s["_market_planned_energy_kwh"] = 0.0
        s["_market_remaining_after_plan_kwh"] = e_need
        return

    scored: list[tuple[float, int]] = []
    for idx in range(now_idx, end_idx):
        ts = time_index[idx]
        raw = lookup_signal(market_map, ts, market_resolution_min)
        if raw is None:
            continue
        price = float(
            convert_strategy_value_to_internal(
                charging_strategy="market",
                raw_value=float(raw),
                strategy_unit=str(market_unit),
                step_hours=float(step_hours_strategy),
            )
        )
        scored.append((price, idx))

    scored.sort(key=lambda x: x[0])

    plan: dict[int, float] = {}
    e_left = float(e_need)

    for _, idx in scored:
        if e_left <= 1e-9:
            break

        free_kw = max(0.0, float(grid_limit_p_avb_kw) - float(market_reserved_kw[idx]))
        if free_kw <= 1e-9:
            continue

        p_cap = _session_step_power_cap_kw_for_energy(
            s, rated_power_kw, time_step_hours, charger_efficiency, e_left
        )
        if p_cap <= 1e-9:
            break

        p_need = e_left / (time_step_hours * charger_efficiency)
        p_take = min(p_cap, free_kw, p_need)

        if p_take > 1e-9:
            plan[idx] = float(p_take)
            market_reserved_kw[idx] += float(p_take)
            e_left -= float(p_take) * time_step_hours * charger_efficiency

    s["market_plan_kw_by_idx"] = plan
    planned = e_need - max(0.0, e_left)
    s["_market_planned_energy_kwh"] = float(planned)
    s["_market_remaining_after_plan_kwh"] = float(max(0.0, e_left))


def replan_market_fallback_for_generation_on_event(
    ts: datetime,
    now_idx: int,
    time_index: list[datetime],
    chargers: list[dict[str, Any] | None],
    market_map: dict[datetime, float],
    market_resolution_min: int,
    market_unit: str,
    step_hours_strategy: float,
    market_reserved_kw: np.ndarray,
    grid_limit_p_avb_kw: float,
    rated_power_kw: float,
    time_step_hours: float,
    charger_efficiency: float,
) -> None:
    """
    Generation-Hybrid: Market wird NUR für Restenergie nach PV-Plan geplant.
    Voraussetzung: PV-Planung wurde vorher bereits replanned, sodass
    s["_pv_remaining_after_plan_kwh"] stimmt.
    """
    plugged = [s for s in chargers if s is not None]
    if not plugged:
        return

    # alte Reservierungen für Zukunft entfernen
    unreserve_market_plan_for_sessions(plugged, market_reserved_kw, from_idx_inclusive=now_idx)

    # Slack priorisiert (wie bei Market)
    for s in plugged:
        if not (s["arrival_time"] <= ts < s["departure_time"]):
            s["_slack_minutes"] = -1e9
        else:
            s["_slack_minutes"] = float(_slack_minutes_for_session(s, ts, rated_power_kw, charger_efficiency))

    plugged_sorted = sorted(
        plugged,
        key=lambda s: (s.get("_slack_minutes", 1e9), s["departure_time"], s["arrival_time"])
    )

    for s in plugged_sorted:
        if not (s["arrival_time"] <= ts < s["departure_time"]):
            continue

        # Market-Fallback nur für Rest nach PV-Plan:
        fallback_need = float(s.get("_pv_remaining_after_plan_kwh", 0.0))
        if fallback_need <= 1e-9:
            s["market_plan_kw_by_idx"] = {}
            s["_market_planned_energy_kwh"] = 0.0
            s["_market_remaining_after_plan_kwh"] = 0.0
            continue

        plan_and_reserve_market_fallback_energy_from_now(
            s=s,
            now_idx=now_idx,
            time_index=time_index,
            market_map=market_map,
            market_resolution_min=market_resolution_min,
            market_unit=str(market_unit),
            step_hours_strategy=step_hours_strategy,
            market_reserved_kw=market_reserved_kw,
            grid_limit_p_avb_kw=grid_limit_p_avb_kw,
            rated_power_kw=rated_power_kw,
            time_step_hours=time_step_hours,
            charger_efficiency=charger_efficiency,
            fallback_energy_need_kwh=fallback_need,
        )



# =============================================================================
# 13) Lademanagement: Generation (PV) – PV-Planung + Critical-only Fallback
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

def split_critical_sessions_market_vs_immediate(
    ts: datetime,
    critical_sessions: list[dict[str, Any]],
    market_enabled: bool,
    market_map: dict[datetime, float] | None,
    market_unit: str | None,
    strategy_resolution_min: int,
    strategy_step_hours: float,
    rated_power_kw: float,
    charger_efficiency: float,
    hard_immediate_slack_minutes: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """
    Entscheidet im Generation-Mode (critical-only Grid-Fallback), welche Sessions
    als 'market' gelabelt werden können und welche zwingend 'immediate' brauchen.

    Market gilt als NICHT umsetzbar für eine Session, wenn z.B.:
      - kein Preis-Signal vorhanden ist
      - Slack <= hard_immediate_slack_minutes (z.B. <= 0) => jetzt sofort laden

    Rückgabe:
      - market_sessions
      - immediate_sessions
      - any_market_used (bool)
    """
    if not critical_sessions:
        return [], [], False

    if not market_enabled or market_map is None or market_unit is None:
        return [], list(critical_sessions), False

    # Market-Signal muss für den aktuellen Zeitpunkt existieren
    raw_price = lookup_signal(market_map, ts, strategy_resolution_min)
    if raw_price is None:
        return [], list(critical_sessions), False

    # Preis-Konvertierung (nur als Plausibilitätscheck; wir nutzen hier keine Reservierung/Planung)
    try:
        _ = convert_strategy_value_to_internal(
            charging_strategy="market",
            raw_value=float(raw_price),
            strategy_unit=str(market_unit),
            step_hours=float(strategy_step_hours),
        )
    except Exception:
        return [], list(critical_sessions), False

    market_sessions: list[dict[str, Any]] = []
    immediate_sessions: list[dict[str, Any]] = []

    for s in critical_sessions:
        slack = float(s.get("_slack_minutes", _slack_minutes_for_session(s, ts, rated_power_kw, charger_efficiency)))
        s["_slack_minutes"] = slack

        # "Market nicht umsetzbar" => harte Regel: sofort laden
        if slack <= float(hard_immediate_slack_minutes):
            immediate_sessions.append(s)
        else:
            market_sessions.append(s)

    any_market_used = len(market_sessions) > 0
    return market_sessions, immediate_sessions, any_market_used


def run_step_generation_planned_pv_with_critical_fallback(
    ts: datetime,
    i: int,
    time_index: list[datetime],
    present_sessions: list[dict[str, Any]],
    pv_surplus_kw_now: float,
    pv_reserved_kw: np.ndarray,
    grid_limit_p_avb_kw: float,
    rated_power_kw: float,
    time_step_hours: float,
    charger_efficiency: float,
    emergency_slack_minutes: float,
    market_enabled: bool,
    market_map: dict[datetime, float] | None,
    market_unit: str | None,
    strategy_resolution_min: int,
    strategy_step_hours: float,
    hard_immediate_slack_minutes: float = 0.0,   # <-- neu: "Market nicht umsetzbar" Schwelle
) -> tuple[float, str, bool, bool, bool]:
    """
    Generation-Hybrid (PV max) + Market NUR als Fallback:

    Schritt:
      1) PV nur aus PV-Reservierung (pv_plan_kw_by_idx)
      2) Market-Fallback aus Market-Reservierung (market_plan_kw_by_idx) für Restenergie nach PV-Plan
         -> ABER: wenn Slack <= hard_immediate_slack_minutes => Market für diese Session nicht mehr umsetzbar
      3) Immediate nur als Notfall (kritisch: Slack <= emergency_slack_minutes) oder Market nicht mehr umsetzbar

    Physik:
      grid_import = max(0, total_power - pv_surplus_kw_now) <= grid_limit_p_avb_kw
      => total_power <= pv_surplus_kw_now + grid_limit_p_avb_kw
    """
    if not present_sessions:
        return 0.0, "PV_ONLY", False, False, False

    did_finish_event = False
    did_use_grid = False
    did_fallback_immediate = False

    # Upper bound (PV + Grid Importlimit)
    total_power_upper = float(max(0.0, float(pv_surplus_kw_now) + float(grid_limit_p_avb_kw)))

    # ------------------------------------------------------------------
    # 1) PV-Ausführung (nur reservierte PV, Slack-priorisiert)
    # ------------------------------------------------------------------
    pv_budget = float(max(0.0, pv_surplus_kw_now))
    pv_alloc: dict[int, float] = {}

    for s in present_sessions:
        s["_slack_minutes"] = float(_slack_minutes_for_session(s, ts, rated_power_kw, charger_efficiency))

    pv_candidates: list[tuple[dict[str, Any], float]] = []
    for s in present_sessions:
        plan = s.get("pv_plan_kw_by_idx", {}) or {}
        p_plan = float(plan.get(i, 0.0))
        if p_plan <= 1e-9:
            continue

        p_phys = _session_step_power_cap_kw(s, rated_power_kw, time_step_hours, charger_efficiency)
        p_cap = max(0.0, min(p_plan, p_phys))
        if p_cap <= 1e-9:
            continue

        pv_candidates.append((s, p_cap))

    pv_candidates.sort(key=lambda x: (x[0].get("_slack_minutes", 1e9), x[0]["departure_time"], x[0]["arrival_time"]))

    for s, cap in pv_candidates:
        if pv_budget <= 1e-9:
            break
        take = min(float(cap), pv_budget)
        if take > 1e-9:
            pv_alloc[id(s)] = float(take)
            pv_budget -= float(take)

    pv_sessions = [s for s in present_sessions if id(s) in pv_alloc]
    pv_power_kw = apply_energy_update(
        ts=ts,
        sessions=pv_sessions,
        power_alloc_kw=pv_alloc,
        time_step_hours=time_step_hours,
        charger_efficiency=charger_efficiency,
        mode_label="generation",
    )

    for s in pv_sessions:
        if float(s.get("energy_required_kwh", 0.0)) <= 1e-9 and "finished_charging_time" in s and s["finished_charging_time"] == ts:
            did_finish_event = True

    # Wenn nach PV niemand mehr Bedarf hat -> fertig
    still_need = [s for s in present_sessions if float(s.get("energy_required_kwh", 0.0)) > 1e-9]
    if not still_need:
        return float(pv_power_kw), "PV_ONLY", False, False, did_finish_event

    remaining_total_power_budget = max(0.0, total_power_upper - float(pv_power_kw))
    if remaining_total_power_budget <= 1e-9:
        return float(pv_power_kw), "PV_ONLY", False, False, did_finish_event

    # ------------------------------------------------------------------
    # 2) Market-Fallback AUSFÜHREN (nicht nur "critical"!)
    #    Aber: wenn Market "nicht umsetzbar" (Slack <= hard_immediate_slack_minutes) -> nicht Market, sondern ggf. Immediate
    # ------------------------------------------------------------------
    market_power_kw = 0.0
    market_sessions_used: list[dict[str, Any]] = []
    immediate_due_to_hard: list[dict[str, Any]] = []

    if market_enabled and market_map is not None and market_unit is not None:
        planned_alloc: dict[int, float] = {}
        remaining = float(remaining_total_power_budget)

        # Slack-priorisiert ausführen (aber nicht nur critical)
        need_sorted = sorted(
            still_need,
            key=lambda s: (s.get("_slack_minutes", 1e9), s["departure_time"], s["arrival_time"])
        )

        for s in need_sorted:
            if remaining <= 1e-9:
                break

            slack = float(s.get("_slack_minutes", _slack_minutes_for_session(s, ts, rated_power_kw, charger_efficiency)))
            s["_slack_minutes"] = slack

            # Market für diese Session "nicht umsetzbar" => wird später ggf. Immediate bekommen
            if slack <= float(hard_immediate_slack_minutes):
                immediate_due_to_hard.append(s)
                continue

            p_plan = float((s.get("market_plan_kw_by_idx", {}) or {}).get(i, 0.0))
            if p_plan <= 1e-9:
                continue

            p_phys = _session_step_power_cap_kw(s, rated_power_kw, time_step_hours, charger_efficiency)
            p_take = min(p_plan, p_phys, remaining)
            if p_take > 1e-9:
                planned_alloc[id(s)] = float(p_take)
                remaining -= float(p_take)

        planned_sessions = [s for s in still_need if id(s) in planned_alloc]
        if planned_sessions:
            market_power_kw = apply_energy_update(
                ts=ts,
                sessions=planned_sessions,
                power_alloc_kw=planned_alloc,
                time_step_hours=time_step_hours,
                charger_efficiency=charger_efficiency,
                mode_label="market",
            )
            if market_power_kw > 1e-9:
                did_use_grid = True
                market_sessions_used = planned_sessions

    remaining_after_market = max(0.0, float(remaining_total_power_budget) - float(market_power_kw))
    if remaining_after_market <= 1e-9:
        # Kein Budget mehr für Grid-Notfall
        total_power_kw = float(pv_power_kw + market_power_kw)
        mode = "PV_ONLY->MARKET" if market_power_kw > 1e-9 else "PV_ONLY"
        return total_power_kw, mode, False, did_use_grid, did_finish_event

    # ------------------------------------------------------------------
    # 3) Immediate-Notfall:
    #    - nur für CRITICAL (Slack <= emergency_slack_minutes)
    #    - plus die "hard" Fälle (Market nicht umsetzbar)
    # ------------------------------------------------------------------
    critical_or_hard: list[dict[str, Any]] = []
    for s in still_need:
        slack = float(s.get("_slack_minutes", _slack_minutes_for_session(s, ts, rated_power_kw, charger_efficiency)))
        s["_slack_minutes"] = slack
        if slack <= float(emergency_slack_minutes):
            critical_or_hard.append(s)

    # hard rule adden (Market nicht umsetzbar)
    for s in immediate_due_to_hard:
        if s not in critical_or_hard:
            critical_or_hard.append(s)

    add_power = 0.0
    if critical_or_hard and remaining_after_market > 1e-9:
        alloc = allocate_power_water_filling(
            sessions=critical_or_hard,
            total_budget_kw=remaining_after_market,
            rated_power_kw=rated_power_kw,
            time_step_hours=time_step_hours,
            charger_efficiency=charger_efficiency,
        )
        add_power = apply_energy_update(
            ts=ts,
            sessions=critical_or_hard,
            power_alloc_kw=alloc,
            time_step_hours=time_step_hours,
            charger_efficiency=charger_efficiency,
            mode_label="immediate",
        )
        if add_power > 1e-9:
            did_use_grid = True
            did_fallback_immediate = True

    for s in present_sessions:
        if float(s.get("energy_required_kwh", 0.0)) <= 1e-9 and "finished_charging_time" in s and s["finished_charging_time"] == ts:
            did_finish_event = True

    total_power_kw = float(pv_power_kw + market_power_kw + add_power)

    # Labeling
    if market_power_kw > 1e-9 and add_power > 1e-9:
        mode_label = "PV->MARKET->IMMEDIATE"
    elif market_power_kw > 1e-9:
        mode_label = "PV->MARKET"
    elif add_power > 1e-9:
        mode_label = "PV->IMMEDIATE"
    else:
        mode_label = "PV_ONLY"

    return total_power_kw, mode_label, bool(did_fallback_immediate), bool(did_use_grid), bool(did_finish_event)


    # ------------------------------------------------------------------
    # 3) Market-Fallback (geplant) für critical: günstige Slots, aber wenn "zu spät" -> Immediate
    # ------------------------------------------------------------------
    mode_label = "CRITICAL_ONLY"
    total_power_upper = float(max(0.0, float(pv_surplus_kw_now) + float(grid_limit_p_avb_kw)))
    remaining_total_power_budget = max(0.0, total_power_upper - float(pv_power_kw))

    if remaining_total_power_budget <= 1e-9:
        return float(pv_power_kw), "PV_ONLY", False, False, did_finish_event

    # Critical bestimmen
    critical = []
    for s in still_need:
        slack_m = float(s.get("_slack_minutes", _slack_minutes_for_session(s, ts, rated_power_kw, charger_efficiency)))
        s["_slack_minutes"] = slack_m
        if slack_m <= emergency_slack_minutes:
            critical.append(s)

    if not critical:
        # Nicht-kritische Sessions bleiben PV-only (max PV bleibt)
        return float(pv_power_kw), "PV_ONLY", False, False, did_finish_event

    # 3a) Market-Plan (Fallback) zuerst ausführen – aber nur für critical (und nur wenn market_enabled)
    market_power_kw = 0.0
    if market_enabled:
        planned_alloc: dict[int, float] = {}
        # Slack-priorisierte Ausführung der geplanten Slot-Leistung in diesem Schritt
        crit_sorted = sorted(
            critical,
            key=lambda s: (s.get("_slack_minutes", 1e9), s["departure_time"], s["arrival_time"])
        )

        remaining = float(remaining_total_power_budget)
        for s in crit_sorted:
            if remaining <= 1e-9:
                break
            p_plan = float((s.get("market_plan_kw_by_idx", {}) or {}).get(i, 0.0))
            if p_plan <= 1e-9:
                continue

            # Physik-Cap
            p_phys = _session_step_power_cap_kw(s, rated_power_kw, time_step_hours, charger_efficiency)
            p_take = min(p_plan, p_phys, remaining)
            if p_take > 1e-9:
                planned_alloc[id(s)] = float(p_take)
                remaining -= float(p_take)

        planned_sessions = [s for s in critical if id(s) in planned_alloc]
        if planned_sessions:
            market_power_kw = apply_energy_update(
                ts=ts,
                sessions=planned_sessions,
                power_alloc_kw=planned_alloc,
                time_step_hours=time_step_hours,
                charger_efficiency=charger_efficiency,
                mode_label="market",
            )
            if market_power_kw > 1e-9:
                did_use_grid = True

    # 3b) Immediate-Notfall: wenn nach Market-Plan critical noch Bedarf haben (Slack klein => jetzt laden)
    remaining_after_market = max(0.0, float(remaining_total_power_budget) - float(market_power_kw))
    critical_still_need = [s for s in critical if float(s.get("energy_required_kwh", 0.0)) > 1e-9]

    add_power = 0.0
    if critical_still_need and remaining_after_market > 1e-9:
        alloc = allocate_power_water_filling(
            sessions=critical_still_need,
            total_budget_kw=remaining_after_market,
            rated_power_kw=rated_power_kw,
            time_step_hours=time_step_hours,
            charger_efficiency=charger_efficiency,
        )
        add_power = apply_energy_update(
            ts=ts,
            sessions=critical_still_need,
            power_alloc_kw=alloc,
            time_step_hours=time_step_hours,
            charger_efficiency=charger_efficiency,
            mode_label="immediate",
        )
        if add_power > 1e-9:
            did_use_grid = True
            did_fallback_immediate = True

    # Finish-Event prüfen
    for s in critical:
        if float(s.get("energy_required_kwh", 0.0)) <= 1e-9 and "finished_charging_time" in s and s["finished_charging_time"] == ts:
            did_finish_event = True

    total_power_kw = float(pv_power_kw + float(market_power_kw) + float(add_power))

    if market_enabled and market_power_kw > 1e-9 and add_power <= 1e-9:
        mode_label = "CRITICAL->MARKET"
    elif market_enabled and add_power > 1e-9:
        mode_label = "CRITICAL->MARKET->IMMEDIATE"
    else:
        mode_label = "CRITICAL->IMMEDIATE"

    return total_power_kw, mode_label, bool(did_fallback_immediate), bool(did_use_grid), bool(did_finish_event)


# =============================================================================
# 14) Hauptsimulation
# =============================================================================

def simulate_load_profile(
    scenario: dict,
    start_datetime: datetime | None = None,
    record_debug: bool = False,
    record_charger_traces: bool = False,
    emergency_slack_minutes: float = 60.0,
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
    validate_scenario(scenario)

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

        # Market-Fallback für Generation ist erlaubt, wenn Market-Signal vorhanden ist.
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
    # D) Parameter & Ergebniscontainer
    # ------------------------------------------------------------
    time_resolution_min = scenario["time_resolution_min"]
    time_step_hours = time_resolution_min / 60.0
    n_steps = len(time_index)

    # Market-Reservierungen (auch für Generation-Fallback)
    market_reserved_kw = None
    if market_map is not None and market_unit is not None:
        # Market wird entweder als Hauptstrategie genutzt ODER als Fallback in Generation
        market_reserved_kw = np.zeros(n_steps, dtype=float)

    load_profile_kw = np.zeros(n_steps, dtype=float)
    charging_count_series: list[int] = []
    debug_rows: list[dict[str, Any]] = []
    charger_trace_rows: list[dict[str, Any]] = []

    grid_limit_p_avb_kw = float(scenario["site"]["grid_limit_p_avb_kw"])
    rated_power_kw = float(scenario["site"]["rated_power_kw"])
    number_of_chargers = int(scenario["site"]["number_chargers"])
    charger_efficiency = float(scenario["site"]["charger_efficiency"])

    emergency_slack_minutes = float(emergency_slack_minutes)

    # Basislast immer (für korrekten Netzimport = EV + Grundlast)
    base_load_series_kw = build_base_load_series(
        scenario=scenario,
        timestamps=time_index,
        base_load_resolution_min=STRATEGY_RESOLUTION_MIN,
    )
    if base_load_series_kw is None:
        base_load_series_kw = np.zeros(n_steps, dtype=float)


    # PV-Serien nur für generation
    pv_available_kw = None
    pv_reserved_kw = None
    if charging_strategy == "generation" and generation_map is not None and generation_unit is not None:
        pv_available_kw = np.zeros(n_steps, dtype=float)
        pv_reserved_kw = np.zeros(n_steps, dtype=float)
        for i0, ts0 in enumerate(time_index):
            pv_available_kw[i0] = float(
                compute_pv_surplus_kw(
                    ts=ts0,
                    i=i0,
                    generation_map=generation_map,
                    generation_unit=str(generation_unit),
                    base_load_series_kw=base_load_series_kw,
                    strategy_resolution_min=STRATEGY_RESOLUTION_MIN,
                    step_hours_strategy=strategy_step_hours,
                )
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
        # --------------------------------------------------------
        # 1) Plug-In / Unplug Events
        # --------------------------------------------------------
        plugged_before = sum(1 for c in chargers if c is not None)

        # Departures: Ladepunkte freigeben
        n_departures = release_departed_sessions(ts, chargers)
        plugged_after_release = sum(1 for c in chargers if c is not None)
        did_unplug_event = (plugged_after_release < plugged_before) or (n_departures > 0)

        # Ankünfte: FCFS Plug-In (drive_off)
        plugged_before_assign = plugged_after_release
        next_arrival_idx = assign_chargers_drive_off_fcfs(
            ts=ts,
            chargers=chargers,
            all_sessions=all_charging_sessions,
            next_arrival_idx=next_arrival_idx,
        )
        plugged_after_assign = sum(1 for c in chargers if c is not None)
        did_plug_event = (plugged_after_assign > plugged_before_assign)

        # Event-basiertes Replanning (Generation) bei Änderung der Plug-Menge
        if charging_strategy == "generation" and pv_available_kw is not None and pv_reserved_kw is not None:
            if did_plug_event or did_unplug_event:
                replan_pv_for_plugged_sessions_on_event(
                    ts=ts,
                    now_idx=i,
                    time_index=time_index,
                    chargers=chargers,
                    pv_available_kw=pv_available_kw,
                    pv_reserved_kw=pv_reserved_kw,
                    rated_power_kw=rated_power_kw,
                    time_step_hours=time_step_hours,
                    charger_efficiency=charger_efficiency,
                )
        
        # Market-Fallback-Replanning (Generation-Hybrid): NUR für Restenergie nach PV-Plan
        if charging_strategy == "generation" and market_map is not None and market_unit is not None and market_reserved_kw is not None:
            if did_plug_event or did_unplug_event:
                replan_market_fallback_for_generation_on_event(
                    ts=ts,
                    now_idx=i,
                    time_index=time_index,
                    chargers=chargers,
                    market_map=market_map,
                    market_resolution_min=STRATEGY_RESOLUTION_MIN,
                    market_unit=str(market_unit),
                    step_hours_strategy=strategy_step_hours,
                    market_reserved_kw=market_reserved_kw,
                    grid_limit_p_avb_kw=grid_limit_p_avb_kw,
                    rated_power_kw=rated_power_kw,
                    time_step_hours=time_step_hours,
                    charger_efficiency=charger_efficiency,
                )


        # Event-basiertes Replanning (Market) bei Plug/Unplug
        if charging_strategy == "market" and market_map is not None and market_unit is not None and market_reserved_kw is not None:
            if did_plug_event or did_unplug_event:
                replan_market_on_event(
                    ts=ts,
                    now_idx=i,
                    time_index=time_index,
                    chargers=chargers,
                    market_map=market_map,
                    market_resolution_min=STRATEGY_RESOLUTION_MIN,
                    market_unit=str(market_unit),
                    step_hours_strategy=strategy_step_hours,
                    market_reserved_kw=market_reserved_kw,
                    grid_limit_p_avb_kw=grid_limit_p_avb_kw,
                    rated_power_kw=rated_power_kw,
                    time_step_hours=time_step_hours,
                    charger_efficiency=charger_efficiency,
                )


        # Sessions, die jetzt physisch anwesend sind + noch Bedarf haben
        present_sessions = get_present_plugged_sessions(ts, chargers)
        charging_count_series.append(len(present_sessions))

        # --------------------------------------------------------
        # 2) Defaults für diesen Zeitschritt (wichtig: keine continues)
        # --------------------------------------------------------
        total_power_kw = 0.0

        # Debug/Labels
        mode_label_for_debug = "IDLE"
        fell_back_market_to_immediate = False

        # Generation-spezifisch
        pv_surplus_kw_now = 0.0
        grid_import_kw_site = 0.0
        did_use_grid = False
        did_finish_event = False
        did_cleanup_event = False

        # --------------------------------------------------------
        # 3) Strategien anwenden
        # --------------------------------------------------------
        if charging_strategy == "immediate":
            if present_sessions:
                total_power_kw = run_step_immediate(
                    ts=ts,
                    i=i,
                    present_sessions=present_sessions,
                    grid_limit_p_avb_kw=grid_limit_p_avb_kw,
                    rated_power_kw=rated_power_kw,
                    time_step_hours=time_step_hours,
                    charger_efficiency=charger_efficiency,
                )
            mode_label_for_debug = "IMMEDIATE"
            grid_import_kw_site = max(0.0, float(total_power_kw))

        elif charging_strategy == "market":
            did_market_event = False

            missing_signal = (market_map is None or market_unit is None or market_reserved_kw is None)
            if missing_signal:
                # ohne Market-Signal => immediate
                if present_sessions:
                    total_power_kw = run_step_immediate(
                        ts=ts,
                        i=i,
                        present_sessions=present_sessions,
                        grid_limit_p_avb_kw=grid_limit_p_avb_kw,
                        rated_power_kw=rated_power_kw,
                        time_step_hours=time_step_hours,
                        charger_efficiency=charger_efficiency,
                    )
                mode_label_for_debug = "MARKET_MISSING_SIGNAL->IMMEDIATE"
                grid_import_kw_site = max(0.0, float(total_power_kw))

            else:
                if present_sessions:
                    total_power_kw, fell_back_market_to_immediate, mode_label_for_debug, did_market_event = run_step_market(
                        ts=ts,
                        i=i,
                        present_sessions=present_sessions,
                        chargers=chargers,
                        time_index=time_index,
                        market_map=market_map,
                        market_unit=str(market_unit),
                        market_reserved_kw=market_reserved_kw,
                        grid_limit_p_avb_kw=grid_limit_p_avb_kw,
                        rated_power_kw=rated_power_kw,
                        time_step_hours=time_step_hours,
                        charger_efficiency=charger_efficiency,
                        emergency_slack_minutes=emergency_slack_minutes,
                        strategy_resolution_min=STRATEGY_RESOLUTION_MIN,
                        strategy_step_hours=strategy_step_hours,
                    )
                else:
                    mode_label_for_debug = "MARKET_IDLE"
                    fell_back_market_to_immediate = False
                    total_power_kw = 0.0

                grid_import_kw_site = max(0.0, float(total_power_kw))

                # Wenn Cleanup/Finish Reservierungen geändert hat => ab nächstem Schritt replanen
                if did_market_event and (i + 1) < n_steps:
                    replan_market_on_event(
                        ts=ts,
                        now_idx=i + 1,
                        time_index=time_index,
                        chargers=chargers,
                        market_map=market_map,
                        market_resolution_min=STRATEGY_RESOLUTION_MIN,
                        market_unit=str(market_unit),
                        step_hours_strategy=strategy_step_hours,
                        market_reserved_kw=market_reserved_kw,
                        grid_limit_p_avb_kw=grid_limit_p_avb_kw,
                        rated_power_kw=rated_power_kw,
                        time_step_hours=time_step_hours,
                        charger_efficiency=charger_efficiency,
                    )

        elif charging_strategy == "generation":
            # Fehlende PV-Signale => als Immediate behandeln
            missing_signal = (
                generation_map is None
                or generation_unit is None
                or pv_available_kw is None
                or pv_reserved_kw is None
            )

            if missing_signal:
                if present_sessions:
                    total_power_kw = run_step_immediate(
                        ts=ts,
                        i=i,
                        present_sessions=present_sessions,
                        grid_limit_p_avb_kw=grid_limit_p_avb_kw,
                        rated_power_kw=rated_power_kw,
                        time_step_hours=time_step_hours,
                        charger_efficiency=charger_efficiency,
                    )
                mode_label_for_debug = "GENERATION_MISSING_SIGNAL->IMMEDIATE"
                pv_surplus_kw_now = 0.0
                grid_import_kw_site = max(0.0, float(total_power_kw))

            else:
                pv_surplus_kw_now = float(pv_available_kw[i])

                if present_sessions:
                    total_power_kw, mode_label_for_debug, _fell_back_immediate, did_use_grid, did_finish_event = (
                        run_step_generation_planned_pv_with_critical_fallback(
                            ts=ts,
                            i=i,
                            time_index=time_index,
                            present_sessions=present_sessions,
                            pv_surplus_kw_now=pv_surplus_kw_now,
                            pv_reserved_kw=pv_reserved_kw,
                            grid_limit_p_avb_kw=grid_limit_p_avb_kw,
                            rated_power_kw=rated_power_kw,
                            time_step_hours=time_step_hours,
                            charger_efficiency=charger_efficiency,
                            emergency_slack_minutes=emergency_slack_minutes,

                            market_enabled=(market_map is not None and market_unit is not None),
                            market_map=market_map,
                            market_unit=market_unit,
                            strategy_resolution_min=STRATEGY_RESOLUTION_MIN,
                            strategy_step_hours=strategy_step_hours,
                            hard_immediate_slack_minutes=0.0,
                        )
                    )
                else:
                    # Niemand lädt, aber PV-Überschuss kann existieren
                    mode_label_for_debug = "PV_ONLY"

                # Physikalischer Grid-Import
                grid_import_kw_site = max(0.0, float(total_power_kw) - float(pv_surplus_kw_now))

                # PV-Plan Cleanup für fertige Sessions (Reservierungen freigeben)
                did_cleanup_event = clear_future_pv_plan_if_finished(
                    sessions=[s for s in chargers if s is not None],
                    current_idx=i,
                    pv_reserved_kw=pv_reserved_kw,
                )

                # Event-basiertes Replanning für die Zukunft, wenn Grid genutzt wurde oder Sessions fertig wurden
                if (did_use_grid or did_finish_event or did_cleanup_event) and (i + 1) < n_steps:
                    replan_pv_for_plugged_sessions_on_event(
                         ts=ts,                      # Slack-Bewertung zum aktuellen Zeitpunkt
                         now_idx=i + 1,               # Planung beginnt erst ab nächstem Schritt
                         time_index=time_index,
                         chargers=chargers,
                         pv_available_kw=pv_available_kw,
                         pv_reserved_kw=pv_reserved_kw,
                         rated_power_kw=rated_power_kw,
                         time_step_hours=time_step_hours,
                         charger_efficiency=charger_efficiency,
                    )
                    
 
                    if market_map is not None and market_unit is not None and market_reserved_kw is not None:
                         replan_market_fallback_for_generation_on_event(
                             ts=ts,
                             now_idx=i + 1,
                             time_index=time_index,
                             chargers=chargers,
                             market_map=market_map,
                             market_resolution_min=STRATEGY_RESOLUTION_MIN,
                             market_unit=str(market_unit),
                             step_hours_strategy=strategy_step_hours,
                             market_reserved_kw=market_reserved_kw,
                             grid_limit_p_avb_kw=grid_limit_p_avb_kw,
                             rated_power_kw=rated_power_kw,
                             time_step_hours=time_step_hours,
                             charger_efficiency=charger_efficiency,
                        )
        else:
            raise ValueError(f"Unbekannte charging_strategy='{charging_strategy}'")

        # --------------------------------------------------------
        # 4) Ergebnis je Zeitschritt setzen
        # --------------------------------------------------------
        load_profile_kw[i] = float(total_power_kw)

        # --------------------------------------------------------
        # 5) Charger Traces (optional): pro Zeitschritt, pro Ladepunkt
        # --------------------------------------------------------
        if record_charger_traces:
            for cid, s in enumerate(chargers):
                if s is None:
                    charger_trace_rows.append(
                        {
                            "ts": ts,
                            "charger_id": cid,
                            "occupied": False,
                            "session_id": None,
                            "vehicle_name": None,
                            "vehicle_class": None,
                            "soc": np.nan,
                            "power_kw": 0.0,
                        }
                    )
                    continue

                delivered = float(s.get("delivered_energy_kwh", 0.0))
                cap = float(s.get("battery_capacity_kwh", np.nan))
                soc_arr = float(s.get("soc_arrival", np.nan))
                soc_target = float(s.get("soc_target", 1.0))

                soc = np.nan
                if cap > 1e-9 and not np.isnan(cap) and not np.isnan(soc_arr):
                    soc = soc_arr + delivered / cap
                    soc = min(soc, soc_target)

                p = float(s.get("_actual_power_kw", 0.0))

                charger_trace_rows.append(
                    {
                        "ts": ts,
                        "charger_id": cid,
                        "occupied": True,
                        "session_id": s.get("session_id"),
                        "vehicle_name": s.get("vehicle_name"),
                        "vehicle_class": s.get("vehicle_class"),
                        "soc": soc,
                        "power_kw": p,
                    }
                )

        # --------------------------------------------------------
        # 6) Debug-Row je Zeitschritt (optional)
        # --------------------------------------------------------
        if record_debug:
            row = {
                "ts": ts,
                "mode": mode_label_for_debug,
                "site_total_power_kw": float(total_power_kw),
                "grid_import_kw_site": float(grid_import_kw_site),
            }
            if charging_strategy == "generation":
                row.update(
                    {
                        "pv_surplus_kw": float(pv_surplus_kw_now),
                        "did_use_grid": bool(did_use_grid),
                        "did_finish_event": bool(did_finish_event),
                        "did_cleanup_event": bool(did_cleanup_event),
                    }
                )
            debug_rows.append(row)

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
        charger_trace_rows if record_charger_traces else None,
    )



# =============================================================================
# Reporting / KPI Helper (Notebook)
# =============================================================================
# Dieser Abschnitt bündelt ausschließlich Notebook-Helfer und Auswertungsfunktionen.
#
# Inhalt:
#   - KPI-Zusammenfassungen über Sessions (Plug-In, Rejects, Zielerreichung)
#   - Kalender-/Gruppierungsfunktionen für Histogramme
#   - PV-Überschuss-Auswertung aus debug_rows (ungennutzter PV-Anteil)
#   - Strategie-Signalreihen für Plotting (Generation / Market)
#   - Zusatz-KPIs: Sessions/Energie nach tatsächlich genutztem Lademodus


def summarize_sessions(
    sessions: list[dict[str, Any]] | None,
    eps_kwh: float = 1e-6,
) -> dict[str, Any]:
    """
    Diese Funktion erzeugt eine KPI-Zusammenfassung über alle Sessions.

    Sie betrachtet:
      - alle Sessions mit Ladebedarf (Grundmodell),
      - Sessions mit physischem Ladezugang (eingesteckt),
      - abgewiesene Sessions (drive_off),
      - Sessions, die ihr Ziel-SoC nicht erreicht haben.

    Rückgabe (notebook-kompatibel):
      - num_sessions_total: Anzahl aller Sessions (mit Ladebedarf im Grundmodell)
      - num_sessions_plugged: Anzahl Sessions, die physisch eingesteckt wurden
      - num_sessions_rejected: Anzahl Sessions, die wegen fehlender Ladepunkte abgewiesen wurden
      - not_reached_rows: Liste von Sessions, die ihr Ziel nicht erreicht haben (Restenergie > eps_kwh)
    """
    if not sessions:
        return {
            "num_sessions_total": 0,
            "num_sessions_plugged": 0,
            "num_sessions_rejected": 0,
            "not_reached_rows": [],
        }

    plugged = [s for s in sessions if s.get("_plug_in_time") is not None]
    rejected = [s for s in sessions if bool(s.get("_rejected", False))]

    not_reached_rows: list[dict[str, Any]] = []
    for s in plugged:
        remaining = float(s.get("energy_required_kwh", 0.0))
        if remaining <= float(eps_kwh):
            continue

        arrival = s.get("arrival_time")
        departure = s.get("departure_time")

        parking_hours = None
        if arrival is not None and departure is not None:
            parking_hours = (departure - arrival).total_seconds() / 3600.0

        not_reached_rows.append(
            {
                "session_id": s.get("session_id", ""),
                "vehicle_name": s.get("vehicle_name", ""),
                "vehicle_class": s.get("vehicle_class", ""),
                "arrival_time": arrival,
                "departure_time": departure,
                "parking_hours": parking_hours,
                "delivered_energy_kwh": float(s.get("delivered_energy_kwh", 0.0)),
                "remaining_energy_kwh": remaining,
                "charger_id": s.get("_charger_id", None),
            }
        )

    return {
        "num_sessions_total": len(sessions),
        "num_sessions_plugged": len(plugged),
        "num_sessions_rejected": len(rejected),
        "not_reached_rows": not_reached_rows,
    }


def get_daytype_calendar(
    start_datetime: datetime,
    horizon_days: int,
    holiday_dates: set[date],
) -> dict[str, list[date]]:
    """
    Diese Funktion erzeugt eine Kalenderliste der Tage je Tagtyp.

    Sie nutzt bewusst die zentrale Logik determine_day_type_with_holidays(...),
    damit die Tagesklassifikation konsistent zur Simulation bleibt.

    Rückgabe:
      - dict mit Keys: working_day, saturday, sunday_holiday
      - Werte: Liste der jeweiligen Datumswerte (date)
    """
    out: dict[str, list[date]] = {"working_day": [], "saturday": [], "sunday_holiday": []}

    for i in range(int(horizon_days)):
        d = start_datetime.date() + timedelta(days=i)

        # Für die Tagesklassifikation wird bewusst eine fixe Tagesmitte verwendet.
        dt_mid = datetime(d.year, d.month, d.day, 12, 0)
        dt_type = determine_day_type_with_holidays(dt_mid, holiday_dates)

        out.setdefault(dt_type, []).append(d)

    return out


def group_sessions_by_day(
    sessions: list[dict[str, Any]] | None,
    only_plugged: bool = False,
) -> dict[date, list[dict[str, Any]]]:
    """
    Diese Funktion gruppiert Sessions nach ihrem Ankunftsdatum.

    Parameter:
      - only_plugged=True filtert auf Sessions, die physisch eingesteckt wurden.

    Rückgabe:
      - dict[date] -> list[session]
    """
    out: dict[date, list[dict[str, Any]]] = {}
    if not sessions:
        return out

    for s in sessions:
        if only_plugged and s.get("_plug_in_time") is None:
            continue

        arrival = s.get("arrival_time", None)
        if arrival is None:
            continue

        d = arrival.date()
        out.setdefault(d, []).append(s)

    return out


def build_pv_unused_table(
    debug_rows: list[dict[str, Any]] | None,
    eps_kw: float = 1e-6,
    eps_unused_kw: float = 1e-3,
):
    """
    Diese Funktion erstellt eine Tabelle für Zeitschritte mit ungenutztem PV-Überschuss.

    Hintergrund:
      - In der generation-Strategie wird PV (bzw. PV-Überschuss = PV - Grundlast) als Budget betrachtet.
      - Diese Funktion zeigt, in welchen Zeitschritten PV verfügbar war, aber nicht vollständig genutzt wurde.

    Erwartete Felder in debug_rows (generation-Strategie):
      - ts
      - pv_surplus_kw
      - site_total_power_kw
      - optional: grid_import_kw_site

    Parameter:
      - eps_kw: Schwelle, ab der PV als "vorhanden" gilt
      - eps_unused_kw: Schwelle, ab der PV als "ungenutzt" gilt

    Rückgabe:
      - pandas.DataFrame (kann leer sein)
      - None, wenn pandas nicht verfügbar ist
    """
    try:
        import pandas as pd
        import numpy as np
    except Exception:
        return None

    if not debug_rows:
        return pd.DataFrame()

    df = pd.DataFrame(debug_rows).copy()
    if "ts" not in df.columns:
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    required = {"ts", "pv_surplus_kw", "site_total_power_kw"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    # Nur Zeitschritte betrachten, in denen PV-Überschuss überhaupt vorhanden ist.
    df_pv = df[df["pv_surplus_kw"] > float(eps_kw)].copy()
    if len(df_pv) == 0:
        return pd.DataFrame()

    # Pro Timestamp aggregieren (debug_rows kann mehrfach pro ts vorkommen)
    agg: dict[str, tuple[str, str]] = {
        "pv_surplus_kw": ("pv_surplus_kw", "first"),
        "site_power_kw": ("site_total_power_kw", "first"),
        "n_rows": ("pv_surplus_kw", "size"),
    }
    if "grid_import_kw_site" in df_pv.columns:
        agg["grid_import_kw"] = ("grid_import_kw_site", "first")

    pv_steps = df_pv.groupby("ts", as_index=False).agg(**agg)

    # PV-Nutzung ist durch Standortleistung begrenzt.
    pv_steps["pv_used_kw"] = np.minimum(pv_steps["pv_surplus_kw"], pv_steps["site_power_kw"])
    pv_steps["pv_unused_kw"] = (pv_steps["pv_surplus_kw"] - pv_steps["pv_used_kw"]).clip(lower=0.0)

    pv_unused_steps = pv_steps[pv_steps["pv_unused_kw"] > float(eps_unused_kw)].copy()
    pv_unused_steps = pv_unused_steps.sort_values(["pv_unused_kw", "ts"], ascending=[False, True]).reset_index(drop=True)

    return pv_unused_steps


def build_strategy_signal_series(
    scenario: dict[str, Any],
    timestamps: list[datetime],
    charging_strategy: str,
    normalize_to_internal: bool = True,
    strategy_resolution_min: int = 15,
) -> tuple[np.ndarray | None, str | None]:
    """
    Diese Funktion erstellt eine an die Simulationstimestamps ausgerichtete Strategie-Zeitreihe
    für Plotting im Notebook.

    Verhalten:
      - "market": Marktpreise (optional normalisiert zu €/kWh)
      - "generation": Erzeugung (optional normalisiert zu kW)

    Rückgabe:
      - series (numpy array) oder None
      - y_label (Achsenbeschriftung) oder None
    """
    strat = (charging_strategy or "").strip().lower()
    if strat not in ("market", "generation") or not timestamps:
        return None, None

    site_cfg = scenario.get("site", {}) or {}
    step_hours = float(strategy_resolution_min) / 60.0

    def _load_cfg(prefix: str, allowed_units: set[str]) -> tuple[str, str, int]:
        """
        Diese Funktion lädt und validiert die Strategie-Konfiguration aus dem Szenario.
        """
        unit = str(site_cfg.get(f"{prefix}_strategy_unit", "") or "").strip()
        csv_rel = site_cfg.get(f"{prefix}_strategy_csv", None)
        col = site_cfg.get(f"{prefix}_strategy_value_col", None)

        if unit not in allowed_units:
            raise ValueError(
                f"❌ Abbruch: 'site.{prefix}_strategy_unit' ungültig "
                f"(erlaubt: {sorted(allowed_units)})."
            )
        if not csv_rel or not isinstance(col, int) or col < 2:
            raise ValueError(
                f"❌ Abbruch: 'site.{prefix}_strategy_csv' oder "
                f"'site.{prefix}_strategy_value_col' fehlt/ungültig."
            )
        return unit, str(csv_rel), int(col)

    if strat == "market":
        unit, csv_rel, col_1_based = _load_cfg("market", {"€/MWh", "€/kWh"})
        csv_path = resolve_path_relative_to_scenario(scenario, csv_rel)
        strat_map = read_strategy_series_from_csv_first_col_time(csv_path, col_1_based, delimiter=";")

        series = np.full(len(timestamps), np.nan, dtype=float)
        for i, ts in enumerate(timestamps):
            v = lookup_signal(strat_map, ts, strategy_resolution_min)
            if v is None:
                continue
            series[i] = (
                convert_strategy_value_to_internal("market", float(v), unit, step_hours)
                if normalize_to_internal else float(v)
            )

        y_label = "Preis [€/kWh]" if normalize_to_internal else f"MARKET [{unit}]"
        return series, y_label

    # generation
    unit, csv_rel, col_1_based = _load_cfg("generation", {"kW", "kWh", "MWh"})
    csv_path = resolve_path_relative_to_scenario(scenario, csv_rel)
    strat_map = read_strategy_series_from_csv_first_col_time(csv_path, col_1_based, delimiter=";")

    series = np.full(len(timestamps), np.nan, dtype=float)
    for i, ts in enumerate(timestamps):
        v = lookup_signal(strat_map, ts, strategy_resolution_min)
        if v is None:
            continue
        series[i] = (
            convert_strategy_value_to_internal("generation", float(v), unit, step_hours)
            if normalize_to_internal else float(v)
        )

    y_label = "Erzeugung [kW]" if normalize_to_internal else f"GENERATION [{unit}]"
    return series, y_label


def summarize_sessions_by_charging_mode(
    sessions: list[dict[str, Any]] | None,
    eps_kwh: float = 1e-6,
) -> dict[str, Any]:
    """
    Diese Funktion zählt, wie viele Sessions mindestens einmal Energie in den Modi
    generation, market oder immediate erhalten haben.

    Hinweis:
      - Eine Session kann in mehreren Modi gezählt werden (z.B. generation + market fallback).
      - Gezählt wird nur, wenn tatsächlich Energie > eps_kwh im jeweiligen Modus geliefert wurde.
    """
    modes = ("generation", "market", "immediate")

    if not sessions:
        return {
            "sessions_with_any_charging": 0,
            "sessions_charged_with_generation": 0,
            "sessions_charged_with_market": 0,
            "sessions_charged_with_immediate": 0,
            "sessions_with_multiple_modes": 0,
        }

    counts = {m: 0 for m in modes}
    sessions_with_any = 0
    sessions_multi_mode = 0

    for s in sessions:
        energy_by_mode = s.get("_energy_by_mode_kwh", {}) or {}
        used = [m for m in modes if float(energy_by_mode.get(m, 0.0)) > float(eps_kwh)]

        if not used:
            continue

        sessions_with_any += 1
        for m in used:
            counts[m] += 1
        if len(used) > 1:
            sessions_multi_mode += 1

    return {
        "sessions_with_any_charging": sessions_with_any,
        "sessions_charged_with_generation": counts["generation"],
        "sessions_charged_with_market": counts["market"],
        "sessions_charged_with_immediate": counts["immediate"],
        "sessions_with_multiple_modes": sessions_multi_mode,
    }


def summarize_energy_by_charging_mode(
    sessions: list[dict[str, Any]] | None,
) -> dict[str, float]:
    """
    Diese Funktion summiert die geladene Energie (kWh) getrennt nach Modus über alle Sessions.

    Voraussetzung:
      - apply_energy_update(...) hat pro Session _energy_by_mode_kwh geschrieben.

    Rückgabe:
      - dict mit Schlüsseln: generation, market, immediate
      - Werte: Energie in kWh
    """
    total = {"generation": 0.0, "market": 0.0, "immediate": 0.0}
    if not sessions:
        return total

    for s in sessions:
        energy_by_mode = s.get("_energy_by_mode_kwh", {}) or {}
        for mode in total.keys():
            total[mode] += float(energy_by_mode.get(mode, 0.0))

    return total
