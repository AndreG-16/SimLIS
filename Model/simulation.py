import yaml
import numpy as np
import csv  # NEU
from dataclasses import dataclass  # NEU
from datetime import datetime, timedelta, date
from typing import Any


# ---------------------------------------------------------------------------
# NEU: Fahrzeugprofil-Dataclass
# ---------------------------------------------------------------------------

@dataclass
class VehicleProfile:
    name: str
    battery_capacity_kwh: float
    soc_grid: np.ndarray        # SoC-Stützstellen (0..1)
    power_grid_kw: np.ndarray   # Ladeleistung in kW zu den SoC-Stützstellen


# ---------------------------------------------------------------------------
# NEU: Fahrzeuge aus CSV laden
# ---------------------------------------------------------------------------

def load_vehicle_profiles_from_csv(path: str) -> list[VehicleProfile]:
    """
    Liest eine CSV mit Fahrzeugen und SoC-abhängigen Ladekurven und gibt
    eine Liste von VehicleProfile-Objekten zurück.

    Erwartete Struktur (vereinfacht):
      Zeile 1: Marken (wird ignoriert)
      Zeile 2: "Modell;ID.3;ID.4;..."
      Zeile 3: "max. Kapazität;77;77;..."
      Zeile 4: "SoC [%];..."
      ab Zeile 5: "0;P_ID3;P_ID4;..." etc.
    """
    vehicle_profiles: list[VehicleProfile] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")

        # 1. Zeile: Marken (ignorieren)
        _brands_row = next(reader, None)

        # 2. Zeile: Modellnamen
        model_row = next(reader, None)
        if model_row is None:
            return []

        model_names = model_row[1:]  # ab Spalte 1 sind die Fahrzeugmodelle

        # 3. Zeile: max. Kapazitäten
        capacity_row = next(reader, None)
        if capacity_row is None:
            return []

        raw_capacities = capacity_row[1:]

        # 4. Zeile: "SoC [%]" (Header)
        _soc_header_row = next(reader, None)

        # Kapazitäten in kWh lesen
        capacities_kwh: list[float] = []
        for val in raw_capacities:
            val = val.strip()
            if val == "":
                capacities_kwh.append(np.nan)
            else:
                try:
                    cap = float(val.replace(",", "."))
                    # 0 oder negative Kapazität ignorieren
                    capacities_kwh.append(cap if cap > 0 else np.nan)
                except ValueError:
                    capacities_kwh.append(np.nan)

        num_vehicles = len(model_names)
        soc_lists: list[list[float]] = [[] for _ in range(num_vehicles)]
        power_lists: list[list[float]] = [[] for _ in range(num_vehicles)]

        # Ab hier: Zeilen mit SoC-Werten und zugehörigen Ladeleistungen
        for row in reader:
            if not row:
                continue

            soc_str = row[0].strip()
            if soc_str == "":
                continue

            try:
                soc_val_percent = float(soc_str.replace(",", "."))
            except ValueError:
                continue

            # SoC in 0..1 umrechnen
            soc_val = soc_val_percent / 100.0

            # Spalten 1..n: Ladeleistungen je Fahrzeug
            for idx in range(num_vehicles):
                if idx + 1 >= len(row):
                    continue
                cell = row[idx + 1].strip()
                if cell == "":
                    continue
                try:
                    power_kw = float(cell.replace(",", "."))
                except ValueError:
                    continue

                soc_lists[idx].append(soc_val)
                power_lists[idx].append(power_kw)

        # VehicleProfile-Objekte bauen
        for i in range(num_vehicles):
            name = model_names[i].strip()
            cap = capacities_kwh[i]

            # Nur Fahrzeuge mit gültiger Kapazität und vorhandener Kurve
            if np.isnan(cap) or len(soc_lists[i]) == 0:
                continue

            soc_grid = np.array(soc_lists[i], dtype=float)
            power_grid = np.array(power_lists[i], dtype=float)

            # nach SoC sortieren
            sort_idx = np.argsort(soc_grid)
            soc_grid = soc_grid[sort_idx]
            power_grid = power_grid[sort_idx]

            vehicle_profiles.append(
                VehicleProfile(
                    name=name,
                    battery_capacity_kwh=cap,
                    soc_grid=soc_grid,
                    power_grid_kw=power_grid,
                )
            )

    return vehicle_profiles


# ---------------------------------------------------------------------------
# NEU: fahrzeugspezifische Ladeleistung für aktuellen SoC
# ---------------------------------------------------------------------------

def vehicle_power_at_soc(session: dict[str, Any]) -> float:
    """
    Gibt die fahrzeugspezifische maximale Ladeleistung (kW) für den aktuellen
    SoC zurück, basierend auf der in der Session gespeicherten Ladekurve.
    """
    soc_grid = session["soc_grid"]          # np.ndarray (0..1)
    power_grid = session["power_grid_kw"]   # np.ndarray

    soc_arrival = session["soc_arrival"]
    delivered_energy = session.get("delivered_energy_kwh", 0.0)
    capacity = session["battery_capacity_kwh"]

    # aktueller SoC = SoC bei Ankunft + geladene Energie / Kapazität
    current_soc = soc_arrival + delivered_energy / capacity
    current_soc = min(current_soc, session["soc_target"])

    # lineare Interpolation
    power_kw = float(np.interp(current_soc, soc_grid, power_grid))

    return max(power_kw, 0.0)


# ---------------------------------------------------------------------------
# Szenario laden
# ---------------------------------------------------------------------------

def load_scenario(path: str) -> dict[str, Any]:
    """
    Liest eine YAML-Szenariodatei ein und gibt sie als Dictionary zurück.
    """
    with open(path, "r", encoding="utf-8") as file:
        scenario = yaml.safe_load(file)
    return scenario


# ---------------------------------------------------------------------------
# Hilfsfunktionen: Range-Verarbeitung und Tagtyp
# ---------------------------------------------------------------------------

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


def parse_holiday_dates_from_scenario(scenario: dict) -> set[date]:
    """
    Liest Feiertage aus scenario['holidays']['dates'] und gibt sie als Set von date-Objekten zurück.
    """
    holiday_dates_strings = scenario.get("holidays", {}).get("dates", [])
    holiday_dates: set[date] = set()

    for date_string in holiday_dates_strings:
        # Format in YAML: 'YYYY-MM-DD'
        holiday_date = datetime.fromisoformat(date_string).date()
        holiday_dates.add(holiday_date)

    return holiday_dates


def determine_day_type_with_holidays(current_datetime: datetime, scenario: dict) -> str:
    """
    Ordnet ein Datum einem Tagtyp zu, unter Berücksichtigung der im Szenario
    definierten Feiertage.
    """
    current_date = current_datetime.date()
    holiday_dates = parse_holiday_dates_from_scenario(scenario)

    if current_date in holiday_dates:
        return "sunday_holiday"

    weekday_index = current_datetime.weekday()  # Montag = 0, Sonntag = 6

    if weekday_index == 6:
        return "sunday_holiday"
    if weekday_index == 5:
        return "saturday"
    return "working_day"


# ---------------------------------------------------------------------------
# Zeitindex
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Mischung von Lognormalverteilungen
# ---------------------------------------------------------------------------

def sample_lognormal_mixture(
    number_of_samples: int,
    mixture_components: list[dict[str, Any]],
    max_value: float | None = None,
    unit_description: str = "generic",
) -> np.ndarray:
    """
    Erzeugt Stichproben aus einer Mischung von Lognormalverteilungen.
    """
    if number_of_samples <= 0:
        return np.array([])

    component_weights = np.array(
        [component["weight"] for component in mixture_components],
        dtype=float,
    )
    component_weights = component_weights / component_weights.sum()

    chosen_component_indices = np.random.choice(
        len(mixture_components),
        size=number_of_samples,
        p=component_weights,
    )

    sampled_values = np.zeros(number_of_samples, dtype=float)

    for sample_index, component_index in enumerate(chosen_component_indices):
        component = mixture_components[component_index]

        mu_value = float(component["mu"])
        sigma_value = float(component["sigma"])

        lognormal_value = np.random.lognormal(mean=mu_value, sigma=sigma_value)

        if "shift_minutes" in component and component["shift_minutes"] is not None:
            lognormal_value = lognormal_value * 60.0 + float(component["shift_minutes"])

        sampled_values[sample_index] = lognormal_value

    if max_value is not None:
        sampled_values = np.minimum(sampled_values, max_value)

    return sampled_values


# ---------------------------------------------------------------------------
# Ankunftszeiten
# ---------------------------------------------------------------------------

def sample_arrival_times_for_day(scenario: dict, day_start_datetime: datetime) -> list[datetime]:
    """
    Generiert alle Ankunftszeitpunkte für einen Tag als datetime-Objekte.
    """
    day_type = determine_day_type_with_holidays(day_start_datetime, scenario)

    number_of_chargers = scenario["site"]["number_chargers"]
    expected_sessions_per_charger_range = scenario["site"]["expected_sessions_per_charger_per_day"]
    expected_sessions_per_charger = sample_from_range(expected_sessions_per_charger_range)

    weekday_weight_range = scenario["arrival_time_distribution"]["weekday_weight"][day_type]
    weekday_weight = sample_from_range(weekday_weight_range)

    number_of_sessions_today = int(
        number_of_chargers * expected_sessions_per_charger * weekday_weight
    )

    if number_of_sessions_today <= 0:
        return []

    component_templates_for_day_type = scenario["arrival_time_distribution"]["components_per_weekday"][day_type]

    realized_mixture_components: list[dict[str, Any]] = []
    for component_template in component_templates_for_day_type:
        realized_component: dict[str, Any] = {
            "mu": sample_from_range(component_template["mu"]),
            "sigma": sample_from_range(component_template["sigma"]),
            "weight": sample_from_range(component_template.get("weight", 1.0)),
            "shift_minutes": component_template.get("shift_minutes", None),
        }
        realized_mixture_components.append(realized_component)

    if not realized_mixture_components:
        return []

    sampled_minutes_after_midnight = sample_lognormal_mixture(
        number_of_samples=number_of_sessions_today,
        mixture_components=realized_mixture_components,
        max_value=None,
        unit_description="minutes",
    )

    arrival_times_for_day = [
        day_start_datetime + timedelta(minutes=float(minutes_value))
        for minutes_value in sampled_minutes_after_midnight
    ]
    arrival_times_for_day.sort()

    return arrival_times_for_day


# ---------------------------------------------------------------------------
# Parkdauer
# ---------------------------------------------------------------------------

def sample_parking_durations(scenario: dict, number_of_sessions: int) -> np.ndarray:
    """
    Generiert Parkdauern in Minuten für die übergebene Anzahl an Sessions.
    """
    parking_duration_distribution = scenario["parking_duration_distribution"]
    maximum_parking_duration_minutes = parking_duration_distribution["max_duration_minutes"]

    realized_mixture_components: list[dict[str, Any]] = []

    for component_template in parking_duration_distribution["components"]:
        realized_component: dict[str, Any] = {
            "mu": sample_from_range(component_template["mu"]),
            "sigma": sample_from_range(component_template["sigma"]),
            "weight": sample_from_range(component_template["weight"]),
        }
        realized_mixture_components.append(realized_component)

    parking_durations_minutes = sample_lognormal_mixture(
        number_of_samples=number_of_sessions,
        mixture_components=realized_mixture_components,
        max_value=maximum_parking_duration_minutes,
        unit_description="minutes",
    )

    return parking_durations_minutes


# ---------------------------------------------------------------------------
# SoC bei Ankunft
# ---------------------------------------------------------------------------

def sample_soc_upon_arrival(scenario: dict, number_of_sessions: int) -> np.ndarray:
    """
    Generiert SoC-Werte (0..1) bei Ankunft für mehrere Ladesessions.
    """
    soc_at_arrival_distribution = scenario["soc_at_arrival_distribution"]
    maximum_soc_value = soc_at_arrival_distribution["max_soc"]

    realized_mixture_components: list[dict[str, Any]] = []

    for component_template in soc_at_arrival_distribution["components"]:
        realized_component: dict[str, Any] = {
            "mu": sample_from_range(component_template["mu"]),
            "sigma": sample_from_range(component_template["sigma"]),
            "weight": sample_from_range(component_template["weight"]),
        }
        realized_mixture_components.append(realized_component)

    soc_values = sample_lognormal_mixture(
        number_of_samples=number_of_sessions,
        mixture_components=realized_mixture_components,
        max_value=maximum_soc_value,
        unit_description="soc_fraction",
    )

    return soc_values


# ---------------------------------------------------------------------------
# Ladesessions eines Tages
# ---------------------------------------------------------------------------

def build_charging_sessions_for_day(
    scenario: dict,
    day_start_datetime: datetime,
    vehicle_profiles: list[VehicleProfile],  # NEU
) -> list[dict[str, Any]]:
    """
    Erzeugt für einen Tag eine Liste von Ladesessions mit allen relevanten Parametern.
    Jetzt mit zufälliger Fahrzeugwahl und fahrzeugspezifischen Ladekurven.
    """
    arrival_times_for_day = sample_arrival_times_for_day(scenario, day_start_datetime)
    number_of_sessions = len(arrival_times_for_day)

    if number_of_sessions == 0:
        return []

    parking_durations_minutes = sample_parking_durations(scenario, number_of_sessions)
    soc_values_at_arrival = sample_soc_upon_arrival(scenario, number_of_sessions)

    target_soc = scenario["vehicles"]["soc_target"]

    charging_sessions_for_day: list[dict[str, Any]] = []

    for session_index in range(number_of_sessions):
        arrival_time = arrival_times_for_day[session_index]
        departure_time = arrival_time + timedelta(
            minutes=float(parking_durations_minutes[session_index])
        )

        soc_at_arrival = float(soc_values_at_arrival[session_index])

        # NEU: zufällig ein Fahrzeug aus der Flotte wählen
        vehicle_profile = np.random.choice(vehicle_profiles)
        battery_capacity_kwh = float(vehicle_profile.battery_capacity_kwh)

        delta_soc = max(target_soc - soc_at_arrival, 0.0)
        required_energy_kwh = delta_soc * battery_capacity_kwh

        # Maximalleistung als Maximum der Kurve (zusätzlich zur SoC-abhängigen Grenze)
        max_vehicle_charging_power_kw = float(vehicle_profile.power_grid_kw.max())

        charging_session: dict[str, Any] = {
            "arrival_time": arrival_time,
            "departure_time": departure_time,
            "soc_arrival": soc_at_arrival,
            "soc_target": target_soc,
            "battery_capacity_kwh": battery_capacity_kwh,
            "energy_required_kwh": required_energy_kwh,
            "delivered_energy_kwh": 0.0,  # NEU: bisher geladene Energie zur SoC-Berechnung
            "max_charging_power_kw": max_vehicle_charging_power_kw,
            "vehicle_name": vehicle_profile.name,
            "soc_grid": vehicle_profile.soc_grid,
            "power_grid_kw": vehicle_profile.power_grid_kw,
        }
        charging_sessions_for_day.append(charging_session)

    return charging_sessions_for_day


# ---------------------------------------------------------------------------
# Hauptsimulation: Lastprofil
# ---------------------------------------------------------------------------

def simulate_load_profile(scenario: dict, start_datetime: datetime | None = None):
    """
    Führt die Lastgang-Simulation durch.
    Jetzt mit fahrzeugspezifischen Batteriekapazitäten und SoC-abhängiger Ladeleistung.
    """
    # NEU: Fahrzeugflotte aus CSV laden
    vehicle_csv_path = scenario["vehicles"]["vehicle_curve_csv"]
    vehicle_profiles = load_vehicle_profiles_from_csv(vehicle_csv_path)

    time_index = create_time_index(scenario, start_datetime)

    time_resolution_min = scenario["time_resolution_min"]
    time_step_hours = time_resolution_min / 60.0

    number_of_time_steps = len(time_index)
    load_profile_kw = np.zeros(number_of_time_steps, dtype=float)

    grid_limit_p_avb_kw = scenario["site"]["grid_limit_p_avb_kw"]
    rated_power_kw = scenario["site"]["rated_power_kw"]
    number_of_chargers = scenario["site"]["number_chargers"]
    charger_efficiency = scenario["site"]["charger_efficiency"]

    all_charging_sessions: list[dict[str, Any]] = []

    if time_index:
        first_day_start_datetime = time_index[0].replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        simulation_horizon_days = scenario["simulation_horizon_days"]

        for day_offset in range(simulation_horizon_days):
            day_start_datetime = first_day_start_datetime + timedelta(days=day_offset)
            charging_sessions_for_day = build_charging_sessions_for_day(
                scenario, day_start_datetime, vehicle_profiles  # ANGEPAST
            )
            all_charging_sessions.extend(charging_sessions_for_day)

    for time_step_index, current_timestamp in enumerate(time_index):

        active_charging_sessions = [
            session
            for session in all_charging_sessions
            if session["arrival_time"] <= current_timestamp < session["departure_time"]
            and session["energy_required_kwh"] > 0.0
        ]

        if not active_charging_sessions:
            load_profile_kw[time_step_index] = 0.0
            continue

        maximum_site_power_kw = min(
            grid_limit_p_avb_kw,
            number_of_chargers * rated_power_kw,
        )

        number_of_active_sessions = len(active_charging_sessions)
        available_power_per_session_kw = maximum_site_power_kw / number_of_active_sessions

        total_power_in_time_step_kw = 0.0

        for charging_session in active_charging_sessions:
            # NEU: SoC-abhängige Leistungsgrenze
            vehicle_power_limit_kw = vehicle_power_at_soc(charging_session)

            # Standortanteil & fahrzeugspezifische Limits
            allowed_power_kw = min(
                available_power_per_session_kw,
                vehicle_power_limit_kw,
                charging_session["max_charging_power_kw"],
            )

            possible_energy_in_time_step_kwh = (
                allowed_power_kw * time_step_hours * charger_efficiency
            )

            energy_needed = charging_session["energy_required_kwh"]

            # Tatsächlich gelieferte Energie (nicht über den Bedarf hinaus)
            if possible_energy_in_time_step_kwh >= energy_needed:
                energy_delivered = energy_needed
                charging_session["energy_required_kwh"] = 0.0
                # tatsächliche Leistung anpassen, um Energie- und Leistungsbilanz konsistent zu halten
                actual_power_kw = energy_delivered / (time_step_hours * charger_efficiency) if time_step_hours > 0 else 0.0
            else:
                energy_delivered = possible_energy_in_time_step_kwh
                charging_session["energy_required_kwh"] -= possible_energy_in_time_step_kwh
                actual_power_kw = allowed_power_kw

            charging_session["delivered_energy_kwh"] += energy_delivered
            total_power_in_time_step_kw += actual_power_kw

        load_profile_kw[time_step_index] = total_power_in_time_step_kw

    return time_index, load_profile_kw, all_charging_sessions
