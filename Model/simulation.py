import yaml
import numpy as np
from datetime import datetime, timedelta, date
from typing import Any


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

    Beispiele
    ---------
    0.45        -> 0.45
    [0.3, 0.7]  -> Zufallswert ~ U(0.3, 0.7)
    [0.0]       -> 0.0
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
    definierten Feiertage:

      - 'sunday_holiday', wenn Datum in der Feiertagsliste oder Sonntag ist
      - 'saturday', wenn Samstag
      - 'working_day', sonst (Montag-Freitag ohne Feiertag)
    """
    current_date = current_datetime.date()
    holiday_dates = parse_holiday_dates_from_scenario(scenario)

    # Feiertag hat Vorrang und wird wie Sonntag/Ferien behandelt
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

    Verwendet:
      - scenario["start_datetime"] oder aktuellen Tag 00:00 als Start
      - scenario["time_resolution_min"] als Zeitschritt
      - scenario["simulation_horizon_days"] als Anzahl der Tage
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

    Grundlage:
      - scenario["arrival_time_distribution"]["weekday_weight"][day_type] (als Range)
      - scenario["site"]["expected_sessions_per_charger_per_day"] (als Range)
      - scenario["arrival_time_distribution"]["components_per_weekday"][day_type]
        (mu-/sigma-Ranges, shift_minutes, weight)
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

def build_charging_sessions_for_day(scenario: dict, day_start_datetime: datetime) -> list[dict[str, Any]]:
    """
    Erzeugt für einen Tag eine Liste von Ladesessions mit allen relevanten Parametern.
    """
    arrival_times_for_day = sample_arrival_times_for_day(scenario, day_start_datetime)
    number_of_sessions = len(arrival_times_for_day)

    if number_of_sessions == 0:
        return []

    parking_durations_minutes = sample_parking_durations(scenario, number_of_sessions)
    soc_values_at_arrival = sample_soc_upon_arrival(scenario, number_of_sessions)

    target_soc = scenario["vehicles"]["soc_target"]
    battery_capacity_kwh = scenario["vehicles"]["battery_capacity_kwh"]
    max_vehicle_charging_power_kw = scenario["vehicles"]["max_pwr_vehicle_kw"]

    charging_sessions_for_day: list[dict[str, Any]] = []

    for session_index in range(number_of_sessions):
        arrival_time = arrival_times_for_day[session_index]
        departure_time = arrival_time + timedelta(
            minutes=float(parking_durations_minutes[session_index])
        )

        soc_at_arrival = float(soc_values_at_arrival[session_index])
        delta_soc = max(target_soc - soc_at_arrival, 0.0)
        required_energy_kwh = delta_soc * battery_capacity_kwh

        charging_session: dict[str, Any] = {
            "arrival_time": arrival_time,
            "departure_time": departure_time,
            "soc_arrival": soc_at_arrival,
            "soc_target": target_soc,
            "battery_capacity_kwh": battery_capacity_kwh,
            "energy_required_kwh": required_energy_kwh,
            "max_charging_power_kw": max_vehicle_charging_power_kw,
        }
        charging_sessions_for_day.append(charging_session)

    return charging_sessions_for_day


# ---------------------------------------------------------------------------
# Hauptsimulation: Lastprofil
# ---------------------------------------------------------------------------

def simulate_load_profile(scenario: dict, start_datetime: datetime | None = None):
    """
    Führt die Lastgang-Simulation für den im Szenario definierten
    Simulationshorizont durch.

    Rückgabe:
      - Liste von Zeitstempeln
      - Array mit Lastwerten in kW
      - Liste aller Ladesessions
    """
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
                scenario, day_start_datetime
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
            allowed_power_kw = min(
                available_power_per_session_kw,
                charging_session["max_charging_power_kw"],
            )

            possible_energy_in_time_step_kwh = (
                allowed_power_kw * time_step_hours * charger_efficiency
            )

            if possible_energy_in_time_step_kwh >= charging_session["energy_required_kwh"]:
                charging_session["energy_required_kwh"] = 0.0
            else:
                charging_session["energy_required_kwh"] -= possible_energy_in_time_step_kwh

            total_power_in_time_step_kw += allowed_power_kw

        load_profile_kw[time_step_index] = total_power_in_time_step_kw

    return time_index, load_profile_kw, all_charging_sessions
