import yaml
import numpy as np
import csv  #NEU
from dataclasses import dataclass  #NEU
from datetime import datetime, timedelta, date
from typing import Any


# ---------------------------------------------------------------------------
# Fahrzeugprofil-Dataclass  #NEU
# ---------------------------------------------------------------------------

@dataclass
class VehicleProfile:
    name: str
    battery_capacity_kwh: float
    vehicle_class: str               #NEU: z.B. "PKW", "Transporter"
    soc_grid: np.ndarray        # SoC-Stützstellen (0..1)
    power_grid_kw: np.ndarray   # Ladeleistung in kW zu den SoC-Stützstellen


# ---------------------------------------------------------------------------
# Fahrzeuge aus CSV laden  #NEU
# ---------------------------------------------------------------------------

def load_vehicle_profiles_from_csv(path: str) -> list[VehicleProfile]:
    """
    Liest eine CSV mit Fahrzeugen und SoC-abhängigen Ladekurven und gibt
    eine Liste von VehicleProfile-Objekten zurück.

    Erwartete Struktur (Spaltentrenner ';'):
      Zeile 1: Hersteller (wird ignoriert)
      Zeile 2: Modellnamen
      Zeile 3: Fahrzeugklasse  #NEU
      Zeile 4: max. Kapazität (kWh)
      Zeile 5: "SoC [%]" (Header)
      ab Zeile 6: SoC-Werte in % + Ladeleistungen je Fahrzeug
    """
    vehicle_profiles: list[VehicleProfile] = []

    with open(path, "r", encoding="utf-8-sig") as f:  # BOM-sicher
        reader = csv.reader(f, delimiter=";")

        # 1. Zeile: Hersteller (ignorieren)
        _brands_row = next(reader, None)

        # 2. Zeile: Modellnamen
        model_row = next(reader, None)
        if not model_row or len(model_row) < 2:
            return []
        model_names = [m.strip() for m in model_row[1:]]

        # 3. Zeile: Fahrzeugklasse  #NEU
        class_row = next(reader, None)
        if not class_row or len(class_row) < 2:
            return []
        vehicle_classes = [c.strip() if c.strip() != "" else "PKW" for c in class_row[1:]]

        # 4. Zeile: max. Kapazitäten
        capacity_row = next(reader, None)
        if not capacity_row or len(capacity_row) < 2:
            return []
        raw_capacities = capacity_row[1:]

        # 5. Zeile: "SoC [%]" (Header)
        _soc_header_row = next(reader, None)

        # Kapazitäten in kWh lesen
        capacities_kwh: list[float] = []
        for val in raw_capacities:
            val = (val or "").strip()
            if val == "":
                capacities_kwh.append(np.nan)
                continue
            try:
                cap = float(val.replace(",", "."))
                capacities_kwh.append(cap if cap > 0 else np.nan)
            except ValueError:
                capacities_kwh.append(np.nan)

        # Konsistenz: Anzahl Spalten vereinheitlichen  #NEU
        num_vehicles = min(len(model_names), len(vehicle_classes), len(capacities_kwh))  #NEU
        model_names = model_names[:num_vehicles]  #NEU
        vehicle_classes = vehicle_classes[:num_vehicles]  #NEU
        capacities_kwh = capacities_kwh[:num_vehicles]  #NEU

        soc_lists: list[list[float]] = [[] for _ in range(num_vehicles)]
        power_lists: list[list[float]] = [[] for _ in range(num_vehicles)]

        # Ab hier: Zeilen mit SoC-Werten und zugehörigen Ladeleistungen
        for row in reader:
            if not row:
                continue

            soc_str = (row[0] or "").strip()
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
                cell = (row[idx + 1] or "").strip()
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
            name = model_names[i]
            vclass = vehicle_classes[i]  #NEU
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
                    battery_capacity_kwh=float(cap),
                    vehicle_class=vclass,      #NEU
                    soc_grid=soc_grid,
                    power_grid_kw=power_grid,
                )
            )

    return vehicle_profiles


# ---------------------------------------------------------------------------
# fahrzeugspezifische Ladeleistung für aktuellen SoC  #NEU
# ---------------------------------------------------------------------------

def vehicle_power_at_soc(session: dict[str, Any]) -> float:
    """
    Gibt die fahrzeugspezifische maximale Ladeleistung (kW) für den aktuellen
    SoC zurück, basierend auf der in der Session gespeicherten Ladekurve.
    """
    soc_grid = session["soc_grid"]
    power_grid = session["power_grid_kw"]

    soc_arrival = session["soc_arrival"]
    delivered_energy = session.get("delivered_energy_kwh", 0.0)
    capacity = session["battery_capacity_kwh"]

    # aktueller SoC = SoC bei Ankunft + geladene Energie / Kapazität
    current_soc = soc_arrival + delivered_energy / capacity
    current_soc = min(current_soc, session["soc_target"])

    power_kw = float(np.interp(current_soc, soc_grid, power_grid))
    return max(power_kw, 0.0)


# ---------------------------------------------------------------------------
# Szenario laden  (bestehend)
# ---------------------------------------------------------------------------

def load_scenario(path: str) -> dict[str, Any]:
    """
    Liest eine YAML-Szenariodatei ein und gibt sie als Dictionary zurück.
    """
    with open(path, "r", encoding="utf-8") as file:
        scenario = yaml.safe_load(file)
    return scenario


# ---------------------------------------------------------------------------
# Hilfsfunktionen: Range-Verarbeitung und Tagtyp  (bestehend)
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

    if current_date in holiday_dates:
        return "sunday_holiday"

    weekday_index = current_datetime.weekday()  # Montag = 0, Sonntag = 6

    if weekday_index == 6:
        return "sunday_holiday"
    if weekday_index == 5:
        return "saturday"
    return "working_day"


# ---------------------------------------------------------------------------
# Zeitindex  (bestehend)
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
# Allgemeine Mischungsverteilung (lognormal, normal, beta, uniform)  #NEU
# ---------------------------------------------------------------------------

def sample_mixture(
    number_of_samples: int,
    mixture_components: list[dict[str, Any]],
    max_value: float | None = None,
    unit_description: str = "generic",
) -> np.ndarray:
    """
    Erzeugt Stichproben aus einer Mischung von Verteilungen.
    Unterstützte Verteilungen pro Komponente:
      - distribution: "lognormal" (default, falls nicht angegeben)
          Parameter: mu, sigma
      - distribution: "normal"
          Parameter: mu, sigma
      - distribution: "beta"
          Parameter: alpha, beta
      - distribution: "uniform"
          Parameter: low, high

    Optional:
      - shift_minutes: wird nach dem Sampling addiert (z.B. Arrival Times).
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


def sample_lognormal_mixture(
    number_of_samples: int,
    mixture_components: list[dict[str, Any]],
    max_value: float | None = None,
    unit_description: str = "generic",
) -> np.ndarray:
    """
    Rückwärtskompatibler Wrapper auf sample_mixture mit default 'lognormal'.
    """
    return sample_mixture(
        number_of_samples=number_of_samples,
        mixture_components=mixture_components,
        max_value=max_value,
        unit_description=unit_description,
    )


# ---------------------------------------------------------------------------
# Hilfsfunktion: Komponenten aus YAML "realisieren" (Ranges → konkrete Werte)  #NEU
# ---------------------------------------------------------------------------

def realize_mixture_components(
    component_templates: list[dict[str, Any]],
    allow_shift: bool = False,
) -> list[dict[str, Any]]:
    """
    Nimmt Komponenten-Templates aus YAML und erzeugt konkrete Komponenten
    für die Mischung (inkl. Ziehen der Ranges mit sample_from_range).
    """
    realized_mixture_components: list[dict[str, Any]] = []

    for component_template in component_templates:
        dist_type = component_template.get("distribution", "lognormal")
        realized_component: dict[str, Any] = {
            "distribution": dist_type,
            "weight": sample_from_range(component_template.get("weight", 1.0)),
        }

        if dist_type in ("lognormal", "normal"):
            realized_component["mu"] = sample_from_range(component_template["mu"])
            realized_component["sigma"] = sample_from_range(component_template["sigma"])
        elif dist_type == "beta":
            realized_component["alpha"] = sample_from_range(component_template["alpha"])
            realized_component["beta"] = sample_from_range(component_template["beta"])
        elif dist_type == "uniform":
            realized_component["low"] = sample_from_range(component_template["low"])
            realized_component["high"] = sample_from_range(component_template["high"])
        else:
            raise ValueError(f"Unbekannte Verteilung in YAML: {dist_type}")

        if allow_shift:
            realized_component["shift_minutes"] = component_template.get("shift_minutes", None)

        realized_mixture_components.append(realized_component)

    return realized_mixture_components


# ---------------------------------------------------------------------------
# Fahrzeugwahl nach Standortgewichtung (PKW/Transporter/...)  #NEU
# ---------------------------------------------------------------------------

def choose_vehicle_profile(
    vehicle_profiles: list[VehicleProfile],
    scenario: dict[str, Any],
) -> VehicleProfile:
    """
    Wählt ein Fahrzeugprofil aus der geladenen Flotte aus.
    Optional kann pro Standort in der YAML ein fleet_mix gesetzt werden, z.B.:

      vehicles:
        fleet_mix:
          PKW: 0.3
          Transporter: 0.7

    Falls fleet_mix fehlt oder ungültig ist, wird gleichverteilt gewählt.  #NEU
    """
    fleet_mix = scenario.get("vehicles", {}).get("fleet_mix", None)  #NEU

    # Fallback: gleichverteilte Auswahl  #NEU
    if not fleet_mix:
        return np.random.choice(vehicle_profiles)

    # nur Fahrzeuge berücksichtigen, deren Klasse im fleet_mix vorkommt  #NEU
    selectable_profiles = [
        vp for vp in vehicle_profiles if vp.vehicle_class in fleet_mix
    ]
    if not selectable_profiles:
        return np.random.choice(vehicle_profiles)

    weights = np.array(
        [float(fleet_mix[vp.vehicle_class]) for vp in selectable_profiles],
        dtype=float,
    )

    # Fallback bei nicht-positiven Summen  #NEU
    if weights.sum() <= 0.0:
        return np.random.choice(selectable_profiles)

    probs = weights / weights.sum()
    return np.random.choice(selectable_profiles, p=probs)


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

    realized_mixture_components = realize_mixture_components(
        component_templates_for_day_type,
        allow_shift=True,
    )

    if not realized_mixture_components:
        return []

    sampled_minutes_after_midnight = sample_mixture(
        number_of_samples=number_of_sessions_today,
        mixture_components=realized_mixture_components,
        max_value=None,
        unit_description="minutes",
    )

    # >>> NEU: auf [0, 24h) begrenzen, damit keine Overflows entstehen
    sampled_minutes_after_midnight = np.maximum(sampled_minutes_after_midnight, 0.0)
    sampled_minutes_after_midnight = np.minimum(sampled_minutes_after_midnight, 24.0 * 60.0 - 1.0)

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
    minimum_parking_duration_minutes = parking_duration_distribution.get("min_duration_minutes", 10.0)

    component_templates = parking_duration_distribution["components"]
    realized_mixture_components = realize_mixture_components(
        component_templates,
        allow_shift=False,
    )

    parking_durations_minutes = sample_mixture(
        number_of_samples=number_of_sessions,
        mixture_components=realized_mixture_components,
        max_value=maximum_parking_duration_minutes,
        unit_description="minutes",
    )
    # korrigiert alle gesampelten Werte auf den Bereich zwischen Minimum und Maximum
    parking_durations_minutes = np.clip(
        parking_durations_minutes,
        minimum_parking_duration_minutes,
        maximum_parking_duration_minutes,
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

    component_templates = soc_at_arrival_distribution["components"]
    realized_mixture_components = realize_mixture_components(
        component_templates,
        allow_shift=False,
    )

    soc_values = sample_mixture(
        number_of_samples=number_of_sessions,
        mixture_components=realized_mixture_components,
        max_value=maximum_soc_value,
        unit_description="soc_fraction",
    )

    # NEU: negative SoC-Werte vermeiden (bei Normalverteilungen möglich)
    soc_values = np.maximum(soc_values, 0.0)  #NEU

    return soc_values


# ---------------------------------------------------------------------------
# Ladesessions eines Tages
# ---------------------------------------------------------------------------

def build_charging_sessions_for_day(
    scenario: dict,
    day_start_datetime: datetime,
    vehicle_profiles: list[VehicleProfile],  #NEU
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

        # NEU: Fahrzeugwahl nach fleet_mix (PKW/Transporter/...)  #NEU
        vehicle_profile = choose_vehicle_profile(vehicle_profiles, scenario)  #NEU
        battery_capacity_kwh = float(vehicle_profile.battery_capacity_kwh)  #NEU

        delta_soc = max(target_soc - soc_at_arrival, 0.0)
        required_energy_kwh = delta_soc * battery_capacity_kwh

        if required_energy_kwh <= 0.0:          #NEU Fahrzeuge ohne Ladebedarf überspringen
            continue

        # Maximalleistung als Maximum der Kurve (zusätzlich zur SoC-abhängigen Grenze)
        max_vehicle_charging_power_kw = float(vehicle_profile.power_grid_kw.max())  #NEU

        charging_session: dict[str, Any] = {
            "arrival_time": arrival_time,
            "departure_time": departure_time,
            "soc_arrival": soc_at_arrival,
            "soc_target": target_soc,
            "battery_capacity_kwh": battery_capacity_kwh,
            "energy_required_kwh": required_energy_kwh,
            "delivered_energy_kwh": 0.0,  #NEU: bisher geladene Energie zur SoC-Berechnung
            "max_charging_power_kw": max_vehicle_charging_power_kw,
            "vehicle_name": vehicle_profile.name,       #NEU
            "vehicle_class": vehicle_profile.vehicle_class,  #NEU (für Auswertung/Debug)
            "soc_grid": vehicle_profile.soc_grid,       #NEU
            "power_grid_kw": vehicle_profile.power_grid_kw,  #NEU
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
    vehicle_csv_path = scenario["vehicles"]["vehicle_curve_csv"]  #NEU
    vehicle_profiles = load_vehicle_profiles_from_csv(vehicle_csv_path)  #NEU

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
                scenario, day_start_datetime, vehicle_profiles  #NEU
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
            vehicle_power_limit_kw = vehicle_power_at_soc(charging_session)  #NEU

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
                actual_power_kw = (
                    energy_delivered / (time_step_hours * charger_efficiency)
                    if time_step_hours > 0
                    else 0.0
                )
            else:
                energy_delivered = possible_energy_in_time_step_kwh
                charging_session["energy_required_kwh"] -= possible_energy_in_time_step_kwh
                actual_power_kw = allowed_power_kw

            charging_session["delivered_energy_kwh"] += energy_delivered  #NEU
            total_power_in_time_step_kw += actual_power_kw

        load_profile_kw[time_step_index] = total_power_in_time_step_kw

    return time_index, load_profile_kw, all_charging_sessions
