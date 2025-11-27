import yaml
import numpy as np
from datetime import datetime, timedelta


def load_scenario(path: str) -> Dict[str, Any]:
    """
    Lädt eine Szenario-YAML-Datei und gibt ein Dictionary zurück.

    Parameters
    ----------
    path : str
        Pfad zur YAML-Szenariodatei.

    Returns
    -------
    dict
        Das eingelesene Szenario als Dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        scenario = yaml.safe_load(f)
    return scenario


def create_time_index(scenario, start_datetime=None):
    """
    Erzeugt eine Liste von Zeitstempeln über den Simulationshorizont.

    Parameters
    ----------
    scenario : dict
        Eingelesenes Szenario.
    start_datetime : datetime or None
        Startzeitpunkt der Simulation (Default: heute 00:00).

    Returns
    -------
    list[datetime]
        Liste der Zeitstempel.
    """

    # 1) Falls start_datetime im Funktionsaufruf angegeben wurde → das hat PRIORITÄT
    if start_datetime is not None:
        base = start_datetime

    # 2) Falls in YAML definiert
    elif "start_datetime" in scenario:
        base = datetime.fromisoformat(scenario["start_datetime"])

    # 3) Fallback: Heute 00:00
    else:
        base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    dt_min = scenario["time_resolution_min"]
    n_days = scenario["sim_horizon_days"]
    total_minutes = n_days * 24 * 60
    n_steps = int(total_minutes / dt_min)

    dt = timedelta(minutes=dt_min)
    timestamps = [base + i * dt for i in range(n_steps)]

    return timestamps


def sample_mixture_lognormal(n_samples, components, max_value=None, unit="generic"):
    """
    Zieht Zufallswerte aus einer Mischung von Lognormalverteilungen.

    Parameters
    ----------
    n_samples : int
        Anzahl der zu erzeugenden Stichproben.
    components : list[dict]
        Liste von Komponenten mit Feldern 'mu', 'sigma', 'weight'.
        Optional: weitere Felder wie 'shift_minutes'.
    max_value : float or None
        Falls angegeben, werden Werte auf diesen Maximalwert begrenzt.
    unit : str
        Nur für Debug/Lesbarkeit, wird nicht verwendet.

    Returns
    -------
    np.ndarray
        Array der gezogenen Werte (float).
    """
    if n_samples <= 0:
        return np.array([])

    weights = np.array([c["weight"] for c in components], dtype=float)
    weights = weights / weights.sum()

    # Welcher Sample gehört zu welcher Komponente?
    comp_idx = np.random.choice(len(components), size=n_samples, p=weights)

    samples = np.zeros(n_samples, dtype=float)

    for i, idx in enumerate(comp_idx):
        c = components[idx]
        mu = float(c["mu"])
        sigma = float(c["sigma"])

        x = np.random.lognormal(mean=mu, sigma=sigma)

        # Optionaler Shift in Minuten, nur für arrival_time_distribution relevant
        if "shift_minutes" in c:
            x = x * 60.0 + float(c["shift_minutes"])

        samples[i] = x

    if max_value is not None:
        samples = np.minimum(samples, max_value)

    return samples


def get_weekday_label(dt):
    """
    Gibt Wochentagskürzel wie 'Mon', 'Tue', ... zurück.
    """
    return dt.strftime("%a")  # Mon, Tue, Wed, ...


def sample_arrivals_for_day(scenario, day_start):
    """
    Erzeugt Ankunftszeitpunkte für einen Tag (als datetime-Objekte).

    Nutzt arrival_time_distribution + weekday_weight + expected_sessions_per_charger_per_day.
    """
    arrival_cfg = scenario["arrival_time_distribution"]
    weekday = get_weekday_label(day_start)

    weekday_weight = arrival_cfg["weekday_weight"][weekday]
    n_chargers = scenario["site"]["number_chargers"]
    sessions_per_charger = scenario["vehicles"]["expected_sessions_per_charger_per_day"]

    n_today = int(n_chargers * sessions_per_charger * weekday_weight)
    if n_today <= 0:
        return []

    components_all = arrival_cfg["components_per_weekday"]
    components = components_all.get(weekday, [])

    samples_min = sample_mixture_lognormal(
        n_samples=n_today,
        components=components,
        max_value=None,
        unit="minutes",
    )

    arrivals = [day_start + timedelta(minutes=float(m)) for m in samples_min]
    arrivals.sort()
    return arrivals


def sample_parking_durations(scenario, n_sessions):
    """
    Erzeugt Parkdauern in Minuten für n_sessions.
    """
    cfg = scenario["parking_duration_distribution"]
    components = cfg["components"]
    max_minutes = cfg["max_duration_minutes"]

    durations = sample_mixture_lognormal(
        n_samples=n_sessions,
        components=components,
        max_value=max_minutes,
        unit="minutes",
    )
    return durations


def sample_soc_at_arrival(scenario, n_sessions):
    """
    Erzeugt SoC-Werte bei Ankunft (0..1) für n_sessions.
    """
    cfg = scenario["soc_at_arrival_distribution"]
    components = cfg["components"]
    max_soc = cfg["max_soc"]

    soc_values = sample_mixture_lognormal(
        n_samples=n_sessions,
        components=components,
        max_value=max_soc,
        unit="soc_fraction",
    )
    return soc_values


def build_sessions_for_day(scenario, day_start):
    """
    Baut für einen Tag eine Liste von Ladesessions (als Dictionaries).

    Jede Session hat:
      - arrival_time
      - departure_time
      - soc_arrival
      - soc_target
      - battery_capacity_kwh
      - energy_required_kwh
      - max_pwr_vehicle_kw
    """
    arrivals = sample_arrivals_for_day(scenario, day_start)
    n = len(arrivals)
    if n == 0:
        return []

    durations_min = sample_parking_durations(scenario, n)
    soc_arrival = sample_soc_at_arrival(scenario, n)

    soc_target = scenario["vehicles"]["soc_target"]
    capacity = scenario["vehicles"]["battery_capacity_kwh"]
    max_pwr_vehicle_kw = scenario["vehicles"]["max_pwr_vehicle_kw"]

    sessions = []
    for i in range(n):
        a = arrivals[i]
        d = a + timedelta(minutes=float(durations_min[i]))
        soc_a = float(soc_arrival[i])
        delta_soc = max(soc_target - soc_a, 0.0)
        energy_req = delta_soc * capacity

        sessions.append(
            {
                "arrival_time": a,
                "departure_time": d,
                "soc_arrival": soc_a,
                "soc_target": soc_target,
                "battery_capacity_kwh": capacity,
                "energy_required_kwh": energy_req,
                "max_pwr_vehicle_kw": max_pwr_vehicle_kw,
            }
        )
    return sessions


def simulate_load_profile(scenario, start_datetime=None):
    """
    Führt die Lastgang-Simulation über den Simulationshorizont aus.

    Sehr einfache Strategie:
      - Ladestrategie 'immediate'
      - alle aktiven Sessions teilen sich die verfügbare Leistung
      - Begrenzung durch grid_limit und Anzahl Ladepunkte
    """
    timestamps = create_time_index(scenario, start_datetime)
    dt_min = scenario["time_resolution_min"]
    dt_h = dt_min / 60.0

    n_steps = len(timestamps)
    load_kw = np.zeros(n_steps, dtype=float)

    grid_limit_kw = scenario["site"]["grid_limit_p_avb_kw"]
    rated_power_kw = scenario["site"]["rated_power_kw"]
    n_chargers = scenario["site"]["number_chargers"]
    charger_eff = scenario["site"]["charger_efficiency"]

    # Sessions pro Tag vorbereiten
    sessions_all = []
    if timestamps:
        first_day = timestamps[0].replace(hour=0, minute=0, second=0, microsecond=0)
        sim_days = scenario["sim_horizon_days"]
        for d in range(sim_days):
            day_start = first_day + timedelta(days=d)
            sessions_all.extend(build_sessions_for_day(scenario, day_start))

    # Haupt-Loop über alle Zeitschritte
    for i, t in enumerate(timestamps):
        # aktive Sessions in diesem Zeitschritt
        active = [
            s for s in sessions_all
            if s["arrival_time"] <= t < s["departure_time"]
            and s["energy_required_kwh"] > 0.0
        ]

        if not active:
            load_kw[i] = 0.0
            continue

        # maximal verfügbare Leistung am Standort
        site_p_max = min(grid_limit_kw, n_chargers * rated_power_kw)

        # simple „faire“ Verteilung: alle bekommen gleich viel, begrenzt durch Fahrzeugsmaximalleistung
        n_active = len(active)
        p_per_session = site_p_max / n_active

        total_power = 0.0
        for s in active:
            p_session = min(p_per_session, s["max_pwr_vehicle_kw"])
            # Energie, die in diesem Zeitschritt übertragen werden könnte (unter Idealannahmen)
            e_possible = p_session * dt_h * charger_eff

            if e_possible >= s["energy_required_kwh"]:
                # Session ist fertig in diesem Schritt
                e_delivered = s["energy_required_kwh"]
                s["energy_required_kwh"] = 0.0
                # Anpassung der Leistung, falls gewünscht, hier vereinfachend ignoriert
            else:
                e_delivered = e_possible
                s["energy_required_kwh"] -= e_delivered

            total_power += p_session

        load_kw[i] = total_power

    return timestamps, load_kw, sessions_all
