import yaml
import numpy as np
from pathlib import Path
import csv  # CSV-Dateien werden genutzt (Fahrzeug-Ladekurven, Erzeugungssignal, Marktpreise)
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Any


# =============================================================================
# Modulüberblick (Erläuterung in 3. Person)
# =============================================================================
# Dieses Modul simuliert den Lastgang eines Ladepark-Standorts über einen konfigurierten Zeithorizont.
# Es kombiniert:
#   1) eine stochastische Session-Generierung (Ankunftszeiten, Parkdauer, SoC bei Ankunft),
#   2) fahrzeugspezifische Ladekurven (max. Ladeleistung abhängig vom SoC),
#   3) Standortgrenzen (Anzahl Ladepunkte, Leistung pro Ladepunkt, Netzanschlusslimit),
#   4) optionales Lademanagement (immediate, market, generation),
#   5) optionales Debug-Logging für Auswertungen im Notebook.
#
# Ziel ist ein Zeitprofil der EV-Ladeleistung (kW) sowie Session-Details für KPI-Analysen.


# =============================================================================
# 0) Projekt-/Pfad-Utilities
# =============================================================================

def resolve_path_relative_to_scenario(scenario: dict[str, Any], p: str) -> str:
    """
    Diese Funktion löst Dateipfade robust auf.

    Sie sorgt dafür, dass:
      - absolute Pfade unverändert bleiben,
      - relative Pfade relativ zum YAML-Ordner interpretiert werden.

    Dadurch können Szenario-YAMLs portable gehalten werden (z.B. in Git),
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

    Sie wird in mehreren CSV-Ladern verwendet, um typische Excel-/Exportformate abzufangen.
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

    # nur '.' oder kein Separator
    return float(s)


# =============================================================================
# 0c) HTML-Statusausgabe (optional im Notebook)
# =============================================================================

def show_strategy_status_html(strategy: str, status: str) -> None:
    """
    Diese Funktion zeigt im Notebook eine farbige Statuszeile an.

    Sie wird genutzt, um dem Anwender schnell zu visualisieren:
      - welche Strategie aktiv ist,
      - ob die Strategie aktiv (Signal geladen) oder inaktiv ist.

    Falls IPython nicht verfügbar ist, fällt sie auf einen normalen print zurück.

    Erwartete status-Werte:
      - "ACTIVE" | "INACTIVE" | "IMMEDIATE"
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
    Diese Datenklasse repräsentiert ein Fahrzeugprofil.

    Enthalten sind:
      - Name/Modell,
      - Batteriekapazität (kWh),
      - Fahrzeugklasse (z.B. PKW / Transporter),
      - SoC-Stützstellen (0..1),
      - zugehörige maximale Ladeleistung (kW) pro SoC-Stützstelle.

    Die Arrays soc_grid und power_grid_kw bilden gemeinsam eine Ladekurve ab,
    die per Interpolation genutzt werden kann.
    """
    name: str
    battery_capacity_kwh: float
    vehicle_class: str
    soc_grid: np.ndarray
    power_grid_kw: np.ndarray


def load_vehicle_profiles_from_csv(path: str) -> list[VehicleProfile]:
    """
    Diese Funktion lädt Fahrzeugprofile aus einer CSV-Datei.

    Erwartete Struktur (Delimiter ';'):
      Zeile 1: Hersteller (wird ignoriert)
      Zeile 2: Modellnamen
      Zeile 3: Fahrzeugklasse
      Zeile 4: Kapazität (kWh)
      Zeile 5: "SoC [%]" (Header)
      ab Zeile 6: SoC-Werte in % + Ladeleistungen je Fahrzeug

    Die Funktion gibt eine Liste von VehicleProfile zurück.
    Fahrzeuge ohne gültige Kapazität oder ohne Ladekurve werden verworfen.
    """
    vehicle_profiles: list[VehicleProfile] = []

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";")

        # Zeile 1: Hersteller (nicht benötigt)
        _brands_row = next(reader, None)

        # Zeile 2: Modelle
        model_row = next(reader, None)
        if not model_row or len(model_row) < 2:
            return []
        model_names = [m.strip() for m in model_row[1:]]

        # Zeile 3: Klassen
        class_row = next(reader, None)
        if not class_row or len(class_row) < 2:
            return []
        vehicle_classes = [c.strip() if c.strip() != "" else "PKW" for c in class_row[1:]]

        # Zeile 4: Kapazitäten
        capacity_row = next(reader, None)
        if not capacity_row or len(capacity_row) < 2:
            return []
        raw_capacities = capacity_row[1:]

        # Zeile 5: SoC-Header (nur überspringen)
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

        # Konsistenz: Arrays auf gleiche Länge schneiden
        num_vehicles = min(len(model_names), len(vehicle_classes), len(capacities_kwh))
        model_names = model_names[:num_vehicles]
        vehicle_classes = vehicle_classes[:num_vehicles]
        capacities_kwh = capacities_kwh[:num_vehicles]

        # Pro Fahrzeug werden SoC- und Power-Punkte gesammelt
        soc_lists: list[list[float]] = [[] for _ in range(num_vehicles)]
        power_lists: list[list[float]] = [[] for _ in range(num_vehicles)]

        # Ab Zeile 6: SoC + Ladeleistung
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

        # Profile finalisieren
        for i in range(num_vehicles):
            name = model_names[i]
            vclass = vehicle_classes[i]
            cap = capacities_kwh[i]

            if np.isnan(cap) or len(soc_lists[i]) == 0:
                continue

            soc_grid = np.array(soc_lists[i], dtype=float)
            power_grid = np.array(power_lists[i], dtype=float)

            # Sortierung stellt sicher, dass Interpolation korrekt funktioniert
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
    Diese Funktion berechnet die maximal mögliche Ladeleistung (kW) eines Fahrzeugs
    bei aktuellem SoC.

    Der aktuelle SoC ergibt sich aus:
      SoC bei Ankunft + (bisher geladene Energie / Batteriekapazität)

    Anschließend wird die Ladekurve (soc_grid, power_grid_kw) linear interpoliert.
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
    Diese Funktion lädt ein YAML-Szenario und ergänzt einen internen Kontextpfad.

    Der Schlüssel scenario["_scenario_dir"] wird gesetzt, damit spätere Pfade
    relativ zum YAML-Standort aufgelöst werden können.
    """
    with open(path, "r", encoding="utf-8") as file:
        scenario = yaml.safe_load(file)

    scenario["_scenario_dir"] = str(Path(path).resolve().parent)
    return scenario


# =============================================================================
# 3) Hilfsfunktionen: Ranges, Feiertage, Zeitindex, Disconnect
# =============================================================================

def sample_from_range(value_definition: Any) -> float:
    """
    Diese Funktion interpretiert YAML-Werte, die entweder:
      - als fester Wert (Skalar) angegeben sind, oder
      - als Bereich [min, max] für eine Uniform-Ziehung.

    Dadurch lassen sich Unsicherheiten in Parametern stochastisch modellieren.
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
    Diese Funktion berechnet eine Menge an Feiertagen für den Simulationszeitraum.

    Sie unterstützt:
      - automatische Feiertage via Paket 'holidays' (Land + optionales Bundesland),
      - zusätzlich manuell definierte Feiertage über YAML.

    Das Ergebnis ist ein set[date], das später für die Tagesklassifikation
    (working_day/saturday/sunday_holiday) verwendet wird.
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
    Diese Funktion klassifiziert einen Zeitpunkt als:
      - working_day
      - saturday
      - sunday_holiday

    Feiertage werden als sunday_holiday behandelt, weil in vielen Use-Cases
    das Betriebsverhalten an Feiertagen einem Sonntag ähnelt.
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

    Der Startpunkt wird gesetzt durch:
      - explizit übergebene start_datetime (Parameter),
      - oder scenario["start_datetime"],
      - sonst: aktueller Tag 00:00.

    Der Zeitschritt wird über scenario["time_resolution_min"] gesteuert.
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


def sample_disconnect_delay_hours(cfg_value: Any) -> float | None:
    """
    Diese Funktion interpretiert Konfigurationswerte für eine optionale Verzögerung
    (in Stunden), wann ein Fahrzeug nach Erreichen des Ziel-SoC abgesteckt wird.

    Sie unterstützt:
      - None: kein Disconnect
      - bool: True => 0h, False => kein Disconnect
      - Zahl: fixe Stunden
      - [min, max]: Uniform-Ziehung in Stunden

    Hinweis: In der aktuell hart kodierten "drive_off"-Variante wird diese Logik
    nicht genutzt, kann aber für spätere Policies reaktiviert werden.
    """
    if cfg_value is None:
        return None

    if isinstance(cfg_value, bool):
        return 0.0 if cfg_value else None

    if isinstance(cfg_value, (int, float)):
        return max(0.0, float(cfg_value))

    if isinstance(cfg_value, (list, tuple)):
        if len(cfg_value) == 1:
            return max(0.0, float(cfg_value[0]))
        if len(cfg_value) == 2:
            return max(0.0, float(np.random.uniform(cfg_value[0], cfg_value[1])))

    raise ValueError(
        "❌ Ungültiger Wert für site.disconnect_when_full. "
        "Erlaubt: true | false | Zahl | [min, max]"
    )


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
    Diese Funktion zieht Stichproben aus einer Mischverteilung.

    Jede Komponente besitzt:
      - weight (Gewicht),
      - distribution (Typ),
      - Parameter (mu/sigma, alpha/beta, low/high),
      - optional shift_minutes (z.B. Umrechnung Stunden->Minuten in Arrival-Peaks).

    Das Ergebnis ist ein numpy-Array mit Samples.
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
    Diese Funktion realisiert stochastische YAML-Templates von Mischkomponenten.

    YAML kann Parameter als Bereiche angeben (z.B. mu: [7.5, 9.0]).
    Diese Funktion zieht daraus konkrete Werte, sodass sample_mixture
    anschließend mit festen Parametern arbeiten kann.
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
    Diese Funktion wählt ein Fahrzeugprofil aus der Flotte aus.

    Wenn scenario["vehicles"]["fleet_mix"] definiert ist, wird die Ziehung
    entsprechend der Klassengewichte durchgeführt (z.B. 98% PKW, 2% Transporter).

    Ist keine Gewichtung vorhanden oder passt keine Klasse, wird uniform gewählt.
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
# 6) Session-Generierung
# =============================================================================

def sample_arrival_times_for_day(
    scenario: dict,
    day_start_datetime: datetime,
    holiday_dates: set[date],
) -> list[datetime]:
    """
    Diese Funktion erzeugt Ankunftszeiten für einen Tag.

    Schritte:
      1) Tagesart bestimmen (working_day/saturday/sunday_holiday).
      2) Erwartete Anzahl Sessions berechnen (Ladepunkte * Sessions/LP * Tagesgewicht).
      3) Ankunftszeiten aus einer Mischverteilung (Peak-Struktur) ziehen.
      4) Ergebnisse auf [0..24h) begrenzen und sortieren.
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
    Diese Funktion zieht Parkdauern (in Minuten) aus einer Mischverteilung.

    Das Ergebnis wird auf [min_duration_minutes, max_duration_minutes] geclippt,
    damit keine unrealistischen Extremwerte entstehen.
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
    Diese Funktion zieht SoC-Werte bei Ankunft aus einer Mischverteilung.

    Der SoC wird nach unten bei 0 begrenzt und nach oben durch max_soc.
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
    Diese Funktion erzeugt Sessions (Ankunft/Abfahrt + Fahrzeugdaten) für einen Tag.

    Für jede Session werden:
      - Ankunft und Abfahrt gesetzt,
      - ein Fahrzeugprofil gezogen,
      - der Energiebedarf bis zum Ziel-SoC berechnet,
      - initiale Session-Felder für die Simulation vorbereitet.
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
                "session_id": f"{day_start_datetime.date()}_{i}",
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

                # Diese Felder werden von der Generation-Strategie (Market-Fallback) genutzt:
                "preferred_market_slot_indices": [],
                "preferred_market_ptr": 0,
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
    Diese Funktion liest eine Zeitreihe aus einer CSV-Datei.

    Annahmen:
      - Spalte 1 enthält Zeitstempel (dd.mm.YYYY HH:MM oder dd.mm.YY HH:MM).
      - value_col_1_based gibt die 1-basierte Spalte für den Signalwert an.
      - delimiter ist standardmäßig ';'.

    Ergebnis:
      - dict[datetime, float] mit Zeitstempel -> Rohwert
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
    """
    Diese Funktion rundet einen Zeitstempel auf das Raster der Auflösung ab.
    Beispiel: resolution_min=15, 10:07 -> 10:00.
    """
    discard = dt.minute % resolution_min
    return dt.replace(minute=dt.minute - discard, second=0, microsecond=0)


def lookup_signal(strategy_map: dict[datetime, float], ts: datetime, resolution_min: int) -> float | None:
    """
    Diese Funktion sucht einen Signalwert, indem sie ts vorher auf das Raster abrundet.
    Ist kein Wert vorhanden, wird None geliefert.
    """
    return strategy_map.get(floor_to_resolution(ts, resolution_min), None)


# =============================================================================
# 7b) Harte Validierung
# =============================================================================

def assert_strategy_csv_covers_simulation(
    strategy_map: dict[datetime, float],
    time_index: list[datetime],
    strategy_resolution_min: int,
    charging_strategy: str,
    strategy_csv_path: str,
) -> None:
    """
    Diese Funktion prüft, ob ein Strategie-CSV den kompletten Simulationszeitraum abdeckt.

    Es werden zwei Fälle geprüft:
      1) CSV beginnt zu spät oder endet zu früh (Zeitraumabdeckung).
      2) Es fehlen einzelne Zeitstempel innerhalb des Zeitraums (Lücken).

    Bei Fehler wird ein ValueError mit konkreten Hinweisen ausgelöst.
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
# 7c) Normalisierung der CSV-Werte
# =============================================================================

def convert_strategy_value_to_internal(
    charging_strategy: str,
    raw_value: float,
    strategy_unit: str,
    step_hours: float,
) -> float:
    """
    Diese Funktion normalisiert CSV-Rohwerte auf interne Einheiten.

    Intern wird verwendet:
      - generation: kW
      - market: €/kWh

    Damit kann die Simulation unabhängig vom Input-Format arbeiten.
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
# 7e) Optional: Basislast aus CSV (für Generation)
# =============================================================================

def build_base_load_series(
    scenario: dict[str, Any],
    timestamps: list[datetime],
    base_load_resolution_min: int = 15,
) -> np.ndarray | None:
    """
    Diese Funktion baut eine Basislast-Zeitreihe (kW) für den Standort.

    Priorität:
      1) base_load_csv (wenn gesetzt) -> CSV wird eingelesen und auf kW normiert.
      2) base_load_kw (konstanter Wert).
      3) sonst: None.

    Die Basislast wird insbesondere in der Generation-Strategie benötigt,
    um PV-Überschuss = PV - Basislast zu berechnen.
    """
    if not timestamps:
        return None

    site_cfg = scenario.get("site", {}) or {}

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
            else:
                series_kw[i] = float(v) * 1000.0 / step_hours

        return series_kw

    base_load_kw = site_cfg.get("base_load_kw", None)
    if base_load_kw is not None:
        return np.full(len(timestamps), float(base_load_kw), dtype=float)

    return None


# =============================================================================
# 8) Slack & Slot-Heuristik
# =============================================================================

def _slack_minutes_for_session(
    s: dict[str, Any],
    ts: datetime,
    rated_power_kw: float,
    charger_efficiency: float,
) -> float:
    """
    Diese Funktion berechnet den "Slack" einer Session in Minuten.

    Slack ist die Zeitreserve bis zur Abfahrt:
      Slack = (Restzeit bis Abfahrt) - (benötigte Ladezeit bei maximal möglicher Leistung)

    Ein kleiner Slack bedeutet hohe Dringlichkeit (Feasibility-Absicherung).
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
    Diese Funktion baut für MARKET pro Session eine priorisierte Slot-Liste.

    Idee:
      - Alle Zeitschritte im [arrival, departure) werden bewertet.
      - Bei market wird der Score = Preis in €/kWh (niedrig = besser).
      - Slots werden nach Score sortiert; daraus entsteht eine Präferenzliste.

    In der Simulation kann dann geprüft werden, ob der aktuelle Zeitschritt
    in der Präferenzliste vorne liegt.
    """
    if charging_strategy != "market":
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
                score = float(sig_internal)

            scored.append((score, idx))

        scored.sort(key=lambda x: x[0])
        s["preferred_slot_indices"] = [idx for _, idx in scored]
        s["preferred_ptr"] = 0


def _build_preferred_market_slots_for_generation_fallback(
    all_sessions: list[dict[str, Any]],
    time_index: list[datetime],
    time_to_idx: dict[datetime, int],
    market_map: dict[datetime, float] | None,
    market_resolution_min: int,
    market_unit: str,
    step_hours: float,
) -> None:
    """
    Diese Funktion baut pro Session eine nach günstigen Preisen sortierte Slot-Liste,
    die ausschließlich innerhalb der Generation-Strategie für Grid-Fallback genutzt wird.

    Ergebnisfelder pro Session:
      - preferred_market_slot_indices
      - preferred_market_ptr
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
            s["preferred_market_slot_indices"] = []
            s["preferred_market_ptr"] = 0
            continue

        start_idx = time_to_idx[a]
        end_idx = time_to_idx[d]
        if end_idx <= start_idx:
            s["preferred_market_slot_indices"] = []
            s["preferred_market_ptr"] = 0
            continue

        idxs = list(range(start_idx, end_idx))

        scored: list[tuple[float, int]] = []
        for idx in idxs:
            t = time_index[idx]
            raw = lookup_signal(market_map, t, market_resolution_min)
            if raw is None:
                score = 1e30
            else:
                price_eur_per_kwh = convert_strategy_value_to_internal(
                    charging_strategy="market",
                    raw_value=float(raw),
                    strategy_unit=market_unit,
                    step_hours=step_hours,
                )
                score = float(price_eur_per_kwh)

            scored.append((score, idx))

        scored.sort(key=lambda x: x[0])
        s["preferred_market_slot_indices"] = [idx for _, idx in scored]
        s["preferred_market_ptr"] = 0


# =============================================================================
# 8b) GENERATION – Rolling-Horizon PV-Reservation
# =============================================================================

def _session_power_cap_kw(s: dict[str, Any], rated_power_kw: float) -> float:
    """
    Diese Funktion berechnet ein Leistungs-Cap pro Session auf Basis:
      - Ladepunktleistung (rated_power_kw)
      - session["max_charging_power_kw"] (Fahrzeuglimit)
    """
    return max(0.0, min(float(rated_power_kw), float(s.get("max_charging_power_kw", rated_power_kw))))


def _idx_of_last_step_before_departure(
    departure_time: datetime,
    time_index: list[datetime],
    time_resolution_min: int,
) -> int:
    """
    Diese Funktion findet den letzten Zeitschrittindex, der noch vor der Abfahrt liegt.

    Dadurch kann die Generation-Planung nur in einem gültigen Ladefenster
    (von "jetzt" bis "departure") planen.
    """
    if not time_index:
        return 0

    d = floor_to_resolution(departure_time, time_resolution_min)

    time_to_idx = {t: i for i, t in enumerate(time_index)}
    while d not in time_to_idx and d > time_index[0]:
        d -= timedelta(minutes=time_resolution_min)

    if d in time_to_idx:
        return int(time_to_idx[d])

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
    Diese Funktion plant PV-Zuweisungen (commitments) für aktuell anwesende Sessions.

    Vorgehen:
      - PV-Überschuss in der Zukunft wird als Ressourcenbudget betrachtet.
      - Sessions werden nach Dringlichkeit (Abfahrt) sortiert.
      - Für jede Session wird versucht, so viel Energie wie möglich in die
        Zeitfenster mit hohem PV-Überschuss zu legen.
      - Wenn PV nicht reicht, wird die Session als "grid_needed" markiert.

    Rückgabe:
      - pv_commit_kw: geplante PV-Leistung pro Zeitschritt
      - grid_needed_session_ids: Menge von Session-IDs, die Grid-Anteil benötigen
    """
    n = len(pv_surplus_series_kw)
    pv_commit_kw = np.zeros(n, dtype=float)

    if not present_sessions or i_now >= n:
        return pv_commit_kw, set()

    pv_available_kw = np.array(pv_surplus_series_kw, dtype=float)
    pv_available_kw[:i_now] = 0.0

    present_sorted = sorted(present_sessions, key=lambda s: (s["departure_time"], s["arrival_time"]))

    grid_needed_session_ids: set[int] = set()

    for s in present_sorted:
        e_need_kwh = float(s.get("energy_required_kwh", 0.0))
        if e_need_kwh <= 1e-9:
            continue

        cap_kw = _session_power_cap_kw(s, rated_power_kw)
        if cap_kw <= 1e-9:
            grid_needed_session_ids.add(id(s))
            continue

        end_idx = _idx_of_last_step_before_departure(
            departure_time=s["departure_time"],
            time_index=time_index,
            time_resolution_min=time_resolution_min,
        )
        start_idx = i_now
        if end_idx <= start_idx:
            grid_needed_session_ids.add(id(s))
            continue

        slot_indices = list(range(start_idx, min(end_idx + 1, n)))
        slot_indices.sort(key=lambda j: (-pv_available_kw[j], j))

        e_remaining_kwh = e_need_kwh

        for j in slot_indices:
            if e_remaining_kwh <= 1e-9:
                break

            avail_kw = float(pv_available_kw[j])
            if avail_kw <= 1e-9:
                continue

            max_kw_for_energy = e_remaining_kwh / (time_step_hours * charger_efficiency)
            take_kw = min(avail_kw, cap_kw, max_kw_for_energy)

            if take_kw <= 1e-9:
                continue

            pv_commit_kw[j] += take_kw
            pv_available_kw[j] -= take_kw
            e_remaining_kwh -= take_kw * time_step_hours * charger_efficiency

        if e_remaining_kwh > 1e-6:
            grid_needed_session_ids.add(id(s))

    return pv_commit_kw, grid_needed_session_ids


# =============================================================================
# 8c) Reporting Helper: Strategy signal series (aligned)
# =============================================================================

def build_strategy_signal_series(
    scenario: dict[str, Any],
    timestamps: list[datetime],
    charging_strategy: str,
    normalize_to_internal: bool = True,
    strategy_resolution_min: int = 15,
) -> tuple[np.ndarray | None, str | None]:
    """
    Diese Funktion erstellt eine an die Simulationstimestamps ausgerichtete Strategie-Zeitreihe.

    Sie wird primär für Notebook-Plotting genutzt.

    Verhalten:
      - charging_strategy="market": Marktpreise werden zu €/kWh normalisiert (optional)
      - charging_strategy="generation": Erzeugung wird zu kW normalisiert (optional)

    Rückgabe:
      - series (numpy array) oder None
      - y_label (Achsenbeschriftung) oder None
    """
    strat = (charging_strategy or "immediate").lower()
    if strat not in ("market", "generation"):
        return None, None
    if not timestamps:
        return None, None

    site_cfg = scenario.get("site", {}) or {}

    if strat == "market":
        unit = str(site_cfg.get("market_strategy_unit", "") or "").strip()
        csv_rel = site_cfg.get("market_strategy_csv", None)
        col_1_based = site_cfg.get("market_strategy_value_col", None)

        if unit not in ("€/MWh", "€/kWh"):
            raise ValueError("❌ Abbruch: 'site.market_strategy_unit' muss '€/MWh' oder '€/kWh' sein.")
        if not csv_rel or not isinstance(col_1_based, int) or col_1_based < 2:
            raise ValueError("❌ Abbruch: 'site.market_strategy_csv' oder 'site.market_strategy_value_col' fehlt/ungültig.")

        csv_path = resolve_path_relative_to_scenario(scenario, str(csv_rel))
        strat_map = read_strategy_series_from_csv_first_col_time(
            csv_path=csv_path,
            value_col_1_based=int(col_1_based),
            delimiter=";",
        )

        step_hours = strategy_resolution_min / 60.0
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
    unit = str(site_cfg.get("generation_strategy_unit", "") or "").strip()
    csv_rel = site_cfg.get("generation_strategy_csv", None)
    col_1_based = site_cfg.get("generation_strategy_value_col", None)

    if unit not in ("kW", "kWh", "MWh"):
        raise ValueError("❌ Abbruch: 'site.generation_strategy_unit' muss 'kW', 'kWh' oder 'MWh' sein.")
    if not csv_rel or not isinstance(col_1_based, int) or col_1_based < 2:
        raise ValueError("❌ Abbruch: 'site.generation_strategy_csv' oder 'site.generation_strategy_value_col' fehlt/ungültig.")

    csv_path = resolve_path_relative_to_scenario(scenario, str(csv_rel))
    strat_map = read_strategy_series_from_csv_first_col_time(
        csv_path=csv_path,
        value_col_1_based=int(col_1_based),
        delimiter=";",
    )

    step_hours = strategy_resolution_min / 60.0
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


# =============================================================================
# 8d) Reporting / KPI Helper (Notebook-freundlich, ohne Plotting)
# =============================================================================

def summarize_sessions(
    sessions: list[dict[str, Any]],
    eps_kwh: float = 1e-6,
) -> dict[str, Any]:
    """
    Diese Funktion erzeugt eine KPI-Zusammenfassung über alle Sessions.

    Metriken:
      - num_sessions_total: alle Sessions mit Ladebedarf
      - num_sessions_plugged: Sessions, die physisch angeschlossen wurden
      - num_sessions_rejected: Sessions, die wegen fehlender Ladepunkte abgewiesen wurden
      - not_reached_rows: Liste der Sessions, die ihr Ziel nicht erreicht haben
    """
    if sessions is None:
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
        if float(s.get("energy_required_kwh", 0.0)) > float(eps_kwh):
            arrival = s["arrival_time"]
            departure = s["departure_time"]
            not_reached_rows.append(
                {
                    "vehicle_name": s.get("vehicle_name", ""),
                    "arrival_time": arrival,
                    "departure_time": departure,
                    "parking_hours": (departure - arrival).total_seconds() / 3600.0,
                    "delivered_energy_kwh": float(s.get("delivered_energy_kwh", 0.0)),
                    "remaining_energy_kwh": float(s.get("energy_required_kwh", 0.0)),
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
    Diese Funktion erzeugt eine Liste der Tage je Tagtyp (Kalenderperspektive).

    Sie wird für Notebook-Auswertungen genutzt, um Histogramme
    nach Tagesarten (working_day/saturday/sunday_holiday) zu gruppieren.
    """
    out: dict[str, list[date]] = {"working_day": [], "saturday": [], "sunday_holiday": []}

    for i in range(int(horizon_days)):
        d = start_datetime.date() + timedelta(days=i)
        dt_mid = datetime(d.year, d.month, d.day, 12, 0)
        dt_type = determine_day_type_with_holidays(dt_mid, holiday_dates)
        out.setdefault(dt_type, []).append(d)

    return out


def group_sessions_by_day(
    sessions: list[dict[str, Any]],
    only_plugged: bool = False,
) -> dict[date, list[dict[str, Any]]]:
    """
    Diese Funktion gruppiert Sessions nach Ankunftsdatum.

    Parameter:
      - only_plugged=True filtert auf Sessions, die physisch geladen haben.
    """
    out: dict[date, list[dict[str, Any]]] = {}

    if not sessions:
        return out

    for s in sessions:
        if only_plugged and s.get("_plug_in_time") is None:
            continue
        d = s["arrival_time"].date()
        out.setdefault(d, []).append(s)

    return out


def build_pv_unused_table(debug_rows: list[dict[str, Any]]):
    """
    Diese Funktion erstellt eine Tabelle für Zeitpunkte mit ungenutztem PV-Überschuss.

    Voraussetzung:
      - debug_rows wurde in simulate_load_profile(record_debug=True) aufgezeichnet.

    Rückgabe:
      - pandas.DataFrame (kann leer sein)
      - None, wenn pandas/numpy nicht verfügbar ist
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

    df_pv = df[df["pv_surplus_kw"] > 1e-6].copy()
    if len(df_pv) == 0:
        return pd.DataFrame()

    grid_col = "grid_import_kw_site" if "grid_import_kw_site" in df_pv.columns else None

    agg = {
        "pv_surplus_kw": ("pv_surplus_kw", "first"),
        "site_power_kw": ("site_total_power_kw", "first"),
        "n_rows": ("vehicle_name", "count") if "vehicle_name" in df_pv.columns else ("pv_surplus_kw", "size"),
    }
    if grid_col:
        agg["grid_import_kw"] = (grid_col, "first")

    pv_steps = df_pv.groupby("ts", as_index=False).agg(**agg)

    pv_steps["pv_used_kw"] = np.minimum(pv_steps["pv_surplus_kw"], pv_steps["site_power_kw"])
    pv_steps["pv_unused_kw"] = (pv_steps["pv_surplus_kw"] - pv_steps["pv_used_kw"]).clip(lower=0.0)

    pv_unused_steps = pv_steps[pv_steps["pv_unused_kw"] > 1e-3].copy()
    pv_unused_steps = pv_unused_steps.sort_values(["pv_unused_kw", "ts"], ascending=[False, True])

    return pv_unused_steps


# =============================================================================
# 9) Hauptsimulation
# =============================================================================

def simulate_load_profile(
    scenario: dict,
    start_datetime: datetime | None = None,
    record_debug: bool = False,
):
    """
    Diese Funktion führt die eigentliche Lastgangsimulation aus.

    Sie liefert:
      - time_index: Zeitachse (datetime-Liste)
      - load_profile_kw: EV-Ladeleistung über der Zeit
      - all_charging_sessions: Session-Objekte mit Ladeergebnis
      - charging_count_series: Anzahl aktiv ladender Sessions pro Zeitschritt
      - holiday_dates: verwendete Feiertage
      - charging_strategy: verwendete Strategie ("immediate", "market", "generation")
      - strategy_status: "ACTIVE"/"INACTIVE"/"IMMEDIATE"
      - debug_rows: optional detaillierte Zeitschrittinfos
    """

    # ------------------------------------------------------------
    # Fixe Heuristik-Parameter
    # ------------------------------------------------------------
    # EMERGENCY_SLACK_MINUTES definiert, ab wann eine Session als "kritisch" gilt.
    # MARKET_TOP_K_SLOTS ist ein optionales Konzept (hier aktuell nicht genutzt).
    EMERGENCY_SLACK_MINUTES = 60.0
    MARKET_TOP_K_SLOTS = 4

    # ------------------------------------------------------------
    # 1) Zeitindex
    # ------------------------------------------------------------
    # Die Simulation arbeitet auf einer festen Zeitachse (z.B. 15-Minuten-Schritte).
    time_index = create_time_index(scenario, start_datetime)

    # ------------------------------------------------------------
    # 2) Feiertage
    # ------------------------------------------------------------
    # Feiertage werden einmalig bestimmt, um Tagesarten korrekt zu klassifizieren.
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
    # 3) Fahrzeuge
    # ------------------------------------------------------------
    # Die Fahrzeugprofile bestimmen, welche Ladeleistungen bei welchem SoC möglich sind.
    vehicle_csv_path = resolve_path_relative_to_scenario(scenario, scenario["vehicles"]["vehicle_curve_csv"])
    vehicle_profiles = load_vehicle_profiles_from_csv(vehicle_csv_path)

    # ------------------------------------------------------------
    # 4) Strategie-Initialisierung (getrennte CSVs für market & generation)
    # ------------------------------------------------------------
    # Je nach Ladestrategie werden unterschiedliche Signale geladen:
    #   - market: Marktpreise
    #   - generation: PV/Erzeugung + Marktpreise als Fallback
    site_cfg = scenario.get("site", {}) or {}
    charging_strategy = (site_cfg.get("charging_strategy") or "immediate").lower()

    STRATEGY_RESOLUTION_MIN = 15
    strategy_step_hours = STRATEGY_RESOLUTION_MIN / 60.0

    # Interne Container für Signalzeitreihen (maps)
    generation_map: dict[datetime, float] | None = None
    generation_csv_path: str | None = None
    generation_unit: str | None = None

    market_map: dict[datetime, float] | None = None
    market_csv_path: str | None = None
    market_unit: str | None = None

    # -----------------------------
    # A) Market-Strategie: nur Market CSV
    # -----------------------------
    if charging_strategy == "market":
        market_unit = str(site_cfg.get("market_strategy_unit", "") or "").strip()
        if market_unit not in ("€/MWh", "€/kWh"):
            raise ValueError("❌ Abbruch: 'site.market_strategy_unit' muss '€/MWh' oder '€/kWh' sein.")

        market_csv = site_cfg.get("market_strategy_csv", None)
        market_col = site_cfg.get("market_strategy_value_col", None)
        if not market_csv:
            raise ValueError("❌ Abbruch: Für charging_strategy='market' muss 'site.market_strategy_csv' gesetzt sein.")
        if not isinstance(market_col, int) or market_col < 2:
            raise ValueError("❌ Abbruch: 'site.market_strategy_value_col' muss int >= 2 sein (1=Zeitspalte).")

        market_csv_path = resolve_path_relative_to_scenario(scenario, str(market_csv))
        market_map = read_strategy_series_from_csv_first_col_time(
            csv_path=market_csv_path,
            value_col_1_based=int(market_col),
            delimiter=";",
        )

        assert_strategy_csv_covers_simulation(
            strategy_map=market_map,
            time_index=time_index,
            strategy_resolution_min=STRATEGY_RESOLUTION_MIN,
            charging_strategy="market",
            strategy_csv_path=market_csv_path,
        )

    # -----------------------------
    # B) Generation-Strategie: Generation CSV + Market CSV (Fallback)
    # -----------------------------
    elif charging_strategy == "generation":
        generation_unit = str(site_cfg.get("generation_strategy_unit", "") or "").strip()
        if generation_unit not in ("kW", "kWh", "MWh"):
            raise ValueError("❌ Abbruch: 'site.generation_strategy_unit' muss 'kW', 'kWh' oder 'MWh' sein.")

        gen_csv = site_cfg.get("generation_strategy_csv", None)
        gen_col = site_cfg.get("generation_strategy_value_col", None)
        if not gen_csv:
            raise ValueError("❌ Abbruch: Für charging_strategy='generation' muss 'site.generation_strategy_csv' gesetzt sein.")
        if not isinstance(gen_col, int) or gen_col < 2:
            raise ValueError("❌ Abbruch: 'site.generation_strategy_value_col' muss int >= 2 sein (1=Zeitspalte).")

        generation_csv_path = resolve_path_relative_to_scenario(scenario, str(gen_csv))
        generation_map = read_strategy_series_from_csv_first_col_time(
            csv_path=generation_csv_path,
            value_col_1_based=int(gen_col),
            delimiter=";",
        )

        assert_strategy_csv_covers_simulation(
            strategy_map=generation_map,
            time_index=time_index,
            strategy_resolution_min=STRATEGY_RESOLUTION_MIN,
            charging_strategy="generation",
            strategy_csv_path=generation_csv_path,
        )

        # Market-Fallback ist bei diesem Setup verpflichtend
        market_unit = str(site_cfg.get("market_strategy_unit", "") or "").strip()
        if market_unit not in ("€/MWh", "€/kWh"):
            raise ValueError("❌ Abbruch: generation + Market-Fallback: 'site.market_strategy_unit' muss '€/MWh' oder '€/kWh' sein.")

        market_csv = site_cfg.get("market_strategy_csv", None)
        market_col = site_cfg.get("market_strategy_value_col", None)
        if not market_csv:
            raise ValueError("❌ Abbruch: generation + Market-Fallback: 'site.market_strategy_csv' fehlt.")
        if not isinstance(market_col, int) or market_col < 2:
            raise ValueError("❌ Abbruch: generation + Market-Fallback: 'site.market_strategy_value_col' muss int >= 2 sein.")

        market_csv_path = resolve_path_relative_to_scenario(scenario, str(market_csv))
        market_map = read_strategy_series_from_csv_first_col_time(
            csv_path=market_csv_path,
            value_col_1_based=int(market_col),
            delimiter=";",
        )

        assert_strategy_csv_covers_simulation(
            strategy_map=market_map,
            time_index=time_index,
            strategy_resolution_min=STRATEGY_RESOLUTION_MIN,
            charging_strategy="market_fallback_for_generation",
            strategy_csv_path=market_csv_path,
        )

    # -----------------------------
    # C) Immediate: kein Signal erforderlich
    # -----------------------------
    elif charging_strategy == "immediate":
        pass
    else:
        raise ValueError(f"❌ Abbruch: Unbekannte charging_strategy='{charging_strategy}'")

    # ------------------------------------------------------------
    # 5) Parameter & Ergebniscontainer
    # ------------------------------------------------------------
    # Hier werden Simulationseinstellungen aus dem Szenario übernommen und Ergebnisarrays vorbereitet.
    time_resolution_min = scenario["time_resolution_min"]
    time_step_hours = time_resolution_min / 60.0

    n_steps = len(time_index)
    load_profile_kw = np.zeros(n_steps, dtype=float)

    grid_limit_p_avb_kw = float(scenario["site"]["grid_limit_p_avb_kw"])
    rated_power_kw = float(scenario["site"]["rated_power_kw"])
    number_of_chargers = int(scenario["site"]["number_chargers"])
    charger_efficiency = float(scenario["site"]["charger_efficiency"])

    # ------------------------------------------------------------
    # 5b) Basislast (nur relevant für generation)
    # ------------------------------------------------------------
    base_load_series = None
    if charging_strategy == "generation":
        base_load_series = build_base_load_series(
            scenario=scenario,
            timestamps=time_index,
            base_load_resolution_min=STRATEGY_RESOLUTION_MIN,
        )

    def _base_load_kw_at(i: int) -> float:
        """Diese Hilfsfunktion liefert die Basislast am Zeitschritt i (NaN -> 0)."""
        if base_load_series is None:
            return 0.0
        v = float(base_load_series[i])
        return 0.0 if np.isnan(v) else max(0.0, v)

    def _generation_kw_at(ts: datetime) -> float:
        """Diese Hilfsfunktion liefert die normierte Erzeugungsleistung (kW) zum Zeitpunkt ts."""
        if charging_strategy != "generation":
            return 0.0
        if not generation_map or not generation_unit:
            return 0.0
        raw = lookup_signal(generation_map, ts, STRATEGY_RESOLUTION_MIN)
        if raw is None:
            return 0.0
        return max(
            0.0,
            float(
                convert_strategy_value_to_internal(
                    charging_strategy="generation",
                    raw_value=float(raw),
                    strategy_unit=str(generation_unit),
                    step_hours=strategy_step_hours,
                )
            ),
        )

    # ------------------------------------------------------------
    # 5c) PV-Überschuss-Zeitreihe (nur generation)
    # ------------------------------------------------------------
    pv_surplus_series_kw = None
    if charging_strategy == "generation":
        pv_surplus_series_kw = np.zeros(len(time_index), dtype=float)
        for ii, tts in enumerate(time_index):
            pv_kw = _generation_kw_at(tts)
            base_kw = _base_load_kw_at(ii)
            pv_surplus_series_kw[ii] = max(0.0, pv_kw - base_kw)

    # ------------------------------------------------------------
    # 5d) Strategieparameter
    # ------------------------------------------------------------
    # emergency_slack_minutes kann im YAML überschrieben werden.
    strategy_params = site_cfg.get("strategy_params", {}) or {}
    emergency_slack_minutes = float(strategy_params.get("emergency_slack_minutes", 60.0))

    charging_count_series: list[int] = []
    all_charging_sessions: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------
    # 6) Sessions erzeugen
    # ------------------------------------------------------------
    # Die Sessions werden tagesweise erzeugt und anschließend als Event-Liste simuliert.
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
    # 6b) Präferenzlisten aufbauen
    # ------------------------------------------------------------
    # MARKET: nutzt preferred_slot_indices
    # GENERATION: nutzt preferred_market_slot_indices für Grid-Fallback
    time_to_idx = {t: idx for idx, t in enumerate(time_index)} if time_index else {}

    if charging_strategy == "market":
        _build_preferred_slots_for_all_sessions(
            all_sessions=all_charging_sessions,
            time_index=time_index,
            time_to_idx=time_to_idx,
            charging_strategy="market",
            strategy_map=market_map,
            strategy_resolution_min=STRATEGY_RESOLUTION_MIN,
            strategy_unit=str(market_unit),
            step_hours=strategy_step_hours,
        )

    if charging_strategy == "generation" and market_map is not None and market_unit is not None:
        _build_preferred_market_slots_for_generation_fallback(
            all_sessions=all_charging_sessions,
            time_index=time_index,
            time_to_idx=time_to_idx,
            market_map=market_map,
            market_resolution_min=STRATEGY_RESOLUTION_MIN,
            market_unit=str(market_unit),
            step_hours=strategy_step_hours,
        )

    # ------------------------------------------------------------
    # 6c) Ankunfts-Policy / Belegung
    # ------------------------------------------------------------
    # In dieser Implementierung ist die Policy hart kodiert:
    #   - Wenn kein Ladepunkt frei ist: Session wird abgewiesen (drive_off).
    #   - Fahrzeuge bleiben bis zur Abfahrt eingesteckt (kein disconnect_when_full).
    all_charging_sessions.sort(key=lambda s: s["arrival_time"])
    arrival_policy = "drive_off"

    chargers: list[dict[str, Any] | None] = [None] * number_of_chargers
    next_arrival_idx = 0

    plugged_session_ids = set()
    rejected_session_ids = set()

    # ============================================================
    # 7) Zeitschrittweise Simulation
    # ============================================================
    for i, ts in enumerate(time_index):

        # --------------------------------------------------------
        # 7.0) Ladepunkte freigeben (nur bei Abfahrt)
        # --------------------------------------------------------
        for c in range(len(chargers)):
            s = chargers[c]
            if s is None:
                continue
            departed = not (s["arrival_time"] <= ts < s["departure_time"])
            if departed:
                chargers[c] = None

        # --------------------------------------------------------
        # 7.1) Neue Ankünfte verarbeiten (drive_off)
        # --------------------------------------------------------
        while (
            next_arrival_idx < len(all_charging_sessions)
            and all_charging_sessions[next_arrival_idx]["arrival_time"] <= ts
        ):
            s = all_charging_sessions[next_arrival_idx]
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
                plugged_session_ids.add(s["session_id"])
                continue

            # Kein Ladepunkt frei -> Abweisung
            s["_rejected"] = True
            s["_rejection_time"] = ts
            s["_no_charge_reason"] = "no_free_charger_at_arrival"
            rejected_session_ids.add(s["session_id"])
            continue

        # --------------------------------------------------------
        # 7.3) Anwesende Sessions bestimmen
        # --------------------------------------------------------
        plugged_sessions = [s for s in chargers if s is not None]
        present_sessions = [
            s for s in plugged_sessions
            if s["arrival_time"] <= ts < s["departure_time"]
            and float(s.get("energy_required_kwh", 0.0)) > 1e-9
        ]

        for s in present_sessions:
            s["_actual_power_kw"] = 0.0

        if not present_sessions:
            load_profile_kw[i] = 0.0
            charging_count_series.append(0)
            continue

        # --------------------------------------------------------
        # 7.4) Auswahl der ladenden Sessions (strategiespezifisch)
        # --------------------------------------------------------
        charging_sessions: list[dict[str, Any]] = []

        if charging_strategy == "immediate":
            present_sessions.sort(key=lambda s: (s["departure_time"], s["arrival_time"]))
            charging_sessions = present_sessions[:number_of_chargers]

        elif charging_strategy == "generation":
            if pv_surplus_series_kw is None:
                pv_surplus_series_kw = np.zeros(len(time_index), dtype=float)

            _pv_commit_kw, grid_needed_ids = _plan_pv_commitments_for_present_sessions(
                present_sessions=present_sessions,
                i_now=i,
                pv_surplus_series_kw=pv_surplus_series_kw,
                time_index=time_index,
                time_resolution_min=time_resolution_min,
                time_step_hours=time_step_hours,
                charger_efficiency=charger_efficiency,
                rated_power_kw=rated_power_kw,
            )

            pv_budget_kw_now = max(0.0, _generation_kw_at(ts) - _base_load_kw_at(i))

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

            has_market_fallback = (market_map is not None and market_unit is not None)

            emergency: list[dict[str, Any]] = []
            market_slot: list[dict[str, Any]] = []
            grid_other: list[dict[str, Any]] = []

            for s in grid_needed_candidates:
                if float(s.get("_slack_minutes", 1e9)) <= emergency_slack_minutes:
                    emergency.append(s)
                    continue

                if has_market_fallback:
                    pref = s.get("preferred_market_slot_indices", [])
                    ptr = int(s.get("preferred_market_ptr", 0))

                    while ptr < len(pref) and pref[ptr] < i:
                        ptr += 1
                    s["preferred_market_ptr"] = ptr

                    if pref and ptr < len(pref) and pref[ptr] == i:
                        market_slot.append(s)
                    else:
                        grid_other.append(s)
                else:
                    grid_other.append(s)

            emergency.sort(key=lambda s: (s.get("_slack_minutes", 1e9), s["departure_time"], s["arrival_time"]))
            market_slot.sort(key=lambda s: (s.get("_slack_minutes", 1e9), s["departure_time"], s["arrival_time"]))
            grid_other.sort(key=lambda s: (s.get("_slack_minutes", 1e9), s["departure_time"], s["arrival_time"]))
            pv_only_candidates.sort(key=lambda s: (s.get("_slack_minutes", 1e9), s["departure_time"], s["arrival_time"]))

            charging_sessions = []
            charging_sessions.extend(emergency[:number_of_chargers])

            remaining = number_of_chargers - len(charging_sessions)
            if remaining > 0 and has_market_fallback:
                charging_sessions.extend(market_slot[:remaining])

            remaining = number_of_chargers - len(charging_sessions)
            if remaining > 0 and pv_budget_kw_now > 1e-9:
                charging_sessions.extend(pv_only_candidates[:remaining])

            remaining = number_of_chargers - len(charging_sessions)
            if remaining > 0:
                charging_sessions.extend(grid_other[:remaining])

        else:
            # market
            emergency: list[dict[str, Any]] = []
            slot_candidates: list[dict[str, Any]] = []

            for s in present_sessions:
                slack_m = _slack_minutes_for_session(
                    s=s,
                    ts=ts,
                    rated_power_kw=rated_power_kw,
                    charger_efficiency=charger_efficiency,
                )
                s["_slack_minutes"] = float(slack_m)

                if slack_m <= emergency_slack_minutes:
                    emergency.append(s)
                    continue

                pref = s.get("preferred_slot_indices", [])
                ptr = int(s.get("preferred_ptr", 0))

                while ptr < len(pref) and pref[ptr] < i:
                    ptr += 1
                s["preferred_ptr"] = ptr

                if pref and ptr < len(pref) and pref[ptr] == i:
                    slot_candidates.append(s)

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
                candidates.sort(key=lambda s: (s.get("_slack_minutes", 1e9), s["departure_time"], s["arrival_time"]))
                charging_sessions = candidates[:number_of_chargers]
            else:
                charging_sessions = []

        charging_count_series.append(len(charging_sessions))

        if not charging_sessions:
            load_profile_kw[i] = 0.0
            continue

        # ---------------------------------------------------------------
        # 7.5) Leistungsverteilung / Energieupdate
        # ---------------------------------------------------------------
        total_power_kw = 0.0

        if charging_strategy == "generation":
            charger_budget_kw = len(charging_sessions) * rated_power_kw
            grid_budget_kw = max(0.0, float(grid_limit_p_avb_kw))
            pv_budget_kw = max(0.0, _generation_kw_at(ts) - _base_load_kw_at(i))

            grid_needed_sessions = [s for s in charging_sessions if bool(s.get("_grid_needed", False))]
            pv_only_sessions = [s for s in charging_sessions if not bool(s.get("_grid_needed", False))]

            # Grid-needed: PV + Grid (gemeinsamer Budgettopf)
            grid_needed_used_kw = 0.0
            if grid_needed_sessions and charger_budget_kw > 1e-9:
                usable_budget_kw = min(charger_budget_kw, pv_budget_kw + grid_budget_kw)
                per_session_kw = usable_budget_kw / len(grid_needed_sessions)

                for s in grid_needed_sessions:
                    vehicle_power_limit_kw = vehicle_power_at_soc(s)

                    allowed_power_kw = min(
                        per_session_kw,
                        rated_power_kw,
                        vehicle_power_limit_kw,
                        float(s.get("max_charging_power_kw", rated_power_kw)),
                    )

                    energy_delivered = 0.0
                    actual_power_kw = 0.0

                    if allowed_power_kw > 1e-9:
                        possible_energy_kwh = allowed_power_kw * time_step_hours * charger_efficiency
                        energy_needed = float(s.get("energy_required_kwh", 0.0))

                        if possible_energy_kwh >= energy_needed:
                            energy_delivered = energy_needed
                            s["energy_required_kwh"] = 0.0
                            actual_power_kw = (
                                energy_delivered / (time_step_hours * charger_efficiency)
                                if energy_delivered > 0.0 else 0.0
                            )
                            if "finished_charging_time" not in s:
                                s["finished_charging_time"] = ts
                        else:
                            energy_delivered = possible_energy_kwh
                            s["energy_required_kwh"] = energy_needed - possible_energy_kwh
                            actual_power_kw = allowed_power_kw

                    s["delivered_energy_kwh"] = float(s.get("delivered_energy_kwh", 0.0)) + float(energy_delivered)
                    s["_actual_power_kw"] = float(actual_power_kw)

                    grid_needed_used_kw += actual_power_kw
                    total_power_kw += actual_power_kw

                pv_used_by_grid_needed_kw = min(pv_budget_kw, grid_needed_used_kw)
                pv_budget_kw = max(0.0, pv_budget_kw - pv_used_by_grid_needed_kw)
                charger_budget_kw = max(0.0, charger_budget_kw - grid_needed_used_kw)

            # PV-only: nur PV-Rest (work-conserving)
            if pv_only_sessions and pv_budget_kw > 1e-9 and charger_budget_kw > 1e-9:
                pv_only_budget_kw = min(pv_budget_kw, charger_budget_kw)

                remaining_budget_kw = float(pv_only_budget_kw)
                pv_only_used_kw = 0.0

                active = [s for s in pv_only_sessions if float(s.get("energy_required_kwh", 0.0)) > 1e-9]
                iter_guard = 0

                while remaining_budget_kw > 1e-9 and active and iter_guard < 20:
                    iter_guard += 1
                    per_kw = remaining_budget_kw / len(active)

                    used_round_kw = 0.0
                    next_active: list[dict[str, Any]] = []

                    for s in active:
                        vehicle_power_limit_kw = vehicle_power_at_soc(s)

                        allowed_power_kw = min(
                            per_kw,
                            rated_power_kw,
                            vehicle_power_limit_kw,
                            float(s.get("max_charging_power_kw", rated_power_kw)),
                        )

                        if allowed_power_kw <= 1e-9:
                            s["_actual_power_kw"] = 0.0
                            next_active.append(s)
                            continue

                        possible_energy_kwh = allowed_power_kw * time_step_hours * charger_efficiency
                        energy_needed = float(s.get("energy_required_kwh", 0.0))

                        energy_delivered = 0.0
                        actual_power_kw = 0.0

                        if possible_energy_kwh >= energy_needed:
                            energy_delivered = energy_needed
                            s["energy_required_kwh"] = 0.0
                            actual_power_kw = (
                                energy_delivered / (time_step_hours * charger_efficiency)
                                if energy_delivered > 0.0 else 0.0
                            )
                            if "finished_charging_time" not in s:
                                s["finished_charging_time"] = ts
                        else:
                            energy_delivered = possible_energy_kwh
                            s["energy_required_kwh"] = energy_needed - possible_energy_kwh
                            actual_power_kw = allowed_power_kw

                        s["delivered_energy_kwh"] = float(s.get("delivered_energy_kwh", 0.0)) + float(energy_delivered)
                        s["_actual_power_kw"] = float(actual_power_kw)

                        used_round_kw += actual_power_kw
                        pv_only_used_kw += actual_power_kw
                        total_power_kw += actual_power_kw

                        if float(s.get("energy_required_kwh", 0.0)) > 1e-9:
                            next_active.append(s)

                    if used_round_kw <= 1e-9:
                        break

                    used_round_kw = min(used_round_kw, remaining_budget_kw)
                    remaining_budget_kw = max(0.0, remaining_budget_kw - used_round_kw)
                    active = next_active

                pv_budget_kw = max(0.0, pv_budget_kw - pv_only_used_kw)
                charger_budget_kw = max(0.0, charger_budget_kw - pv_only_used_kw)

            load_profile_kw[i] = float(total_power_kw)

        else:
            # immediate/market: klassische Fair-Share-Logik unter Netz- und Ladepunktlimits
            max_site_power_kw = min(grid_limit_p_avb_kw, len(charging_sessions) * rated_power_kw)
            available_power_per_session_kw = max_site_power_kw / len(charging_sessions)

            for s in charging_sessions:
                vehicle_power_limit_kw = vehicle_power_at_soc(s)

                allowed_power_kw = min(
                    available_power_per_session_kw,
                    rated_power_kw,
                    vehicle_power_limit_kw,
                    float(s.get("max_charging_power_kw", rated_power_kw)),
                )

                energy_delivered = 0.0
                actual_power_kw = 0.0

                if allowed_power_kw > 1e-9:
                    possible_energy_kwh = allowed_power_kw * time_step_hours * charger_efficiency
                    energy_needed = float(s.get("energy_required_kwh", 0.0))

                    if possible_energy_kwh >= energy_needed:
                        energy_delivered = energy_needed
                        s["energy_required_kwh"] = 0.0
                        actual_power_kw = (
                            energy_delivered / (time_step_hours * charger_efficiency)
                            if energy_delivered > 0.0 else 0.0
                        )
                        if "finished_charging_time" not in s:
                            s["finished_charging_time"] = ts
                    else:
                        energy_delivered = possible_energy_kwh
                        s["energy_required_kwh"] = energy_needed - possible_energy_kwh
                        actual_power_kw = allowed_power_kw

                s["delivered_energy_kwh"] = float(s.get("delivered_energy_kwh", 0.0)) + float(energy_delivered)
                s["_actual_power_kw"] = float(actual_power_kw)

                total_power_kw += actual_power_kw

            load_profile_kw[i] = float(total_power_kw)

        # ---------------------------------------------------------------
        # 7.6) Debug loggen
        # ---------------------------------------------------------------
        if record_debug:
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

            if charging_strategy == "generation":
                site_grid_import_kw = max(0.0, total_power_kw - pv_surplus_kw)
            else:
                site_grid_import_kw = max(0.0, total_power_kw)

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

    # ============================================================
    # 8) Strategie-Status bestimmen
    # ============================================================
    if charging_strategy == "immediate":
        strategy_status = "IMMEDIATE"
    elif charging_strategy == "market":
        strategy_status = "ACTIVE" if market_map else "INACTIVE"
    elif charging_strategy == "generation":
        strategy_status = "ACTIVE" if generation_map else "INACTIVE"
    else:
        raise ValueError(f"❌ Abbruch: Unbekannte charging_strategy='{charging_strategy}'")

    # ============================================================
    # 9) Rückgabe
    # ============================================================
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
