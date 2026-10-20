"""Explicit location/time lookup for the existing regional NBM forecast."""
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import logging
import math
import re
from threading import Lock
import time
from zoneinfo import ZoneInfo

import numpy as np
import requests

PACIFIC = ZoneInfo("America/Los_Angeles")
BOUNDS = (-116.3, 32.6, -114.7, 34.3)
SELECTOR = r":(?:WETGLBT|var discipline=0 center=7 local_table=1 parmcat=0 parm=206):surface:"
_geocode_lock = Lock()
_forecast_lock = Lock()
_last_geocode = 0.0
logger = logging.getLogger(__name__)
MONTHS = {name: number for number, name in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
MONTH_PATTERN = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
NAMED_DATE = re.compile(
    rf"\b(?P<month>{MONTH_PATTERN})\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b"
    r"(?:,?\s+(?P<year>\d{4})\b)?", re.I)


class LookupUnavailable(ValueError):
    """A clarification or unavailable-data message safe to show to the user."""


def requested_lookup(message):
    return bool(re.search(r"\b(wbgt|heat|hot|temperature)\b", message, re.I) and
                (re.search(r"\b(today|tomorrow|now)\b|\d{4}-\d{2}-\d{2}|\d\s*(am|pm)\b|\bat\s+\d{1,2}:\d{2}\b", message, re.I)
                 or NAMED_DATE.search(message))
                and not re.search(r"\b(SST|ONI|MEI|ENSO)\b|ni[ñn]o", message, re.I))


def parse_request(message, now):
    """Parse relative, ISO, or named-month dates; omitted years use Pacific's current year."""
    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2}|today|tomorrow|now)\b", message, re.I)
    named_date = None
    if not date_match:
        named_date = NAMED_DATE.search(message)
        date_match = named_date
    if not date_match:
        raise LookupUnavailable("Please specify the location, date (YYYY-MM-DD, today, or tomorrow), and time, for example: WBGT in Chiriaco Summit tomorrow at 2 PM.")
    token = date_match[1].lower()
    local_now = now.astimezone(PACIFIC)
    try:
        if named_date:
            day = datetime(
                int(named_date["year"] or local_now.year),
                MONTHS[named_date["month"][:3].lower()],
                int(named_date["day"]),
            ).date()
        else:
            day = (local_now.date() + timedelta(days=token == "tomorrow")
               if token in {"today", "tomorrow", "now"} else
               datetime.strptime(token, "%Y-%m-%d").date())
    except ValueError as exc:
        raise LookupUnavailable("That date is invalid. Please use YYYY-MM-DD.") from exc
    clock = re.search(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b|\bat\s+(\d{1,2}):(\d{2})\b", message, re.I)
    if clock:
        if clock[1]:
            hour, minute = int(clock[1]), int(clock[2] or 0)
            if not 1 <= hour <= 12:
                raise LookupUnavailable("Please use a valid hour, such as 2 PM or 14:00.")
            hour = hour % 12 + (12 if clock[3].lower() == "pm" else 0)
        else:
            hour, minute = int(clock[4]), int(clock[5])
        if hour > 23 or minute != 0:
            raise LookupUnavailable("WBGT forecasts are hourly. Please request a whole hour, such as 2 PM or 14:00.")
    elif token == "now":
        hour, minute = local_now.hour, 0
    else:
        raise LookupUnavailable("What time should I use? Please specify a whole hour, such as 2 PM. Times are Pacific Time.")
    if re.search(r"\b(UTC|GMT|EST|EDT|CST|CDT|MST|MDT)\b", message, re.I):
        raise LookupUnavailable("Please give the time in Pacific Time for this regional forecast.")
    naive = datetime.combine(day, datetime.min.time()).replace(hour=hour, minute=minute)
    target = naive.replace(tzinfo=PACIFIC)
    if target.astimezone(timezone.utc).astimezone(PACIFIC).replace(tzinfo=None) != naive:
        raise LookupUnavailable("That local time does not exist because of daylight saving time. Please choose another hour.")
    if target.utcoffset() != target.replace(fold=1).utcoffset():
        raise LookupUnavailable("That hour occurs twice at the daylight saving transition. Please choose another hour.")
    coordinates = re.search(r"(?<![\d.])(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)", message)
    if coordinates:
        location = (float(coordinates[1]), float(coordinates[2]))
    else:
        # Extract a place after "in/at/for", ending before the date or clock.
        prefix = message[:date_match.start()]
        if clock and clock.start() < date_match.start():
            prefix = message[:clock.start()]
        prefix = re.sub(r"\s+on\s*$", "", prefix, flags=re.I)
        place = re.search(r"\b(?:in|at|for)\s+(.+)$", prefix, re.I)
        if not place:
            raise LookupUnavailable("Please name a location, for example: WBGT in Chiriaco Summit tomorrow at 2 PM. You can also supply latitude, longitude.")
        location = place[1].strip(" ,?")
        if len(location) > 180:
            raise LookupUnavailable("Please provide a shorter city or place name.")
    return location, target


@lru_cache(maxsize=128)
def geocode(place):
    global _last_geocode
    with _geocode_lock:
        time.sleep(max(0, 1.1 - (time.monotonic() - _last_geocode)))
        _last_geocode = time.monotonic()
        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": place, "format": "json", "limit": 3,
                        "viewbox": "-116.3,34.3,-114.7,32.6", "bounded": 1},
                headers={"User-Agent": "iHeat-Sol-WBGT-Lookup/1.0"},
                timeout=15,
            )
            response.raise_for_status()
            results = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise LookupUnavailable("The location service is unavailable. Try again or supply latitude, longitude.") from exc
    if not results:
        raise LookupUnavailable("I couldn't find that location within iHeat's Imperial/Coachella Valley coverage. Please use a more specific place or coordinates.")
    if len(results) > 1:
        raise LookupUnavailable("That place has multiple matches. Please supply latitude, longitude to identify the exact point.")
    try:
        item = results[0]
        return float(item["lat"]), float(item["lon"]), item["display_name"]
    except (KeyError, TypeError, ValueError) as exc:
        raise LookupUnavailable("The location service returned an invalid result. Please supply latitude, longitude.") from exc


@lru_cache(maxsize=32)
def forecast_point(run_iso, hour, latitude, longitude):
    from herbie import Herbie
    run = datetime.fromisoformat(run_iso).replace(tzinfo=None)
    with _forecast_lock:
        h = Herbie(run, model="nbm", product="co", fxx=hour)
        ds = h.xarray(SELECTOR, backend_kwargs={
            "read_keys": ["centre", "discipline", "parameterCategory", "parameterNumber"],
        })
        try:
            variables = [name for name, value in ds.data_vars.items()
                         if value.ndim >= 2]
            if len(variables) != 1:
                raise ValueError("Expected a single WBGT grid")
            da = ds[variables[0]].squeeze()
            lat = np.asarray(ds["latitude"])
            lon = (np.asarray(ds["longitude"]) + 180) % 360 - 180
            distance = (lat - latitude) ** 2 + ((lon - longitude) * math.cos(math.radians(latitude))) ** 2
            index = np.unravel_index(np.nanargmin(distance), distance.shape)
            value = float(da.values[index])
            units = str(da.attrs.get("units", "")).lower()
            # Older ecCodes tables do not recognize NCEP's local WBGT parameter.
            # NOAA table 4.2-0-0 specifies Kelvin for NCEP parameter 206:
            # https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-0-0.shtml
            # Only recover missing units after verifying the exact parameter.
            if units in {"", "unknown"} and (
                da.attrs.get("GRIB_centre") in {"kwbc", 7, "7"}
                and da.attrs.get("GRIB_discipline") == 0
                and da.attrs.get("GRIB_parameterCategory") == 0
                and da.attrs.get("GRIB_parameterNumber") == 206
            ):
                units = "k"
            if not math.isfinite(value) or value > 1e10:
                raise ValueError("Missing WBGT grid value")
            if units in {"k", "kelvin"}:
                value = (value - 273.15) * 9 / 5 + 32
            elif units in {"c", "degc", "celsius", "degree_celsius"}:
                value = value * 9 / 5 + 32
            elif units not in {"f", "degf", "fahrenheit"}:
                raise ValueError(f"Unsupported WBGT units: {units}")
            return value, float(lat[index]), float(lon[index])
        finally:
            ds.close()


def lookup(message, now=None):
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    place, target = parse_request(message, now)
    target_utc = target.astimezone(timezone.utc)
    latest = (now - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
    lead = int((target_utc - latest).total_seconds() / 3600)
    if not 1 <= lead <= 36:
        raise LookupUnavailable("That time is outside the current iHeat forecast window (1–36 hours from the latest likely available model run). Historical WBGT lookup is not available here.")
    if isinstance(place, tuple):
        lat, lon = place
        name = f"{lat}, {lon}"
    else:
        lat, lon, name = geocode(place)
    west, south, east, north = BOUNDS
    if not south <= lat <= north or not west <= lon <= east:
        raise LookupUnavailable("That point is outside iHeat's current Imperial/Coachella Valley forecast coverage.")
    for lag in range(4):
        run = latest - timedelta(hours=lag)
        hour = lead + lag
        if hour > 36:
            break
        try:
            value, grid_lat, grid_lon = forecast_point(run.isoformat(), hour, lat, lon)
            return {"wbgt_f": round(value, 1), "location": name,
                    "latitude": lat, "longitude": lon,
                    "grid_latitude": grid_lat, "grid_longitude": grid_lon,
                    "forecast_valid_at": target.isoformat(),
                    "forecast_valid_utc": target_utc.isoformat(),
                    "model_run_utc": run.isoformat(), "forecast_hour": hour,
                    "timezone": "America/Los_Angeles",
                    "source": "NOAA NBM forecast (nearest grid point, not a station observation)"}
        except Exception:
            logger.exception("WBGT point lookup failed for run %s lead %s", run, hour)
    raise LookupUnavailable("The requested WBGT forecast could not be downloaded or decoded. Please try again later. I cannot provide a measured value for this request.")
