from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .models import RoadContext

COORDINATE_RE = re.compile(
    r"(?:N(?:orth)?\s*[:=]?\s*)?(?<![\d.])([+-]?\d{1,2}(?:\s*\.\s*\d+)?)"
    r"\s*[,; ]+\s*(?:E(?:ast)?\s*[:=]?\s*)?(?<![\d.])([+-]?\d{1,3}(?:\s*\.\s*\d+)?)",
    re.IGNORECASE,
)
ROAD_CONTEXT_CACHE_VERSION = 4


def parse_coordinate_text(text: str) -> tuple[float, float] | None:
    match = COORDINATE_RE.search(str(text))
    if not match:
        return None
    try:
        latitude = float(re.sub(r"\s+", "", match.group(1)))
        longitude = float(re.sub(r"\s+", "", match.group(2)))
    except ValueError:
        return None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None
    return latitude, longitude


def sample_trajectory(points: list[tuple[float, float]], maximum: int = 7) -> list[tuple[float, float]]:
    if len(points) <= maximum:
        return list(dict.fromkeys(points))
    indexes = {round(index * (len(points) - 1) / (maximum - 1)) for index in range(maximum)}
    return list(dict.fromkeys(points[index] for index in sorted(indexes)))


class RoadTypeResolver:
    """Resolve OSM hints without converting API failure into a fake highway class."""

    highway_map = {
        "motorway": "Expressway",
        "motorway_link": "Expressway",
        "trunk": "National Highway",
        "trunk_link": "National Highway",
        "primary": "State Highway",
        "primary_link": "State Highway",
        "secondary": "Major District Road",
        "tertiary": "Other District Road",
        "residential": "Local Street",
        "living_street": "Local Street",
        "unclassified": "Rural Road",
    }
    ignored = {"footway", "path", "service", "steps", "pedestrian", "track", "cycleway"}

    def __init__(
        self,
        overpass_url: str = "https://overpass-api.de/api/interpreter",
        elevation_url: str = "https://api.open-meteo.com/v1/elevation",
        user_agent: str = "RoadSafetyAuditor/2.0 (offline-RAG research project)",
        timeout_s: int = 18,
        cache_dir: Path | None = None,
    ):
        self.overpass_url = overpass_url
        self.elevation_url = elevation_url
        self.user_agent = user_agent
        self.timeout_s = timeout_s
        self.cache_dir = cache_dir

    def resolve(self, trajectory: list[tuple[float, float]]) -> RoadContext:
        if not trajectory:
            return RoadContext(notes=["No valid trajectory coordinates; road class is unknown."])

        sampled = sample_trajectory(trajectory, maximum=12)
        cached = self._read_cache(sampled)
        if cached is not None:
            cached.notes.append("Automatic GPS context was loaded from the local lookup cache.")
            return cached

        observations: list[dict[str, Any]] = []
        errors: list[str] = []
        try:
            observations = self._features_near(sampled)
        except Exception as exc:
            errors.append(str(exc))

        road_observations = [
            tags
            for tags in observations
            if tags.get("highway") and tags.get("highway") not in self.ignored
        ]
        if not road_observations:
            note = "No usable OSM road ways were returned."
            if errors:
                note += f" Resolver errors: {errors[0]}"
            return RoadContext(
                road_class_source="OpenStreetMap Overpass (no match)",
                road_class_confidence=0.0,
                notes=[note],
            )

        classified: list[tuple[str, float, dict[str, str]]] = []
        for tags in road_observations:
            road_class, confidence = self._classify(tags)
            if road_class:
                classified.append((road_class, confidence, tags))
        if not classified:
            return RoadContext(
                road_class_source="OpenStreetMap Overpass (unmapped highway tag)",
                road_class_confidence=0.0,
                notes=["OSM returned road ways, but none mapped to an IRC functional class."],
            )

        route_match = self._select_route_class(classified, sampled)
        if route_match is not None:
            selected_class, best, match_quality = route_match
            selected_observations = [
                item for item in classified if item[0] == selected_class
            ]
            representative_tags = best[2]
            confidence = min(0.95, best[1] * (0.55 + 0.45 * match_quality))
        else:
            # Geometry is absent in some cached/test payloads. Preserve an
            # explicitly labelled classification-only fallback rather than
            # pretending it was trajectory map matching.
            vote: Counter[str] = Counter()
            for road_class, item_confidence, _tags in classified:
                vote[road_class] += item_confidence
            selected_class, _ = vote.most_common(1)[0]
            selected_observations = [
                item for item in classified if item[0] == selected_class
            ]
            best = max(selected_observations, key=lambda item: item[1])
            representative_tags = best[2]
            agreement = len(selected_observations) / len(classified)
            confidence = min(0.90, best[1] * (0.60 + 0.30 * agreement))
            match_quality = None

        refs = [tags.get("ref") for _, _, tags in selected_observations if tags.get("ref")]
        setting = self._infer_setting(representative_tags, observations)
        (
            carriageway,
            osm_way_lanes,
            lanes_per_carriageway,
            opposite_carriageway_lanes,
            osm_total_lanes,
            carriageway_count,
            total_road_lanes,
            lane_count_source,
            lane_note,
        ) = self._infer_lane_context(selected_observations, representative_tags)
        speeds = [
            self._parse_speed(tags.get("maxspeed"))
            for _, _, tags in selected_observations
        ]
        speed_values = [speed for speed in speeds if speed is not None]
        # A route may cross more than one posted-speed zone. Until thresholds
        # are resolved per individual OSM segment, use the highest observed
        # posted speed as the conservative route-level proxy and retain the
        # complete set for traceability.
        posted_speed = max(speed_values) if speed_values else None
        distinct_speeds = sorted(set(speed_values))
        terrain, terrain_note = self._infer_terrain(sampled)
        notes = []
        if match_quality is not None:
            notes.append(
                "OSM road class was selected using trajectory-to-way distance, route coverage, "
                f"and heading agreement (map-match quality {match_quality:.2f}; heuristic, not a probability)."
            )
        else:
            notes.append(
                "OSM way geometry was unavailable; road class used a lower-confidence tag-vote fallback."
            )
        if best[1] <= 0.62:
            notes.append(
                "Road class is inferred from an OSM highway tag rather than an authoritative NH/SH/MDR reference."
            )
        if errors:
            notes.append(f"Automatic map lookup reported: {errors[0]}")
        if posted_speed is not None:
            notes.append(
                "OSM maxspeed is retained as a posted-speed proxy for radius assessment; "
                "it is not treated as verified highway design speed."
            )
            if len(distinct_speeds) > 1:
                notes.append(
                    "Multiple OSM posted speeds were observed along/near the sampled route "
                    f"({', '.join(f'{value:g}' for value in distinct_speeds)} km/h); "
                    f"the highest value ({posted_speed:g} km/h) is used as the conservative route-level proxy."
                )
        else:
            notes.append(
                "OSM supplied no numeric maxspeed; both posted-speed proxy and verified design speed remain unknown."
            )
        if lane_note:
            notes.append(lane_note)
        if terrain_note:
            notes.append(terrain_note)
        result = RoadContext(
            road_class=selected_class,
            road_class_source="OpenStreetMap Overpass",
            road_class_confidence=round(confidence, 3),
            highway_ref=Counter(refs).most_common(1)[0][0] if refs else None,
            setting=setting,
            terrain=terrain,
            carriageway=carriageway,
            osm_way_lanes=osm_way_lanes,
            lanes_per_carriageway=lanes_per_carriageway,
            opposite_carriageway_lanes=opposite_carriageway_lanes,
            osm_total_lanes=osm_total_lanes,
            carriageway_count=carriageway_count,
            total_road_lanes=total_road_lanes,
            lane_count_source=lane_count_source,
            posted_speed_kmph=posted_speed,
            posted_speed_source="OpenStreetMap maxspeed tag" if posted_speed is not None else None,
            osm_maxspeed_values_kmph=distinct_speeds,
            osm_highway=representative_tags.get("highway"),
            osm_tags={
                str(key): str(value)
                for key, value in representative_tags.items()
                if not str(key).startswith("_")
            },
            notes=notes,
        )
        self._write_cache(sampled, result)
        return result

    def _select_route_class(
        self,
        classified: list[tuple[str, float, dict[str, Any]]],
        trajectory: list[tuple[float, float]],
    ) -> tuple[str, tuple[str, float, dict[str, Any]], float] | None:
        """Choose the road class whose ways best cover the sampled trajectory."""

        grouped: dict[str, list[tuple[str, float, dict[str, Any]]]] = {}
        for item in classified:
            if self._geometry_points(item[2]):
                grouped.setdefault(item[0], []).append(item)
        if not grouped or not trajectory:
            return None

        route_bearing = (
            _bearing(trajectory[0], trajectory[-1])
            if len(trajectory) >= 2
            else None
        )
        scored: list[
            tuple[float, str, tuple[str, float, dict[str, Any]], float]
        ] = []
        for road_class, items in grouped.items():
            distances = [
                min(
                    _distance_to_polyline_m(point, self._geometry_points(item[2]))
                    for item in items
                )
                for point in trajectory
            ]
            ordered = sorted(distances)
            median_distance = ordered[len(ordered) // 2]
            coverage = sum(distance <= 35.0 for distance in distances) / len(distances)
            proximity = max(0.0, 1.0 - median_distance / 60.0)
            alignments: list[float] = []
            if route_bearing is not None:
                for item in items:
                    way_bearing = self._travel_bearing(item[2])
                    if way_bearing is None:
                        continue
                    difference = abs(
                        ((way_bearing - route_bearing + 180.0) % 360.0) - 180.0
                    )
                    undirected = min(difference, abs(180.0 - difference))
                    alignments.append(max(0.0, 1.0 - undirected / 90.0))
            alignment = max(alignments, default=0.5)
            match_quality = 0.65 * coverage + 0.25 * proximity + 0.10 * alignment
            representative = min(
                items,
                key=lambda item: statistics_median(
                    [
                        _distance_to_polyline_m(
                            point, self._geometry_points(item[2])
                        )
                        for point in trajectory
                    ]
                ),
            )
            prior = max(item[1] for item in items)
            scored.append(
                (match_quality * (0.85 + 0.15 * prior), road_class, representative, match_quality)
            )
        if not scored:
            return None
        _score, road_class, representative, quality = max(
            scored, key=lambda item: item[0]
        )
        return road_class, representative, quality

    def _features_near(
        self, points: list[tuple[float, float]]
    ) -> list[dict[str, Any]]:
        coordinates = ",".join(
            f"{latitude:.7f},{longitude:.7f}" for latitude, longitude in points
        )
        query = (
            f"[out:json][timeout:{self.timeout_s}];"
            "("
            f"way(around:25,{coordinates})[highway];"
            f'nwr(around:140,{coordinates})[landuse~"^(residential|commercial|retail|industrial)$"];'
            f'nwr(around:140,{coordinates})[place~"^(city|town|suburb|neighbourhood|village)$"];'
            ");out tags geom;"
        )
        url = f"{self.overpass_url}?{urllib.parse.urlencode({'data': query})}"
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=self.timeout_s + 2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        features: list[dict[str, Any]] = []
        for element in payload.get("elements", []):
            if not element.get("tags"):
                continue
            feature: dict[str, Any] = {
                str(key): str(value) for key, value in element.get("tags", {}).items()
            }
            feature["_osm_id"] = str(element.get("id", ""))
            feature["_osm_type"] = str(element.get("type", ""))
            if isinstance(element.get("geometry"), list):
                feature["_geometry"] = element["geometry"]
            features.append(feature)
        return features

    def _classify(self, tags: dict[str, Any]) -> tuple[str | None, float]:
        reference = tags.get("ref", "").upper().replace(" ", "")
        if re.search(r"(?:^|;)NE?[-]?\d+", reference):
            return "Expressway", 0.92
        if re.search(r"(?:^|;)NH[-]?\d+", reference):
            return "National Highway", 0.92
        if re.search(r"(?:^|;)SH[-]?\d+", reference):
            return "State Highway", 0.90
        if re.search(r"(?:^|;)MDR[-]?\d+", reference):
            return "Major District Road", 0.88
        if re.search(r"(?:^|;)ODR[-]?\d+", reference):
            return "Other District Road", 0.88
        mapped = self.highway_map.get(tags.get("highway", ""))
        return (mapped, 0.60) if mapped else (None, 0.0)

    @staticmethod
    def _infer_setting(
        tags: dict[str, Any], nearby_features: list[dict[str, Any]]
    ) -> str:
        highway = tags.get("highway")
        if highway in {"residential", "living_street"}:
            return "urban"
        if tags.get("lit") == "yes" or "sidewalk" in tags:
            return "urban"
        if any(
            feature.get("landuse") in {"residential", "commercial", "retail", "industrial"}
            or feature.get("place") in {"city", "town", "suburb", "neighbourhood"}
            for feature in nearby_features
        ):
            return "urban"
        if highway in {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified"}:
            return "rural"
        return "unknown"

    def _infer_lane_context(
        self,
        observations: list[tuple[str, float, dict[str, Any]]],
        representative: dict[str, Any],
    ) -> tuple[
        str,
        int | None,
        int | None,
        int | None,
        int | None,
        int | None,
        int | None,
        str | None,
        str,
    ]:
        """Interpret OSM lane tags without silently doubling or undercounting.

        A bidirectional way's lanes=* is the road total. A one-way way's value
        is per carriageway; a road total is produced only after an opposite,
        geometrically parallel OSM way is confirmed.
        """

        way_lanes = self._parse_lane_count(representative.get("lanes"))
        is_oneway = self._is_oneway(representative)
        explicitly_divided = (
            str(representative.get("dual_carriageway", "")).casefold() == "yes"
            or representative.get("divider") not in (None, "", "no")
        )

        if is_oneway:
            pair = self._find_opposite_carriageway(
                representative, [item[2] for item in observations]
            )
            if pair is not None:
                pair_lanes = self._parse_lane_count(pair.get("lanes"))
                total = way_lanes + pair_lanes if way_lanes and pair_lanes else None
                per_carriageway = (
                    way_lanes if way_lanes and pair_lanes == way_lanes else None
                )
                return (
                    "divided",
                    way_lanes,
                    per_carriageway,
                    pair_lanes,
                    total,
                    2,
                    total if per_carriageway is not None else None,
                    "OSM paired one-way carriageways",
                    (
                        "The raw and base lane totals agree after matching a nearby opposite-direction OSM carriageway."
                        if per_carriageway is not None
                        else "The paired OSM carriageways have asymmetric lane counts. Their raw total is recorded separately, while the base through-lane total remains unverified."
                    ),
                )
            carriageway = "divided" if explicitly_divided else "one_way"
            return (
                carriageway,
                way_lanes,
                way_lanes,
                None,
                None,
                None,
                None,
                "OSM one-way way (unpaired)",
                "OSM lanes describes only the observed one-way carriageway; no opposite carriageway was verified, so total road lanes remain unknown.",
            )

        if way_lanes is not None:
            carriageway = "divided" if explicitly_divided else "undivided"
            return (
                carriageway,
                way_lanes,
                None,
                None,
                way_lanes,
                1,
                way_lanes,
                "OSM bidirectional way",
                "OSM lanes was treated as the road total because the selected way is not tagged one-way.",
            )
        return (
            "divided" if explicitly_divided else "unknown",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "OSM supplied no usable lane count.",
        )

    @staticmethod
    def _parse_lane_count(value: object) -> int | None:
        text = str(value or "").strip()
        if not text.isdigit():
            return None
        lanes = int(text)
        return lanes if 1 <= lanes <= 30 else None

    @staticmethod
    def _is_oneway(tags: dict[str, Any]) -> bool:
        return str(tags.get("oneway", "")).casefold() in {"yes", "1", "true", "-1"}

    def _find_opposite_carriageway(
        self, representative: dict[str, Any], candidates: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        representative_id = str(representative.get("_osm_id", ""))
        representative_bearing = self._travel_bearing(representative)
        if not representative_id or representative_bearing is None:
            return None
        compatible: list[tuple[float, dict[str, Any]]] = []
        for candidate in candidates:
            if str(candidate.get("_osm_id", "")) in {"", representative_id}:
                continue
            if not self._is_oneway(candidate) or not self._same_road(representative, candidate):
                continue
            candidate_bearing = self._travel_bearing(candidate)
            if candidate_bearing is None:
                continue
            direction_difference = abs(
                ((candidate_bearing - representative_bearing + 180.0) % 360.0) - 180.0
            )
            if direction_difference < 120.0:
                continue
            separation = self._geometry_separation_m(representative, candidate)
            if separation is None or separation > 80.0:
                continue
            compatible.append((separation, candidate))
        return min(compatible, key=lambda item: item[0])[1] if compatible else None

    @staticmethod
    def _same_road(first: dict[str, Any], second: dict[str, Any]) -> bool:
        if first.get("highway") != second.get("highway"):
            return False
        first_ref = re.sub(r"\s+", "", str(first.get("ref", ""))).casefold()
        second_ref = re.sub(r"\s+", "", str(second.get("ref", ""))).casefold()
        if first_ref or second_ref:
            return bool(first_ref and first_ref == second_ref)
        first_name = re.sub(r"\s+", " ", str(first.get("name", "")).strip()).casefold()
        second_name = re.sub(r"\s+", " ", str(second.get("name", "")).strip()).casefold()
        return bool(first_name and first_name == second_name)

    @staticmethod
    def _geometry_points(tags: dict[str, Any]) -> list[tuple[float, float]]:
        geometry = tags.get("_geometry")
        if not isinstance(geometry, list):
            return []
        points: list[tuple[float, float]] = []
        for point in geometry:
            if not isinstance(point, dict):
                continue
            try:
                points.append((float(point["lat"]), float(point["lon"])))
            except (KeyError, TypeError, ValueError):
                continue
        return points

    @classmethod
    def _travel_bearing(cls, tags: dict[str, Any]) -> float | None:
        points = cls._geometry_points(tags)
        if len(points) < 2:
            return None
        bearing = _bearing(points[0], points[-1])
        if str(tags.get("oneway", "")).strip() == "-1":
            bearing = (bearing + 180.0) % 360.0
        return bearing

    @classmethod
    def _geometry_separation_m(
        cls, first: dict[str, Any], second: dict[str, Any]
    ) -> float | None:
        first_points = sample_trajectory(cls._geometry_points(first), maximum=12)
        second_points = sample_trajectory(cls._geometry_points(second), maximum=12)
        if not first_points or not second_points:
            return None
        return min(_distance_m(a, b) for a in first_points for b in second_points)

    def _infer_terrain(
        self, points: list[tuple[float, float]]
    ) -> tuple[str, str | None]:
        if len(points) < 2:
            return "unknown", "Terrain could not be estimated from fewer than two GPS points."
        bearing = _bearing(points[0], points[-1])
        offsets: list[tuple[float, float]] = []
        for point in sample_trajectory(points, maximum=7):
            offsets.extend(
                [
                    _destination(point, bearing - 90.0, 150.0),
                    _destination(point, bearing + 90.0, 150.0),
                ]
            )
        try:
            elevations = self._elevations(offsets)
        except Exception as exc:
            return "unknown", f"Terrain elevation lookup failed: {exc}"
        slopes = [
            abs(elevations[index] - elevations[index + 1]) / 300.0 * 100.0
            for index in range(0, len(elevations) - 1, 2)
        ]
        if not slopes:
            return "unknown", "Terrain elevation lookup returned no usable cross-slope samples."
        slopes.sort()
        median_slope = slopes[len(slopes) // 2]
        if median_slope < 10:
            terrain = "plain"
        elif median_slope < 25:
            terrain = "rolling"
        elif median_slope < 40:
            terrain = "mountainous"
        else:
            terrain = "steep"
        return (
            terrain,
            "Terrain was estimated from 300 m cross-route Copernicus DEM GLO-90 samples "
            f"via Open-Meteo (median cross-slope {median_slope:.1f}%); verify for statutory use.",
        )

    def _elevations(self, points: list[tuple[float, float]]) -> list[float]:
        params = urllib.parse.urlencode(
            {
                "latitude": ",".join(f"{point[0]:.7f}" for point in points),
                "longitude": ",".join(f"{point[1]:.7f}" for point in points),
            }
        )
        request = urllib.request.Request(
            f"{self.elevation_url}?{params}", headers={"User-Agent": self.user_agent}
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s + 2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        values = payload.get("elevation")
        if not isinstance(values, list) or len(values) != len(points):
            raise RuntimeError("Elevation API returned an unexpected response.")
        return [float(value) for value in values]

    def _cache_path(self, points: list[tuple[float, float]]) -> Path | None:
        if self.cache_dir is None:
            return None
        payload = json.dumps(
            {
                "schema": ROAD_CONTEXT_CACHE_VERSION,
                "points": [[round(lat, 5), round(lon, 5)] for lat, lon in points],
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"gps_context_{digest}.json"

    def _read_cache(self, points: list[tuple[float, float]]) -> RoadContext | None:
        path = self._cache_path(points)
        if path is None or not path.exists():
            return None
        try:
            return RoadContext.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_cache(self, points: list[tuple[float, float]], context: RoadContext) -> None:
        path = self._cache_path(points)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(context.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def _parse_speed(value: str | None) -> float | None:
        if not value:
            return None
        match = re.search(r"\d+(?:\.\d+)?", value)
        if not match:
            return None
        speed = float(match.group())
        if "mph" in value.casefold():
            speed *= 1.609344
        return round(speed, 1)


def median_integer(values: Iterable[object]) -> int | None:
    clean: list[int] = []
    for value in values:
        try:
            numeric = int(float(value))
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            clean.append(numeric)
    if not clean:
        return None
    clean.sort()
    return clean[len(clean) // 2]


def _bearing(start: tuple[float, float], end: tuple[float, float]) -> float:
    lat1, lat2 = math.radians(start[0]), math.radians(end[0])
    delta_lon = math.radians(end[1] - start[1])
    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _destination(
    start: tuple[float, float], bearing_degrees: float, distance_m: float
) -> tuple[float, float]:
    radius = 6_371_000.0
    angular = distance_m / radius
    bearing = math.radians(bearing_degrees)
    lat1 = math.radians(start[0])
    lon1 = math.radians(start[1])
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular)
        + math.cos(lat1) * math.sin(angular) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(lat1),
        math.cos(angular) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _distance_m(first: tuple[float, float], second: tuple[float, float]) -> float:
    radius = 6_371_000.0
    lat1, lat2 = math.radians(first[0]), math.radians(second[0])
    delta_lat = lat2 - lat1
    delta_lon = math.radians(second[1] - first[1])
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * radius * math.asin(min(1.0, math.sqrt(value)))


def _distance_to_polyline_m(
    point: tuple[float, float], polyline: list[tuple[float, float]]
) -> float:
    if not polyline:
        return float("inf")
    if len(polyline) == 1:
        return _distance_m(point, polyline[0])
    latitude_scale = 111_320.0
    longitude_scale = latitude_scale * math.cos(math.radians(point[0]))

    def local(candidate: tuple[float, float]) -> tuple[float, float]:
        return (
            (candidate[1] - point[1]) * longitude_scale,
            (candidate[0] - point[0]) * latitude_scale,
        )

    best = float("inf")
    for start, end in zip(polyline, polyline[1:]):
        ax, ay = local(start)
        bx, by = local(end)
        dx, dy = bx - ax, by - ay
        denominator = dx * dx + dy * dy
        if denominator == 0:
            distance = math.hypot(ax, ay)
        else:
            projection = max(0.0, min(1.0, -(ax * dx + ay * dy) / denominator))
            distance = math.hypot(ax + projection * dx, ay + projection * dy)
        best = min(best, distance)
    return best


def statistics_median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("inf")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0
