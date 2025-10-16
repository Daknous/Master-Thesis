#!/usr/bin/env python3
"""
Count transformers inside an OSM substation.

Usage examples:
  python count_substation_transformers.py way/24767088
  python count_substation_transformers.py relation/654321 --verbose
  python count_substation_transformers.py way/940635546 --radius 300

Notes & limitations:
- Works best when the substation is mapped as an AREA (closed way or multipolygon relation).
- For NODE substations, we can only approximate via a radius search (no true boundary).
- OSM completeness varies: missing or inconsistently mapped transformers will reduce the count.
"""

import argparse
import sys
import requests
from typing import Dict, List, Tuple

OVERPASS_URL_DEFAULT = "https://overpass-api.de/api/interpreter"
UA = "OSM-TransformerCounter/1.0"

class OverpassError(RuntimeError):
    pass

def parse_osm_ref(ref: str) -> Tuple[str, int]:
    """
    Parse an OSM reference like 'way/123', 'relation/456', 'node/789',
    or short forms 'w123', 'r456', 'n789'.
    Returns (osm_type, osm_id) with osm_type in {'node','way','relation'}.
    """
    ref = ref.strip().lower()
    if "/" in ref:
        t, i = ref.split("/", 1)
        t = {"rel": "relation", "relation": "relation", "way": "way", "node": "node"}.get(t, t)
        return t, int(i)
    # short forms
    if ref.startswith(("w", "r", "n")) and ref[1:].isdigit():
        tmap = {"w": "way", "r": "relation", "n": "node"}
        return tmap[ref[0]], int(ref[1:])
    raise ValueError("OSM reference must look like 'way/123', 'relation/456', 'node/789', 'w123', 'r456', or 'n789'.")

def overpass(query: str, endpoint: str) -> Dict:
    resp = requests.post(
        endpoint,
        data={"data": query},
        headers={"User-Agent": UA, "Accept": "application/json"},
        timeout=120
    )
    if resp.status_code != 200:
        raise OverpassError(f"Overpass HTTP {resp.status_code}: {resp.text[:200]}")
    js = resp.json()
    if "remark" in js and js["remark"]:
        # Overpass sometimes returns 'remark' with errors/warnings
        # Treat as error if no elements are returned
        if not js.get("elements"):
            raise OverpassError(f"Overpass remark: {js['remark']}")
    return js

def get_element_tags_and_center(osm_type: str, osm_id: int, endpoint: str) -> Dict:
    """
    Fetch the element's tags and center (if available).
    """
    q = f"""
    [out:json][timeout:60];
    {osm_type}({osm_id});
    out tags center;
    """
    js = overpass(q, endpoint)
    if not js.get("elements"):
        raise OverpassError(f"{osm_type} {osm_id} not found on OSM.")
    return js["elements"][0]

def count_transformers_in_area(osm_type: str, osm_id: int, endpoint: str) -> Tuple[int, List[str]]:
    """
    Count transformers using an AREA derived from a closed way or relation.
    Overpass 'area' id is 3600000000 + OSM id (works for closed ways and relations).
    """
    area_id = 3600000000 + osm_id
    q = f"""
    [out:json][timeout:120];
    area({area_id})->.a;
    (
      node["power"="transformer"](area.a);
      way["power"="transformer"](area.a);
      relation["power"="transformer"](area.a);
    );
    out ids;
    """
    js = overpass(q, endpoint)
    ids = []
    for el in js.get("elements", []):
        et = el.get("type")
        eid = el.get("id")
        if et and eid is not None:
            ids.append(f"{et}/{eid}")
    # de-duplicate (paranoia)
    ids = sorted(set(ids))
    return len(ids), ids

def count_transformers_around_point(lat: float, lon: float, radius_m: int, endpoint: str) -> Tuple[int, List[str]]:
    """
    Fallback for node-type substations: approximate by radius search.
    """
    q = f"""
    [out:json][timeout:60];
    (
      node["power"="transformer"](around:{radius_m},{lat},{lon});
      way["power"="transformer"](around:{radius_m},{lat},{lon});
      relation["power"="transformer"](around:{radius_m},{lat},{lon});
    );
    out ids;
    """
    js = overpass(q, endpoint)
    ids = []
    for el in js.get("elements", []):
        et = el.get("type")
        eid = el.get("id")
        if et and eid is not None:
            ids.append(f"{et}/{eid}")
    ids = sorted(set(ids))
    return len(ids), ids

def maybe_bbox_fallback_for_way(osm_id: int, endpoint: str, pad_deg: float = 0.0003) -> Tuple[int, List[str]]:
    """
    If area() fails (e.g., the given way isn't considered an area), we do a bounding-box fallback.
    This is less precise but better than nothing.
    """
    # Fetch full geometry for the way to compute a bbox
    q_geom = f"""
    [out:json][timeout:60];
    way({osm_id});
    out geom;
    """
    js = overpass(q_geom, endpoint)
    if not js.get("elements"):
        return 0, []
    geom = js["elements"][0].get("geometry", [])
    if not geom:
        return 0, []
    lats = [pt["lat"] for pt in geom]
    lons = [pt["lon"] for pt in geom]
    south, north = min(lats) - pad_deg, max(lats) + pad_deg
    west, east = min(lons) - pad_deg, max(lons) + pad_deg

    q_bbox = f"""
    [out:json][timeout:120];
    (
      node["power"="transformer"]({south},{west},{north},{east});
      way["power"="transformer"]({south},{west},{north},{east});
      relation["power"="transformer"]({south},{west},{north},{east});
    );
    out ids;
    """
    js2 = overpass(q_bbox, endpoint)
    ids = []
    for el in js2.get("elements", []):
        et = el.get("type")
        eid = el.get("id")
        if et and eid is not None:
            ids.append(f"{et}/{eid}")
    ids = sorted(set(ids))
    return len(ids), ids

def main():
    ap = argparse.ArgumentParser(description="Count OSM transformers within a substation.")
    ap.add_argument("osm_ref", help="OSM element like 'way/123456', 'relation/654321', or 'node/987654'. Short forms 'w123', 'r456', 'n789' also work.")
    ap.add_argument("--endpoint", default=OVERPASS_URL_DEFAULT, help=f"Overpass API endpoint (default: {OVERPASS_URL_DEFAULT})")
    ap.add_argument("--radius", type=int, default=250, help="Radius in meters for NODE fallback (default: 250).")
    ap.add_argument("--verbose", action="store_true", help="Print found transformer IDs.")
    args = ap.parse_args()

    try:
        osm_type, osm_id = parse_osm_ref(args.osm_ref)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Fetch basic info (tags + center if any)
    try:
        meta = get_element_tags_and_center(osm_type, osm_id, args.endpoint)
    except Exception as e:
        print(f"Lookup failed: {e}", file=sys.stderr)
        sys.exit(1)

    tags = meta.get("tags", {})
    power_tag = tags.get("power")
    if power_tag != "substation":
        # Not strictly required, but helpful info
        print(f"Warning: {osm_type}/{osm_id} does not have power=substation (power={power_tag}). Continuing...", file=sys.stderr)

    try:
        if osm_type in ("way", "relation"):
            # Primary path: use area()
            count, ids = count_transformers_in_area(osm_type, osm_id, args.endpoint)
            if count == 0 and osm_type == "way":
                # Some ways are not treated as areas; try bbox fallback
                count, ids = maybe_bbox_fallback_for_way(osm_id, args.endpoint)
        elif osm_type == "node":
            center = meta.get("center")
            if not center:
                raise OverpassError("No coordinates available for the node.")
            lat, lon = center["lat"], center["lon"]
            count, ids = count_transformers_around_point(lat, lon, args.radius, args.endpoint)
        else:
            raise ValueError("Unsupported OSM type.")
    except Exception as e:
        print(f"Query failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Transformers found: {count}")
    if args.verbose and count > 0:
        print("IDs:")
        for i in ids:
            print(f"  {i}")

if __name__ == "__main__":
    main()
