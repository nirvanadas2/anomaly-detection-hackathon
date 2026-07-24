"""
Geo-IP style reference table: maps the "City, Country" strings that appear
in access_logs.csv's geo_location column to lat/lon, so geo-velocity can be
computed. This module knows nothing about how the data was generated -- it
stands in for a geo-IP lookup service a real detector would call.
"""

import math

CITY_COORDS = {
    "New York, USA": (40.7128, -74.0060),
    "Los Angeles, USA": (34.0522, -118.2437),
    "Chicago, USA": (41.8781, -87.6298),
    "Toronto, Canada": (43.6532, -79.3832),
    "Mexico City, Mexico": (19.4326, -99.1332),
    "Sao Paulo, Brazil": (-23.5505, -46.6333),
    "Buenos Aires, Argentina": (-34.6037, -58.3816),
    "Bogota, Colombia": (4.7110, -74.0721),
    "Lima, Peru": (-12.0464, -77.0428),
    "Santiago, Chile": (-33.4489, -70.6693),
    "London, UK": (51.5074, -0.1278),
    "Paris, France": (48.8566, 2.3522),
    "Berlin, Germany": (52.5200, 13.4050),
    "Madrid, Spain": (40.4168, -3.7038),
    "Rome, Italy": (41.9028, 12.4964),
    "Amsterdam, Netherlands": (52.3676, 4.9041),
    "Stockholm, Sweden": (59.3293, 18.0686),
    "Warsaw, Poland": (52.2297, 21.0122),
    "Vienna, Austria": (48.2082, 16.3738),
    "Zurich, Switzerland": (47.3769, 8.5417),
    "Dublin, Ireland": (53.3498, -6.2603),
    "Moscow, Russia": (55.7558, 37.6173),
    "Istanbul, Turkey": (41.0082, 28.9784),
    "Dubai, UAE": (25.2048, 55.2708),
    "Cairo, Egypt": (30.0444, 31.2357),
    "Lagos, Nigeria": (6.5244, 3.3792),
    "Nairobi, Kenya": (-1.2921, 36.8219),
    "Cape Town, South Africa": (-33.9249, 18.4241),
    "Mumbai, India": (19.0760, 72.8777),
    "Bangalore, India": (12.9716, 77.5946),
    "Singapore, Singapore": (1.3521, 103.8198),
    "Bangkok, Thailand": (13.7563, 100.5018),
    "Jakarta, Indonesia": (-6.2088, 106.8456),
    "Beijing, China": (39.9042, 116.4074),
    "Shanghai, China": (31.2304, 121.4737),
    "Seoul, South Korea": (37.5665, 126.9780),
    "Tokyo, Japan": (35.6762, 139.6503),
    "Sydney, Australia": (-33.8688, 151.2093),
    "Auckland, New Zealand": (-36.8485, 174.7633),
    "Helsinki, Finland": (60.1699, 24.9384),
    "Oslo, Norway": (59.9139, 10.7522),
}


def haversine_km(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def distance_km(geo_a, geo_b):
    """Returns None if either location is unknown to the reference table."""
    if geo_a not in CITY_COORDS or geo_b not in CITY_COORDS:
        return None
    if geo_a == geo_b:
        return 0.0
    return haversine_km(CITY_COORDS[geo_a], CITY_COORDS[geo_b])
