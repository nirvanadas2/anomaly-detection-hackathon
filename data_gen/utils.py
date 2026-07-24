"""Small shared helpers: geodesic distance and sequential ID generation."""

import math


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in kilometers."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class IdCounter:
    """Simple monotonic counter used to mint entity/event IDs with a prefix."""

    def __init__(self, prefix, width=6):
        self.prefix = prefix
        self.width = width
        self._n = 0

    def next(self):
        self._n += 1
        return f"{self.prefix}-{self._n:0{self.width}d}"
