# Copy this file to deploy_sh/devices_local.py and replace every placeholder.
# The addresses below are RFC 5737 documentation addresses and are not routable.

SERVER_DEVICES = [
    ("192.0.2.10", "CHANGE_ME", "CHANGE_ME"),
]

# List clients in CID order. The first entry is CID 0, the second is CID 1,
# and so on.
CLIENT_DEVICES = [
    ("198.51.100.20", "CHANGE_ME", "CHANGE_ME"),
    ("198.51.100.21", "CHANGE_ME", "CHANGE_ME"),
    ("198.51.100.22", "CHANGE_ME", "CHANGE_ME"),
    ("198.51.100.23", "CHANGE_ME", "CHANGE_ME"),
    ("198.51.100.24", "CHANGE_ME", "CHANGE_ME"),
    ("198.51.100.25", "CHANGE_ME", "CHANGE_ME"),
    ("198.51.100.26", "CHANGE_ME", "CHANGE_ME"),
    ("198.51.100.27", "CHANGE_ME", "CHANGE_ME"),
    ("198.51.100.28", "CHANGE_ME", "CHANGE_ME"),
    ("198.51.100.29", "CHANGE_ME", "CHANGE_ME"),
]

ALL_DEVICES = list(CLIENT_DEVICES)
