"""Lightweight inference constants shared by the runner and policy driver."""

DECODE_MODES = ("greedy", "official")
# Offline oracle ablations only; the deployed judge does not read the route.
ORACLE_ROUTE_MARGIN = 2.0
