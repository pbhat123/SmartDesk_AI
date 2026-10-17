"""Loads scripts/seed_tickets.py (not a package) so tests can seed a mock client the way the script does."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("seed_tickets", ROOT / "scripts" / "seed_tickets.py")
seed_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed_module)

seed = seed_module.seed
ZERO_EMAIL, ONE_EMAIL, MANY_EMAIL, CLOSED_EMAIL = (
    seed_module.ZERO_EMAIL, seed_module.ONE_EMAIL, seed_module.MANY_EMAIL, seed_module.CLOSED_EMAIL,
)
HAND_MADE_EMAIL = seed_module.HAND_MADE_EMAIL
