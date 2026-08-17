"""Rental listings and the focused replication design."""

from __future__ import annotations


ATTRIBUTES = (
    ("Monthly Rent", "USD/month", "lower is better"),
    ("Transit Time", "minutes", "lower is better"),
    ("Internal Floor Area", "m^2", "higher is better"),
    ("Natural Light", "hours/day", "higher is better"),
    ("Neighborhood Noise", "dB", "lower is better"),
    ("Minimum Lease", "months", "lower is better"),
    ("Walk to Essentials", "minutes", "lower is better"),
    ("Energy Efficiency", "1-10", "higher is better"),
    ("Private Outdoor Area", "m^2", "higher is better"),
    ("Modernity", "1-10", "higher is better"),
)

HOUSES = {
    "A": (2400, 8, 42, 3, 68, 12, 4, 6, 0, 8),
    "B": (1750, 35, 110, 6, 42, 12, 15, 5, 25, 4),
    "C": (2100, 18, 85, 5, 61, 6, 8, 4, 3, 6),
    "D": (1150, 12, 38, 4, 55, 9, 5, 5, 2, 5),
    "E": (1950, 22, 95, 7, 48, 24, 10, 8, 12, 9),
    "F": (900, 55, 130, 8, 30, 6, 25, 3, 60, 2),
    "G": (1600, 10, 28, 2, 58, 3, 3, 9, 4, 10),
    "H": (1850, 15, 70, 6, 50, 12, 6, 7, 15, 7),
    "I": (1300, 25, 90, 4, 52, 12, 7, 2, 5, 3),
    "J": (2250, 6, 60, 5, 45, 18, 2, 9, 8, 9),
}

# Pairs whose order-consistency status differed at k=3, k=5, or k=10 in the
# published experiment. This is a deliberately post-hoc, targeted replication.
TARGET_PAIRS = ("BG", "BH", "CG", "DH", "EG", "EI", "HI", "GJ")

# Drawn once, without replacement, from the other 37 pairs with
# random.Random(20260817). These are the confirmatory rather than post-hoc
# selected pairs.
CONFIRMATORY_PAIRS = (
    "AD", "DE", "AE", "IJ", "FJ", "AJ", "CF", "AH", "DF", "AF", "CJ", "EF"
)

STUDY_PAIRS = TARGET_PAIRS + CONFIRMATORY_PAIRS
