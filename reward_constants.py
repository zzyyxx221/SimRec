"""Reward constants for the active native-tool SimRec training path."""

import os


FORMAT_BONUS = float(os.environ.get("SIMREC_FORMAT_BONUS", "0.01"))
VALID_SEARCH_BONUS = float(os.environ.get("SIMREC_VALID_SEARCH_BONUS", "0.03"))
VALID_ITEM_DETAILS_BONUS = float(os.environ.get("SIMREC_VALID_ITEM_DETAILS_BONUS", "0.01"))
VALID_USER_PREFERENCE_BONUS = float(os.environ.get("SIMREC_VALID_USER_PREFERENCE_BONUS", "0.0"))
