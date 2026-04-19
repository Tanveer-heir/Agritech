"""
Location Agent
==============
Given a farmer's PIN code, finds the nearest Krishi Vigyan Kendra (KVK)
within 50km and returns contact details.
"""

import json
import os
from pathlib import Path


def find_nearest_kvk(pin_code: str) -> dict:
    """
    Look up the nearest KVK for a given Indian PIN code.

    Args:
        pin_code: 6-digit Indian postal PIN code

    Returns:
        {
            "name": str,
            "district": str,
            "state": str,
            "phone": str,
            "distance_km": float
        }
    """
    # TODO: Load kvk_centers.json
    # TODO: Match PIN code to district
    # TODO: Return nearest KVK within 50km
    # TODO: Fallback: return state-level KVK helpline
    raise NotImplementedError("location_agent.find_nearest_kvk() not yet implemented")
