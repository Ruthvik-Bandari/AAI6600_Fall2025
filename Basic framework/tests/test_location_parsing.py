#!/usr/bin/env python3
"""
Quick test script to verify location parsing works correctly
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from chatbot_pipeline import parse_location_input

# Test cases
test_cases = [
    ("Charlotte North Carolina", "Charlotte", "NC"),
    ("charlotte north carolina", "Charlotte", "NC"),
    ("Charlotte, NC", "Charlotte", "NC"),
    ("charlotte nc", "Charlotte", "NC"),
    ("charlotte, nc", "Charlotte", "NC"),
    ("New York, NY", "New York", "NY"),
    ("new york city", "New York City", None),  # No state given
    ("San Francisco California", "San Francisco", "CA"),
    ("Raleigh, North Carolina", "Raleigh", "NC"),
    ("DC", "Dc", None),  # Just city
    ("Los Angeles CA", "Los Angeles", "CA"),
    ("Chicago Illinois", "Chicago", "IL"),
]

print("="*70)
print("LOCATION PARSING TEST")
print("="*70)
print()

passed = 0
failed = 0

for input_str, expected_city, expected_state in test_cases:
    city, state = parse_location_input(input_str)
    
    # Check if results match expected
    city_match = city == expected_city if expected_city else city is None
    state_match = state == expected_state if expected_state else state is None
    
    if city_match and state_match:
        status = "✓ PASS"
        passed += 1
    else:
        status = "✗ FAIL"
        failed += 1
    
    print(f"{status}")
    print(f"  Input:    '{input_str}'")
    print(f"  Expected: city='{expected_city}', state='{expected_state}'")
    print(f"  Got:      city='{city}', state='{state}'")
    print()

print("="*70)
print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
print("="*70)

if failed == 0:
    print("\n✓ All tests passed!")
    sys.exit(0)
else:
    print(f"\n✗ {failed} test(s) failed")
    sys.exit(1)
