#!/usr/bin/env python3
"""
Test Phase 6: Direct Resource Request (should skip conversation)
Test case: User directly asks to find a therapist
"""

import sys
import io
from chatbot_pipeline import run_pipeline

print("="*70)
print("PHASE 6 TEST: Direct Resource Request")
print("Scenario: User directly asks to find a therapist")
print("="*70 + "\n")

# Test input sequence - direct request
test_input = """TestUser
I need to find a therapist for my anxiety.
Boston
MA
02115
I am from the US
y
Blue Cross
"""

sys.stdin = io.StringIO(test_input)

try:
    result = run_pipeline()
    print("\n" + "="*70)
    print("✓ Test Complete")
    print("="*70)
except (KeyboardInterrupt, EOFError, Exception) as e:
    print(f"\nTest stopped: {e}")
finally:
    sys.stdin = sys.__stdin__

print("\n" + "="*70)
print("EXPECTED BEHAVIOR:")
print("1. Harbor detects 'find a therapist' (direct request)")
print("2. SKIPS conversation mode (no advice turns)")
print("3. Goes directly to location collection")
print("4. Completes facility search and displays results")
print("="*70)
