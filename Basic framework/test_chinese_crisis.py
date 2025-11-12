#!/usr/bin/env python3
"""
Quick test for Chinese language crisis detection
"""

import sys
import io
from chatbot_pipeline import run_pipeline

# Test 2: Crisis in Chinese
print("\n" + "="*70)
print("TESTING CHINESE CRISIS DETECTION")
print("Input: 我很抑郁，不想活了。(I'm very depressed and don't want to live.)")
print("="*70 + "\n")

inputs = [
    'TestUser',              # Name
    '我很抑郁，不想活了。',  # Crisis concern in Chinese
    'Boston',                # City  
    'MA',                    # State
    '02115',                 # Zip
    'I am from the US',      # US confirmation
    'n',                     # No insurance
    '',                      # Skip urgency
    'exit'                   # Exit early
]

original_stdin = sys.stdin
sys.stdin = io.StringIO('\n'.join(inputs))

try:
    run_pipeline()
except (KeyboardInterrupt, EOFError):
    pass
finally:
    sys.stdin = original_stdin

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
