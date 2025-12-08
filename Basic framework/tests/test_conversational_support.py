#!/usr/bin/env python3
"""
Test Phase 6: Conversational Support System
Test case: User asks for help planning stressful day
"""

import sys
import io
from chatbot_pipeline import run_pipeline

print("="*70)
print("PHASE 6 TEST: Conversational Support System")
print("Scenario: User asks for help planning their stressful day")
print("="*70 + "\n")

# Test input sequence
test_input = """TestUser
Can you help me plan my stressful day? I have so much to do.
I have 3 exams this week, a job interview, and my mom is sick.
That's really helpful, thanks.
Yes, I'd like that.
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
print("1. Harbor detects 'help me plan' trigger")
print("2. Enters supportive conversation mode (2-3 turns)")
print("3. Provides stress management advice")
print("4. Asks about mental health support")
print("5. Transitions naturally to location collection")
print("6. Completes facility search and displays results")
print("="*70)
