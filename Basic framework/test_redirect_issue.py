#!/usr/bin/env python3
"""
Test Chinese with actual pipeline to see where redirect happens
"""

import sys
import io

# Test what the pipeline actually receives
test_input = """TestUser
我感到很焦虑，需要帮助。
Boston
MA
02115
I am from the US
n

exit
"""

print("SIMULATING FULL PIPELINE WITH CHINESE INPUT")
print("="*70)
print("Input sequence:")
print("1. Name: TestUser")
print("2. Concern: 我感到很焦虑，需要帮助。")
print("="*70)

sys.stdin = io.StringIO(test_input)

try:
    from chatbot_pipeline import run_pipeline
    run_pipeline()
except (KeyboardInterrupt, EOFError, Exception) as e:
    print(f"\nStopped: {e}")
