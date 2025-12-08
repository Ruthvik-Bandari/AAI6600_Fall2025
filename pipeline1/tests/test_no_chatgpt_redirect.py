#!/usr/bin/env python3
"""
Test that unrelated requests don't recommend other chatbots
"""

import sys
import io
from chatbot_pipeline import run_pipeline

print("="*70)
print("TEST: Unrelated Request (should NOT recommend ChatGPT)")
print("="*70)

# Test unrelated request
test_input = """TestUser
What's the weather today?
no
"""

sys.stdin = io.StringIO(test_input)

try:
    run_pipeline()
except (KeyboardInterrupt, EOFError, Exception) as e:
    print(f"\nTest complete: {e}")
finally:
    sys.stdin = sys.__stdin__

print("\n" + "="*70)
print("VERIFICATION: Check that output does NOT mention:")
print("  ❌ 'ChatGPT'")
print("  ❌ 'Google'")
print("  ❌ 'other AI assistant'")
print("  ❌ 'try another service'")
print("\nInstead it SHOULD:")
print("  ✅ Keep user engaged with mental health focus")
print("  ✅ Ask if they need mental health support")
print("  ✅ Invite them to clarify their needs")
print("="*70)
