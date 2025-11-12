#!/usr/bin/env python3
"""
Quick test script for the hybrid crisis detection system
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from chatbot_pipeline import detect_crisis_hybrid

# Test cases
test_messages = [
    # Should detect as CRISIS
    ("I want to end my life", True),
    ("im thinking about ending it", True),
    ("I feel like hurting myself", True),
    ("I can't go on anymore", True),
    ("everyone would be better off without me", True),
    ("I'm planning to kill myself", True),
    
    # Should NOT detect as crisis
    ("I'm feeling anxious and need help", False),
    ("I've been really depressed lately", False),
    ("I'm having panic attacks", False),
    ("I need help with my mental health", False),
    ("I'm struggling with substance abuse", False),
]

print("="*70)
print("TESTING HYBRID CRISIS DETECTION SYSTEM")
print("="*70)
print()

passed = 0
failed = 0

for message, expected_crisis in test_messages:
    result = detect_crisis_hybrid(message)
    is_crisis = result['is_crisis']
    confidence = result['confidence']
    method = result['method']
    
    status = "✅ PASS" if is_crisis == expected_crisis else "❌ FAIL"
    if is_crisis == expected_crisis:
        passed += 1
    else:
        failed += 1
    
    print(f"{status} | Crisis: {is_crisis} (conf: {confidence:.2f}) | Method: {method}")
    print(f"       Message: \"{message}\"")
    if is_crisis != expected_crisis:
        print(f"       Expected: {expected_crisis}, Got: {is_crisis}")
    print()

print("="*70)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
print("="*70)
