#!/usr/bin/env python3
"""
Debug Chinese intent classification
"""

from chatbot_pipeline import classify_user_intent_lightweight

test_cases = [
    "我感到很焦虑，需要帮助。",  # I feel very anxious and need help
    "我很抑郁，不想活了。",  # I'm very depressed and don't want to live
    "我需要心理治疗。",  # I need psychotherapy
    "我有压力，睡不着觉。",  # I have stress, can't sleep
]

print("="*70)
print("DEBUGGING CHINESE INTENT CLASSIFICATION")
print("="*70 + "\n")

for i, text in enumerate(test_cases, 1):
    print(f"Test {i}: {text}")
    result = classify_user_intent_lightweight(text)
    print(f"  Intent: {result['intent']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Method: {result['method']}")
    print(f"  Needs Redirect: {result['needs_redirect']}")
    print()
