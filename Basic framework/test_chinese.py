#!/usr/bin/env python3
"""
Test script for Chinese language support (Phase 5)
Tests: anxiety, crisis, therapy intent recognition
"""

import sys
import io
from chatbot_pipeline import run_pipeline

def test_chinese_scenarios():
    """Test Chinese language support with various scenarios"""
    
    test_cases = [
        {
            'name': '我感到很焦虑，需要帮助。',
            'english': 'I feel very anxious and need help.',
            'expected': 'Non-crisis anxiety response in Chinese'
        },
        {
            'name': '我很抑郁，不想活了。',
            'english': 'I\'m very depressed and don\'t want to live.',
            'expected': 'Crisis detection + Chinese resources'
        },
        {
            'name': '我需要心理治疗。',
            'english': 'I need psychotherapy.',
            'expected': 'Mental health intent recognized'
        }
    ]
    
    print("="*70)
    print("PHASE 5: MANDARIN CHINESE LANGUAGE SUPPORT TEST")
    print("="*70 + "\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {test['name']}")
        print(f"English: {test['english']}")
        print(f"Expected: {test['expected']}")
        print(f"{'='*70}\n")
        
        # Create mock inputs
        inputs = [
            'TestUser',    # Name (first question)
            test['name'],  # User concern in Chinese (second question)
            'Boston',      # City
            'MA',          # State
            '02115',       # Zip
            'I am from the US',  # US confirmation
            'n'            # No insurance (to test basic flow)
        ]
        
        # Redirect stdin to simulate user input
        original_stdin = sys.stdin
        sys.stdin = io.StringIO('\n'.join(inputs))
        
        try:
            print(f"Running pipeline with Chinese input: {test['name']}")
            print("-"*70)
            run_pipeline()
            print("-"*70)
            print(f"✓ Test {i} completed\n")
        except Exception as e:
            print(f"✗ Test {i} failed with error: {e}\n")
        finally:
            sys.stdin = original_stdin
        
        # Pause between tests
        if i < len(test_cases):
            print("\n" + "."*70)
            print("Press Enter to continue to next test...")
            print("."*70)
            input()

if __name__ == "__main__":
    test_chinese_scenarios()
