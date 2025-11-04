def pytest_sessionfinish(session, exitstatus):
    """
    PyTest hook: runs after the whole test session finishes.
    If all tests passed (exitstatus == 0), print a short, consistent
    recap message as requested.
    """
    if exitstatus == 0:
        # Exact two-sentence recap requested by the user
        print("\nThe tests exercised fast_search_scored_csv with a small sample CSV and confirmed it returns sorted results and correctly filters by state and by city (case-insensitive).")
        print("All three assertions matched expected outputs, so the test suite passed.")
