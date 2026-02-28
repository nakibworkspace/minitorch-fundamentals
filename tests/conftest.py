"""Pytest configuration for MiniTorch."""

import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "task0_1: Task 0.1 operators")
    config.addinivalue_line("markers", "task0_2: Task 0.2 property tests") 
    config.addinivalue_line("markers", "task0_3: Task 0.3 functional")
    config.addinivalue_line("markers", "task0_4: Task 0.4 modules")
    # Module 1 markers
    config.addinivalue_line("markers", "task1_1: Task 1.1 numerical derivatives")
    config.addinivalue_line("markers", "task1_2: Task 1.2 scalar forward")
    config.addinivalue_line("markers", "task1_3: Task 1.3 chain rule")
    config.addinivalue_line("markers", "task1_4: Task 1.4 backpropagation")
    config.addinivalue_line("markers", "task1_5: Task 1.5 training")