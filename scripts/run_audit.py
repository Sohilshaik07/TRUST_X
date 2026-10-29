"""
Script to execute dataset audit for Phase 1.
"""
import sys
import os

# Add root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from src.data.audit import run_comprehensive_audit

if __name__ == "__main__":
    run_comprehensive_audit()
