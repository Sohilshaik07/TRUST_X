"""
Script to execute data preprocessing pipeline (Phase 2).
"""
import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from src.data.preprocessing import run_all_preprocessing

if __name__ == "__main__":
    run_all_preprocessing()
