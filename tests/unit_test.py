import unittest
from pathlib import Path

class UnitTest(unittest.TestCase):

    root = Path(__file__).absolute().parent
    input_dir = root / 'inputs'
    output_dir = root / 'outputs'
    sample_trajectory_data = input_dir / 'sample_trajectory.csv'