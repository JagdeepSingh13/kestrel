import unittest

from calc.stats import mean, middle


class TestStats(unittest.TestCase):
    def test_mean(self) -> None:
        self.assertEqual(mean([2, 4, 6]), 4)

    def test_mean_of_nothing(self) -> None:
        with self.assertRaises(ValueError):
            mean([])

    def test_middle(self) -> None:
        self.assertEqual(middle([3, 1, 2]), 2)
        self.assertEqual(middle([4, 1, 3, 2]), 2.5)
