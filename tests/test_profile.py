import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from profile import ORDER, load_sets, max_abs_residual, summarize

class ProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sets = load_sets(ROOT / "data" / "anscombe.csv")

    def test_four_series_of_eleven(self) -> None:
        self.assertEqual(tuple(self.sets), ORDER)
        for name in ORDER:
            self.assertEqual(len(self.sets[name]), 11)

    def test_summaries_match_across_series(self) -> None:
        rows = [summarize(self.sets[name]) for name in ORDER]
        for row in rows:
            self.assertEqual(row["n"], 11)
            self.assertAlmostEqual(row["mean_x"], 9.0, places=3)
            self.assertAlmostEqual(row["mean_y"], 7.5, places=2)
            self.assertAlmostEqual(row["var_x"], 11.0, places=3)
            self.assertAlmostEqual(row["var_y"], 4.125, places=2)
            self.assertAlmostEqual(row["corr"], 0.816, places=2)

    def test_set_iii_sits_farther_from_the_line(self) -> None:
        far = max_abs_residual(self.sets["III"])
        near = max_abs_residual(self.sets["I"])
        self.assertGreater(far, near)
        self.assertGreater(far, 3.0)
        self.assertLess(near, 2.0)

    def test_set_iv_x_is_almost_constant(self) -> None:
        xs = [x for x, _ in self.sets["IV"]]
        self.assertEqual(xs.count(8), 10)
        self.assertIn(19, xs)


if __name__ == "__main__":
    unittest.main()
