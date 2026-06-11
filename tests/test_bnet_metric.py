import unittest

import numpy as np

from bnet.metric import (
    BnetResult,
    _split_trapezium_areas,
    calculate_bnet,
)


class BnetMetricTests(unittest.TestCase):
    def test_calculate_bnet_matches_reference_values(self) -> None:
        cases = (
            (
                (0.8, 0.9, 1.0, 1.1, 1.2),
                (0.7, 0.9, 1.2, 1.8, 2.1),
                1.9029557835725344,
                0.3438851593644208,
                0.6543982528972873,
            ),
            (
                (0.5, 0.7, 0.8, 1.0, 4.0),
                (0.6, 0.7, 0.9, 1.1, 3.0),
                1.51272659749217,
                0.3970935230383145,
                0.6006939339919282,
            ),
        )

        for all_bdamage_values, bnet_site_values, bnet, left_area, right_area in cases:
            with self.subTest(bnet_site_values=bnet_site_values):
                result = calculate_bnet(
                    all_bdamage_values=all_bdamage_values,
                    bnet_site_bdamage_values=bnet_site_values,
                )

                self.assertIsInstance(result, BnetResult)
                self.assertAlmostEqual(result.bnet, bnet, places=12)
                self.assertAlmostEqual(result.left_area, left_area, places=12)
                self.assertAlmostEqual(result.right_area, right_area, places=12)
                self.assertEqual(result.site_count, len(bnet_site_values))

    def test_tail_padded_kde_handles_sites_above_all_atom_median(self) -> None:
        result = calculate_bnet(
            all_bdamage_values=(0.8, 0.9, 1.0, 1.1, 1.2),
            bnet_site_bdamage_values=(1.2, 1.4, 1.8, 2.2, 2.8),
        )

        self.assertAlmostEqual(result.bnet, 7.086895177174404, places=12)
        self.assertAlmostEqual(result.left_area, 0.12345074451835888, places=12)
        self.assertAlmostEqual(result.right_area, 0.8748824859457471, places=12)

    def test_split_trapezium_areas_uses_left_edge_rule(self) -> None:
        left_area, right_area = _split_trapezium_areas(
            x_values=np.array([0.0, 1.0, 2.0]),
            y_values=np.array([1.0, 1.0, 1.0]),
            split_value=1.0,
        )

        self.assertEqual(left_area, 1.0)
        self.assertEqual(right_area, 1.0)

    def test_invalid_inputs_raise_value_error(self) -> None:
        invalid_cases = (
            {
                "all_bdamage_values": (),
                "bnet_site_bdamage_values": (1.0, 2.0),
            },
            {
                "all_bdamage_values": (1.0, float("nan")),
                "bnet_site_bdamage_values": (1.0, 2.0),
            },
            {
                "all_bdamage_values": (1.0, 2.0),
                "bnet_site_bdamage_values": (1.0,),
            },
            {
                "all_bdamage_values": (1.0, 2.0),
                "bnet_site_bdamage_values": (1.0, 1.0),
            },
        )

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    calculate_bnet(**kwargs)

    def test_invalid_trapezium_count_raises_value_error(self) -> None:
        for trapezium_count in (0, -1, 1.5, True):
            with self.subTest(trapezium_count=trapezium_count):
                with self.assertRaises(ValueError):
                    calculate_bnet(
                        all_bdamage_values=(0.8, 0.9, 1.0),
                        bnet_site_bdamage_values=(0.7, 1.2),
                        trapezium_count=trapezium_count,
                    )

    def test_numpy_integer_trapezium_count_is_accepted(self) -> None:
        result = calculate_bnet(
            all_bdamage_values=(0.8, 0.9, 1.0, 1.1, 1.2),
            bnet_site_bdamage_values=(0.7, 0.9, 1.2, 1.8, 2.1),
            trapezium_count=np.int64(99),
        )

        self.assertAlmostEqual(result.bnet, 1.9029557835725344, places=12)
