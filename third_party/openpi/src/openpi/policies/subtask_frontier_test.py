import unittest

from openpi.policies.subtask_frontier import subtask_frontier


class SubtaskFrontierTest(unittest.TestCase):
    def test_subtask_frontier(self):
        cases = [
            ([False, False, False, False, False, False], [0, 1]),
            ([True, False, False, False, False, False], [2, 1]),
            ([True, True, False, False, False, False], [2, 3]),
            ([True, False, True, False, False, False], [4, 1]),
            ([False, False, True, False, False, False], [0, 1]),
            ([True, False, True, False, True, False], [1]),
            ([True, True, True, True, True, True], []),
        ]
        for done, expected in cases:
            with self.subTest(done=done):
                self.assertEqual(subtask_frontier(done, [True] * 6), expected)

    def test_does_not_skip_missing_goal(self):
        self.assertEqual(
            subtask_frontier([False] * 6, [False, True, True, True, True, True]),
            [1],
        )
