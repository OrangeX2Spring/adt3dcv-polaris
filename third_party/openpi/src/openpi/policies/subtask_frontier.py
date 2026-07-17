from collections.abc import Sequence


SUBTASK_CHAINS = ((0, 2, 4), (1, 3, 5))


def subtask_frontier(done: Sequence[bool], goal_available: Sequence[bool]) -> list[int]:
    frontier = []
    for chain in SUBTASK_CHAINS:
        next_subtask = next((index for index in chain if not done[index]), None)
        if next_subtask is not None and goal_available[next_subtask]:
            frontier.append(next_subtask)
    return frontier
