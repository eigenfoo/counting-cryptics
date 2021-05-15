#!/usr/bin/env python
# coding: utf-8

# # Counting Cryptic Crossword Grids

# In[1]:


import itertools
import numpy as np


# In[2]:


GRID_LENGTH = 5               # Must be odd, and must be > 3
MIN_WORD_LENGTH = 3           # Minimum allowable word length
ROUND_UP_UNCHECKED = True     # I.e. must a 7-letter word have 4 unchecked squares?
ONLY_FIND_INTERESTING = True  # Restrict search to only "interesting" grids


# In[3]:


# For now, True is white, False is black. We will reverse this later.
grid = np.meshgrid(*[[0, 1] for _ in range(GRID_LENGTH)])
grid = [mesh.flatten() for mesh in grid]
grid = np.transpose(np.vstack(grid))
grid = grid.astype(bool)


# In[4]:


def contains_run_length(row, length):
    """
    If True, `row` contains a run of True values of length equal to `length`
    E.g.
        row_contains_run_length([0, 1, 1], 1) returns False
        row_contains_run_length([0, 1, 1], 2) returns True
    """
    # This is why we have white and black flipped - to sum lengths of _whites_
    return any([sum(group) == length for _, group in itertools.groupby(row)])


# In[5]:


# Compute all valid rows

for length in range(2, MIN_WORD_LENGTH):
    rows_to_delete = []
    for i, row in enumerate(grid):
        if contains_run_length(row, length):
            rows_to_delete.append(i)
    grid = np.delete(grid, rows_to_delete, axis=0)

valid_rows = np.copy(grid)
valid_rows = ~valid_rows  # True is black, False is white
valid_rows = valid_rows[
    np.sum(valid_rows, axis=1) < GRID_LENGTH
]  # Remove all-black row


# In[6]:


# Compute all symmetric rows

symmetric_row_indexes = []
for i, row in enumerate(grid):
    if all(row == np.flip(row)):
        symmetric_row_indexes.append(i)

symmetric_rows = np.copy(grid[symmetric_row_indexes])
symmetric_rows = ~symmetric_rows  # True is black, False is white
symmetric_rows = symmetric_rows[
    np.sum(symmetric_rows, axis=1) < 13
]  # Remove all-black row


# In[7]:


# A grid is 'interesting' if each row has at least some number of white squares.
# See `min_white_squares` for the mapping. This was decided by hand.
if ONLY_FIND_INTERESTING:
    min_white_squares = {5: 2, 7: 3, 9: 4, 11: 4, 13: 4, 15: 5}

    valid_rows = valid_rows[(~valid_rows).sum(axis=1) >= min_white_squares[GRID_LENGTH]]
    symmetric_rows = symmetric_rows[
        (~symmetric_rows).sum(axis=1) >= min_white_squares[GRID_LENGTH]
    ]

flipped_valid_rows = np.flip(valid_rows, axis=1)  # Precompute flipped rows for later


# In[8]:


def row_is_valid(grid, row_num):
    (word_starts_indexes,) = np.where(
        np.append(
            [np.logical_not(grid[row_num][0])], np.diff(grid[row_num].astype(int)) == -1
        )
    )
    (word_ends_indexes,) = np.where(
        np.append(
            np.diff(grid[row_num].astype(int)) == 1, [np.logical_not(grid[row_num][-1])]
        )
    )

    for word_start_index, word_end_index in zip(word_starts_indexes, word_ends_indexes):
        subgrid = grid[:, word_start_index : word_end_index + 1]
        if 0 < row_num < grid.shape[0] - 1:
            unchecked_squares = np.logical_and(*subgrid[[row_num - 1, row_num + 1]])
        elif row_num == 0:
            unchecked_squares = subgrid[1]
        elif row_num == grid.shape[0] - 1:
            unchecked_squares = subgrid[grid.shape[0] - 2]

        expected_num_unchecked = (word_end_index - word_start_index) / 2
        has_expected_num_unchecked = (
            sum(unchecked_squares) == np.ceil(expected_num_unchecked)
            if ROUND_UP_UNCHECKED
            else (
                np.floor(expected_num_unchecked)
                <= sum(unchecked_squares)
                <= np.ceil(expected_num_unchecked)
            )
        )
        has_max_two_consecutive_unchecked = all(
            [
                sum(group) <= 2
                for bit, group in itertools.groupby(unchecked_squares)
                if bit
            ]
        )
        ends_not_two_consecutive_unchecked = not (
            all(unchecked_squares[:2]) or all(unchecked_squares[-2:])
        )

        if not (
            has_expected_num_unchecked
            and has_max_two_consecutive_unchecked
            and ends_not_two_consecutive_unchecked
        ):
            return False

    return True


# In[9]:


valid_middle_rows_indexes = []

for i, row1 in enumerate(valid_rows):
    for j, row2 in enumerate(symmetric_rows):
        grid = np.vstack(
            [
                row1,
                row2,
                np.flip(row1),
            ]
        )
        if all(
            [
                row_is_valid(
                    grid,
                    k,
                )
                for k in range(2)  # By symmetry, only need to check 0 and 1
            ]
        ):
            valid_middle_rows_indexes.append((i, j))


# In[10]:


def grid_from_indexes(*indexes):
    return np.vstack(
        [valid_rows[i] for i in indexes[:-1]]
        + [symmetric_rows[indexes[-1]]]
        # Make sure to reverse the flipped rows!
        + [flipped_valid_rows[i] for i in indexes[:-1]][::-1]
    )


def translate(row):
    return "".join(["⬛" if x else "⬜" for x in row])


def visualize(grid):
    return "\n".join([translate(row) for row in grid])


def is_connected(grid):
    x, y = np.where(~grid)
    to_visit = [(x[0], y[0])]
    visited = set()

    while to_visit:
        x, y = to_visit.pop()
        for i, j in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if (
                0 <= i < GRID_LENGTH
                and 0 <= j < GRID_LENGTH
                and (i, j) not in visited
                and not grid[i, j]
            ):
                to_visit.append((i, j))
                visited.add((i, j))

    return len(visited) == np.sum(~grid)


# In[11]:


valid_grids_indexes = []
set_valid_rows = set([translate(row) for row in valid_rows])

while valid_middle_rows_indexes:
    indexes = valid_middle_rows_indexes.pop()

    (must_be_whites_indexes,) = np.where(
        (grid_from_indexes(*indexes)[:3].T == np.array([True, False, False])).all(
            axis=1
        )
    )
    (valid_rows_to_search_indexes,) = np.where(
        valid_rows[:, must_be_whites_indexes].all(axis=1)
    )

    for i in valid_rows_to_search_indexes:
        grid = grid_from_indexes(i, *indexes)
        if row_is_valid(grid, 0):
            if len([i, *indexes]) == GRID_LENGTH // 2 + 1:
                rot90_grid = np.rot90(grid)
                grid_is_connected = is_connected(grid)
                columns_are_valid_rows = all(
                    [
                        translate(row) in set_valid_rows
                        for row in rot90_grid[: GRID_LENGTH // 2 + 1]
                    ]
                )
                columns_are_valid = all(
                    [row_is_valid(rot90_grid, i) for i in range(GRID_LENGTH // 2 + 1)]
                )
                if grid_is_connected and columns_are_valid_rows and columns_are_valid:
                    valid_grids_indexes.append([i, *indexes])
            else:
                valid_middle_rows_indexes.append([i, *indexes])


# In[12]:


with open(
    f"{'interesting' if ONLY_FIND_INTERESTING else 'valid'}_{GRID_LENGTH}x{GRID_LENGTH}_grids.txt",
    "w",
) as f:
    msg = (
        f"There are {len(valid_grids_indexes)} {'interesting' if ONLY_FIND_INTERESTING else 'valid'} {GRID_LENGTH}x{GRID_LENGTH} grids.\n"
        f"- Assuming minimum word length of {MIN_WORD_LENGTH}.\n"
        f"- And that words must be half-checked, rounded up{'' if ROUND_UP_UNCHECKED else ' or down'}.\n"
    )

    if ONLY_FIND_INTERESTING:
        msg += f"- And where 'interesting' means each row must have at least {min_white_squares[GRID_LENGTH]} white squares.\n"

    print(msg, file=f)
    for valid_grid_index in valid_grids_indexes:
        s = visualize(grid_from_indexes(*valid_grid_index))
        print(s + "\n", file=f)


# In[ ]:




