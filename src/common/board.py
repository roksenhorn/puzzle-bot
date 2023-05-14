import os
import yaml
import math


"""
(0, 0) ... (w, 0)
  .          .
  .          .
(0, h) ... (w, h)

Piece orientation:

 ___0___
|       |
3       1
|       |
 ---2---

"""

TOP = 0
RIGHT = 1
BOTTOM = 2
LEFT = 3

OPPOSITE = {
    TOP: BOTTOM, RIGHT: LEFT, BOTTOM: TOP, LEFT: RIGHT,
}

class Orientation(object):
    ZERO_POINTS_UP = 0
    ZERO_POINTS_RIGHT = 1
    ZERO_POINTS_DOWN = 2
    ZERO_POINTS_LEFT = 3

# dimensions for the puzzle you're solving
# ...I know... :(
# WIDTH = 40
# HEIGHT = 25
WIDTH = 10
HEIGHT = 10


class Board(object):
    @staticmethod
    def copy(board):
        # make a deep copy of each element in the board
        _board = [list(e) for e in board._board]
        return Board(board.width, board.height, _board=_board, _placed_piece_ids=set(board._placed_piece_ids))

    def __init__(self, width, height, _board=None, _placed_piece_ids=None) -> None:
        self.width = width
        self.height = height

        if _board is not None:
            self._board = _board
        else:
            self._board = []
            for y in range(self.height):
                self._board.append([])
                for x in range(self.width):
                    self._board[y].append(None)

        self._placed_piece_ids = _placed_piece_ids or set()

    def __repr__(self) -> str:
        num_digits = math.floor(math.log(self.width * self.height, 10)) + 3
        s = '\n  ' + '-' * num_digits * self.width + '\n'
        for y in range(self.height):
            for x in range(self.width):
                cell = self._board[y][x]
                if cell is None:
                    spaces = ' ' * (num_digits) + '-'
                    s += spaces
                else:
                    piece_id, _, orientation = cell
                    ori_str = '^>v<'[orientation]
                    s += '{:>{}}{}'.format(piece_id, num_digits, ori_str)
            s += '\n\n'
        return s

    def is_available(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self._board[y][x] is None

    def can_place(self, piece_id, fits, x, y, orientation):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False, f"Cannot place {piece_id} at ({x}, {y}) because it is outside the board"

        if piece_id in self._placed_piece_ids:
            return False, f"Cannot place {piece_id} at ({x}, {y}) because it has already been placed"

        if self._board[y][x] is not None:
            return False, f"Cannot place {piece_id} at ({x}, {y}) because it is already occupied by {self._board[y][x][0]}"

        sides_that_must_be_edges = self._sides_that_must_be_edges(x, y)
        # print(f"\t\tPlacing {piece_id} @ ori {orientation} @ ({x}, {y}) requires sides {sides_that_must_be_edges} to be edges")
        for side_i in range(4):
            expect_edge = side_i in sides_that_must_be_edges
            rotated_i = (side_i - orientation) % 4
            fits_i = fits[rotated_i]
            # print(f"\t\t    checking side {side_i}, which rotates to fits[{rotated_i}]")
            is_edge = len(fits_i) == 0
            if not is_edge and expect_edge:
                return False, f"Cannot place {piece_id} at ({x}, {y}) because side @ index {rotated_i} is not an edge piece"
            elif is_edge and not expect_edge:
                return False, f"Cannot place {piece_id} at ({x}, {y}) because side @ index {rotated_i} is an edge but it shouldn't be"

        # check connectivity of neighbors
        # if we have someone in a space next to us, let's make sure we connect properly to that space
        if x > 0 and self._board[y][x - 1] is not None:
            # check left neighbor has a fit
            neighbor_piece_id, _, _ = self._board[y][x - 1]
            rotated_i = (LEFT - orientation) % 4
            fits_i = [f[0] for f in fits[rotated_i]]
            if neighbor_piece_id not in fits_i:
                return False, f"Cannot place {piece_id} at ({x}, {y}) because it does not connect to the right neighbor {neighbor_piece_id} (only connects to {fits_i})"
        if x < self.width - 1 and self._board[y][x + 1] is not None:
            # check right neighbor has a fit
            neighbor_piece_id, _, _ = self._board[y][x + 1]
            rotated_i = (RIGHT - orientation) % 4
            fits_i = [f[0] for f in fits[rotated_i]]
            if neighbor_piece_id not in fits_i:
                return False, f"Cannot place {piece_id} at ({x}, {y}) because it does not connect to the right neighbor {neighbor_piece_id} (only connects to {fits_i})"
        if y > 0 and self._board[y - 1][x] is not None:
            # check top neighbor has a fit
            neighbor_piece_id, _, _ = self._board[y - 1][x]
            rotated_i = (TOP - orientation) % 4
            fits_i = [f[0] for f in fits[rotated_i]]
            if neighbor_piece_id not in fits_i:
                return False, f"Cannot place {piece_id} at ({x}, {y}) because it does not connect to the right neighbor {neighbor_piece_id} (only connects to {fits_i})"
        if y < self.height - 1 and self._board[y + 1][x] is not None:
            # check bottom neighbor has a fit
            neighbor_piece_id, _, _ = self._board[y + 1][x]
            rotated_i = (BOTTOM - orientation) % 4
            fits_i = [f[0] for f in fits[rotated_i]]
            if neighbor_piece_id not in fits_i:
                return False, f"Cannot place {piece_id} at ({x}, {y}) because it does not connect to the right neighbor {neighbor_piece_id} (only connects to {fits_i})"
        return True, None

    def place(self, piece_id, fits, x, y, orientation):
        self._board[y][x] = (piece_id, fits, orientation)
        self._placed_piece_ids.add(piece_id)

    @property
    def placed_count(self):
        return len(self._placed_piece_ids)

    def _sides_that_must_be_edges(self, x, y):
        sides = []
        if y == 0:
            sides.append(TOP)
        if y == self.height - 1:
            sides.append(BOTTOM)
        if x == 0:
            sides.append(LEFT)
        if x == self.width - 1:
            sides.append(RIGHT)
        return sides


def build(input_path, output_path):
    print("> Loading connectivity graph...")
    with open(os.path.join(input_path, 'connectivity.json'), 'r') as f:
        ps_raw = yaml.safe_load(f)

    ps = {}
    for piece_id, fits in ps_raw.items():
        piece_id = int(piece_id)
        ps[piece_id] = [[], [], [], []]
        for i in range(4):
            for p, s in fits[i]:
                ps[piece_id][i].append((p, s))

    board = Board(width=WIDTH, height=HEIGHT)

    corners = []
    edges = []
    edge_length = 2 * (WIDTH + HEIGHT) - 4
    for piece_id, neighbors in ps.items():
        edge_count = sum([1 for n in neighbors if len(n) == 0])
        if edge_count > 0:
            edges.append(piece_id)
            if edge_count > 1:
                corners.append(piece_id)

    print(f"Corners: {corners}, Edges: {len(edges)}")
    if len(corners) != 4:
        raise Exception(f"Expected 4 corners, got {len(corners)}")
    if len(edges) != edge_length:
        raise Exception(f"Expected {edge_length} pieces on the edge, got {len(edges)}")

    start_piece_id = corners[0]
    start_piece_fits = ps[start_piece_id]
    start_orientation = _orient_start_corner_to_top_left(start_piece_fits)

    x, y = (0, 0)
    board.place(start_piece_id, start_piece_fits, x, y, start_orientation)

    direction = RIGHT
    x += 1

    stack = [(board, start_piece_id, start_orientation, x, y, direction)]

    iteration = 0
    longest = 0
    while stack:
        board, start_piece_id, start_orientation, x, y, direction = stack.pop()
        # print(f"({x}, {y}) moving {direction}")
        if iteration % 1 == 0:
            print(f"Iteration {iteration}, longest: {longest}")
            print(board)

        if board.placed_count == WIDTH * HEIGHT:
            print(f"Placed 100 pieces in {iteration} iterations")
            break
        elif board.placed_count > longest:
            longest = board.placed_count

        index_of_neighbor_in_direction = (direction - start_orientation) % 4
        iteration += 1
        # print(f'Checking neighbors of {start_piece_id}[{index_of_neighbor_in_direction}] @ ori {start_orientation} --> checking: {ps[start_piece_id][index_of_neighbor_in_direction]}\n================')

        for neighbor_piece_id, neighbor_side_index in ps[start_piece_id][index_of_neighbor_in_direction]:
            neighbor_orientation = (OPPOSITE[direction] - neighbor_side_index) % 4
            # print(f'\t {neighbor_piece_id}[{neighbor_side_index}] @ ori {neighbor_orientation}')
            ok, err = board.can_place(piece_id=neighbor_piece_id, fits=ps[neighbor_piece_id], x=x, y=y, orientation=neighbor_orientation)
            if ok:
                # print(">>> OK! \t Adding to stack")
                next_board = Board.copy(board)
                next_board.place(neighbor_piece_id, ps[neighbor_piece_id], x, y, neighbor_orientation)
                next_direction = direction
                next_x = x + (1 if next_direction == RIGHT else -1 if next_direction == LEFT else 0)
                next_y = y + (1 if next_direction == BOTTOM else -1 if next_direction == TOP else 0)

                if not next_board.is_available(next_x, next_y):
                    # print(">>> ROTATING DIRECTION ~~~~~~~~~~")
                    # if we can't go further in this direction, time to turn
                    next_direction = (direction + 1) % 4
                    next_x = x + (1 if next_direction == RIGHT else -1 if next_direction == LEFT else 0)
                    next_y = y + (1 if next_direction == BOTTOM else -1 if next_direction == TOP else 0)

                stack.append((next_board, neighbor_piece_id, neighbor_orientation, next_x, next_y, next_direction))

    if board.placed_count == WIDTH * HEIGHT:
        print("Found solution after {iteration} iterations!")
        print(board)
    else:
        raise Exception(f"No solution found after {iteration} iterations, longest found: {longest}")

def _orient_start_corner_to_top_left(p):
    if len(p[0]) == 0 and len(p[1]) == 0:
        # ''|   --> |''
        return Orientation.ZERO_POINTS_LEFT
    elif len(p[1]) == 0 and len(p[2]) == 0:
        #  __|  -->  |''
        return Orientation.ZERO_POINTS_DOWN
    elif len(p[2]) == 0 and len(p[3]) == 0:
        #  |__  -->  |''
        return Orientation.ZERO_POINTS_RIGHT
    elif len(p[3]) == 0 and len(p[0]) == 0:
        # |''  -->  |''
        return Orientation.ZERO_POINTS_UP
    else:
        raise ValueError(f"Piece {p} is not a corner piece")