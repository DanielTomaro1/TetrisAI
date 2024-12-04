import random
from grid import is_valid_position

TETROMINOES = {
    "I": [
        [[0, 0, 0, 0],
         [1, 1, 1, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 1, 0],
         [0, 0, 1, 0],
         [0, 0, 1, 0],
         [0, 0, 1, 0]]
    ],
    "O": [
        [[0, 2, 2, 0],
         [0, 2, 2, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
    ],
    "T": [
        [[0, 3, 0, 0],
         [3, 3, 3, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 3, 0, 0],
         [0, 3, 3, 0],
         [0, 3, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [3, 3, 3, 0],
         [0, 3, 0, 0],
         [0, 0, 0, 0]],
        [[0, 3, 0, 0],
         [3, 3, 0, 0],
         [0, 3, 0, 0],
         [0, 0, 0, 0]]
    ],
    "L": [
        [[0, 0, 4, 0],
         [4, 4, 4, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 4, 0, 0],
         [0, 4, 0, 0],
         [0, 4, 4, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [4, 4, 4, 0],
         [4, 0, 0, 0],
         [0, 0, 0, 0]],
        [[4, 4, 0, 0],
         [0, 4, 0, 0],
         [0, 4, 0, 0],
         [0, 0, 0, 0]]
    ],
    "J": [
        [[5, 0, 0, 0],
         [5, 5, 5, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 5, 5, 0],
         [0, 5, 0, 0],
         [0, 5, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [5, 5, 5, 0],
         [0, 0, 5, 0],
         [0, 0, 0, 0]],
        [[0, 5, 0, 0],
         [0, 5, 0, 0],
         [5, 5, 0, 0],
         [0, 0, 0, 0]]
    ],
    "S": [
        [[0, 6, 6, 0],
         [6, 6, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 6, 0, 0],
         [0, 6, 6, 0],
         [0, 0, 6, 0],
         [0, 0, 0, 0]]
    ],
    "Z": [
        [[7, 7, 0, 0],
         [0, 7, 7, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 7, 0],
         [0, 7, 7, 0],
         [0, 7, 0, 0],
         [0, 0, 0, 0]]
    ]
}

def spawn_tetromino():
    tetromino_key = random.choice(list(TETROMINOES.keys()))
    return tetromino_key, TETROMINOES[tetromino_key][0], 0

def rotate_tetromino(grid, tetromino_key, rotation_index, position):
    rotations = TETROMINOES[tetromino_key]
    new_rotation_index = (rotation_index + 1) % len(rotations)
    new_tetromino = rotations[new_rotation_index]
    
    if is_valid_position(grid, new_tetromino, position):
        return new_tetromino, new_rotation_index
    return rotations[rotation_index], rotation_index