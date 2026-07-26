import random
import os

GRID_SIZE = 4


def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')


def spawn_fella(board):
    empty = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if board[r][c] == 0]
    if not empty:
        return False
    r, c = random.choice(empty)
    board[r][c] = random.choice([1, 1, 2])
    return True


def slide(line):
    new_line = [i for i in line if i != 0]
    merged = []
    i = 0
    while i < len(new_line):
        if i + 1 < len(new_line) and new_line[i] == new_line[i+1]:
            merged.append(new_line[i] + 1)
            i += 2
        else:
            merged.append(new_line[i])
            i += 1
    merged += [0] * (GRID_SIZE - len(merged))
    return merged


def move_board(board, direction):
    changed = False
    if direction == 'a':  # left
        for r in range(GRID_SIZE):
            original = board[r][:]
            board[r] = slide(board[r])
            if board[r] != original:
                changed = True
    elif direction == 'd':  # right
        for r in range(GRID_SIZE):
            original = board[r][:]
            board[r] = list(reversed(slide(list(reversed(board[r])))))
            if board[r] != original:
                changed = True
    elif direction == 'w':  # up
        for c in range(GRID_SIZE):
            col = slide([board[r][c] for r in range(GRID_SIZE)])
            for r in range(GRID_SIZE):
                if board[r][c] != col[r]:
                    changed = True
                board[r][c] = col[r]
    elif direction == 's':  # down
        for c in range(GRID_SIZE):
            col = list(reversed(slide([board[r][c] for r in range(GRID_SIZE)][::-1])))
            for r in range(GRID_SIZE):
                if board[r][c] != col[r]:
                    changed = True
                board[r][c] = col[r]
    return changed


def print_board(board):
    clear_screen()
    print("Italian Brainrot Merge Fella")
    print("Use WASD keys to move. Q to quit.")
    for row in board:
        print("|" + "|".join(f"{n:^5}" if n > 0 else "     " for n in row) + "|")
    print()


def game_over(board):
    if any(0 in row for row in board):
        return False
    # check moves available
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE-1):
            if board[r][c] == board[r][c+1]:
                return False
    for c in range(GRID_SIZE):
        for r in range(GRID_SIZE-1):
            if board[r][c] == board[r+1][c]:
                return False
    return True


def main():
    board = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    spawn_fella(board)
    spawn_fella(board)
    while True:
        print_board(board)
        if game_over(board):
            print("Game Over!")
            break
        move = input("Move (WASD): ").strip().lower()
        if move == 'q':
            break
        if move in ['w', 'a', 's', 'd']:
            if move_board(board, move):
                spawn_fella(board)


if __name__ == "__main__":
    main()
