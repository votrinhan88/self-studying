from pickletools import int4
import numpy as np
import time
import pygame

screen_size = 600
game_size = 100
time_step = 0.05

game_state = np.zeros((game_size, game_size), dtype = int)
for i in [(50, 52), (51, 52), (51, 53), (51, 54), (52, 53)]:
    game_state[i] = 1

def game_step(game_state:np.ndarray, game_size: int):
    # Prepare a 0-bordered game state
    sum_neighbors = np.zeros((game_size + 2, game_size + 2), dtype = np.int8)
    # Only count sum neighbors in "middle zone" (excluding edges)
    for row in np.arange(start = 1, stop = game_size + 1):
        for col in np.arange(start = 1, stop = game_size + 1):
            true_row, true_col = (row - 1, col - 1)
            sum_neighbors[row, col] =   np.sum(game_state[true_row - 1:true_row + 2, true_col - 1:true_col + 2])\
                                      - game_state[true_row, true_col]
    sum_neighbors = sum_neighbors[1:game_size+1, 1:game_size+1]
    
    # Calculate new state
    # Any live cell with fewer than two live neighbours dies, as if by underpopulation.
    # Any live cell with two or three live neighbours lives on to the next generation.
    # Any live cell with more than three live neighbours dies, as if by overpopulation.
    # Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
    new_game_state = np.zeros(game_state.shape, dtype = int)
    new_game_state[(game_state == 1) & (sum_neighbors < 2)] = 0
    new_game_state[(game_state == 1) & (sum_neighbors == 2) | (sum_neighbors == 3)] = 1
    new_game_state[(game_state == 1) & (sum_neighbors > 3)] = 0
    new_game_state[(game_state == 0) & (sum_neighbors == 3)] = 1
    return new_game_state

def get_rect_coords(screen_size: int, game_size: int, game_state:np.ndarray):
    edge_len = int(screen_size/game_size)
    xs, ys =  np.where(game_state == True)
    rect_xs = xs*edge_len
    rect_ys = ys*edge_len
    rects = list(zip(rect_xs, rect_ys, [edge_len]*len(xs), [edge_len]*len(ys)))
    return rects

pygame.init()
screen = pygame.display.set_mode(size = (screen_size, screen_size))
pygame.display.set_caption("Conway's Game of Life")

running = True
while(running):
    start_time = time.time()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill(color = (255, 255, 255))

    rects = get_rect_coords(screen_size, game_size, game_state)
    for rect in rects:
        pygame.draw.rect(surface = screen, color = (0, 0, 0), rect = rect)
    
    pygame.display.update()

    game_state = game_step(game_state, game_size)
    end_time = time.time()
    while end_time - start_time < time_step:
        end_time = time.time()