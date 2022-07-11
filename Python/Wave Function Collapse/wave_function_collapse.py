import numpy as np
from typing import List, Tuple

def divisors(x):
    # By alain-t @ https://stackoverflow.com/questions/70635382
    factors = {1}
    maxP  = int(x**0.5)
    p, inc = 2, 1
    while p <= maxP:
        while x % p==0:
            factors.update([f*p for f in factors])
            x //= p
            maxP = int(x**0.5)
        p, inc = p + inc, 2
    if x > 1:
        factors.update([f*x for f in factors])
    return sorted(factors) 

class Cell():
    def __init__(self, num_options:int, coords:Tuple[int, int]):
        self.num_options = num_options
        self.coords = coords

        self.collapsed:bool = False
        self.entropy:np.ndarray = None
        self.options:np.ndarray = np.arange(self.num_options) + 1
        self.value:np.ndarray = -1

        self.allies:List[Cell] = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.coords} (Ɛ = {self.entropy})"

    def find_unique_allies(self):
        coords = np.array([cell.coords for cell in self.allies])
        idx_uniques = np.unique(coords, axis = 0, return_index = True)[1]
        self.allies = self.allies[idx_uniques]

    def collapse(self):
        if self.options.size == 1:
            self.value = self.options[0]
        elif self.options.size > 1:
            self.value = np.random.permutation(self.options)[0]
        self.collapsed = True
        
    def propagate(self):
        for cell in self.allies:
            cell.options = cell.options[cell.options != self.value]

    def compute_entropy(self):
        self.entropy = self.options.size

class SudokuTable():
    def __init__(self, size:int = 9, verbose:bool = False):
        self.size = size
        self.verbose = verbose

        self.make_table()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} of {self.size}x{self.size}" + '\n' + np.array_str(self.value)

    def make_table(self):
        self.cells:np.ndarray = np.array([[Cell(self.size, (row, col))
                                            for col in range(self.size)]
                                                for row in range(self.size)], dtype = Cell)
        
        self.value:np.ndarray = -np.ones([self.size, self.size], dtype = int)
        self.collapsed:np.ndarray = np.zeros([self.size, self.size], dtype = bool)
        self.stuck:bool = False

        # Make cells in the same row, column, or 3x3 grid allies
        stride = round(np.sqrt(self.size))
        for row in np.arange(self.size):
            self.make_allies(self.cells[row, :])
        for col in np.arange(self.size):
            self.make_allies(self.cells[:, col])
        for row in np.arange(start = 0, stop = self.size, step = stride):
            for col in np.arange(start = 0, stop = self.size, step = stride):
                self.make_allies(self.cells[row:row+stride, col:col+stride])

    def make_allies(self, cells:np.ndarray):
        for cell in cells.flatten():
            cell.allies = np.concatenate([cell.allies, cells[cells != cell]])
            cell.find_unique_allies()

    def collapse(self):
        while (self.collapsed.prod() == False):
            if self.stuck == False:
                # Continue finding a solution
                self.iterate()
            elif self.stuck == True:
                # Start over when stuck. To-do: backtracking(?)
                if self.verbose == True:
                    print('Got stuck. Starting over...')
                self.make_table()
                self.collapse()

    def iterate(self):
        # Check non-collapsed cells
        coord_row, coord_col = np.asarray(self.collapsed == False).nonzero()
        
        min_entropy = np.array(float('inf'))
        clps_coords = None
        # Loop through non-collapsed cells to find new collapsible cells
        for coords in zip(coord_row, coord_col):
            ## Calculate entropy
            cell:Cell = self.cells[coords]
            cell.compute_entropy()
            # Memorize cells with lowest entropy
            if cell.entropy < min_entropy:
                min_entropy = cell.entropy
                clps_coords = np.expand_dims(coords, axis = 0)
            elif cell.entropy == min_entropy:
                clps_coords = np.concatenate([clps_coords, np.expand_dims(coords, axis = 0)], axis = 0)
        
        if min_entropy == 0:
            # The collapse has no solution
            self.stuck = True
        elif min_entropy > 0:
            # Collapse cell with lowest entropy (shuffle if many)
            clps_coords = tuple(np.random.permutation(clps_coords)[0])
            self.collapse_single(clps_coords)  

    def collapse_single(self, clps_coords:np.ndarray):
        clps_cell:Cell = self.cells[clps_coords]

        clps_cell.collapse()
        self.value[clps_coords]     = clps_cell.value
        self.collapsed[clps_coords] = clps_cell.collapsed

        clps_cell.propagate()

class MineCell(Cell):
    def __init__(self, coords:Tuple[int, int]):
        self.NUM_MINES = 9

        self.num_options = self.NUM_MINES + 1
        self.coords = coords

        self.collapsed:bool = False
        self.entropy:np.ndarray = None
        self.options:np.ndarray = np.arange(self.num_options)
        self.value:np.ndarray = -1

        self.allies:List[MineCell] = []
    
    def collapse(self):
        # if self.options.size == 1:
        #     self.value = self.options[0]
        # elif self.options.size > 1:
        #     self.value = np.random.permutation(self.options)[0]
        # self.collapsed = True
        pass
        
    def propagate(self):
        # for cell in self.allies:
        #     cell.options = cell.options[cell.options != self.value]
        pass



class MineField():
    def __init__(self, size:Tuple[int, int], mine_ratio:float):
        self.size = size
        self.mine_ratio = mine_ratio
        
        self._mines:np.ndarray = None
        self._values:np.ndarray = None

        self.MINE_VALUE = 9

        self._make_field()
    
    def _make_field(self):
        self._mines = (np.random.rand(self.size[0], self.size[1]) <= self.mine_ratio).astype(int)
        mines_padded:np.ndarray = np.pad(self._mines, pad_width = 1, constant_values = 0)    

        # kernel = np.array([[1, 1, 1],[1, 0, 1],[1, 1, 1]])
        self._num_mines = mines_padded[0:-2, 0:-2] + mines_padded[0:-2, 1:-1] + mines_padded[0:-2, 2:] + \
                          mines_padded[1:-1, 0:-2] +                        0 + mines_padded[1:-1, 2:] + \
                          mines_padded[2:  , 0:-2] + mines_padded[2:  , 1:-1] + mines_padded[2:  , 2:]
        self._num_mines[self._mines == 1] = self.MINE_VALUE

    def print_field(self):
        field = self._num_mines.astype(np.dtype('U25'))
        field[field == str(self.MINE_VALUE)] = 'X'
        print(field)

class MineSweeper():
    def __init__(self, field:MineField, verbose:bool = False):
        self.field = field
        self.verbose = verbose

        self.size = self.field.size
        self.make_table()

    def make_table(self):
        self.cells:np.ndarray = np.array([[MineCell((row, col))
                                            for col in range(self.size[1])]
                                                for row in range(self.size[0])], dtype = MineCell)
        
        self.value:np.ndarray = -np.ones([self.size[0], self.size[1]], dtype = int)
        self.collapsed:np.ndarray = np.zeros([self.size[0], self.size[1]], dtype = bool)
        self.stuck:bool = False

        # Make cells in the 1-cell radius allies
        for row in np.arange(self.size[0]):
            for col in np.arange(self.size[1]):
                xs = np.arange(np.maximum(0, row-1), np.minimum(self.size[0], row+1)+1)
                ys = np.arange(np.maximum(0, col-1), np.minimum(self.size[1], col+1)+1)
                

                # print()

    def make_allies(self, cell:np.ndarray):
        cell.allies = np.concatenate([cell.allies, cell.flatten()])

if __name__ == '__main__':
    # Task 1: Generate a sudoku table
    # t = SudokuTable(9)
    # t.collapse()
    # print(t)

    # Task 2:
    m = MineField((10, 10), 0.3)
    m.print_field()
    ms = MineSweeper(field = m)
    # ms.collapse()
    print()
