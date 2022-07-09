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
    def __init__(self, size:int = 9):
        self.size = size

        self.cells:np.ndarray = np.array([[Cell(self.size, (row, col))
                                            for col in range(self.size)]
                                                for row in range(self.size)], dtype = Cell)
        
        self.value:np.ndarray = -np.ones([self.size, self.size], dtype = int)
        self.collapsed:np.ndarray = np.zeros([self.size, self.size], dtype = bool)

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
        # Exists cells that haven't collapsed
        while self.collapsed.prod() == False:
            self.iterate()

    def iterate(self):
        # Check collapsed
        coord_row, coord_col = np.asarray(self.collapsed == False).nonzero()
        min_entropy = np.array(float('inf'))

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

        # Collapse cell with lowest entropy (shuffle if many)
        clps_coords = tuple(np.random.permutation(clps_coords)[0])
        self.collapse_single(clps_coords)

    def collapse_single(self, clps_coords:np.ndarray):
        clps_cell:Cell = self.cells[clps_coords]

        clps_cell.collapse()
        self.value[clps_coords]     = clps_cell.value
        self.collapsed[clps_coords] = clps_cell.collapsed

        clps_cell.propagate()

t = SudokuTable(9)
t.collapse()
print(t.value)