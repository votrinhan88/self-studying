class Team():
    def __init__(self, name:str, pos:int, win:int, draw:int, loss:int, gf:int, ga:int, form = [str]):
        self.name = name
        self.pos  = pos
        self.win  = win 
        self.draw = draw
        self.loss = loss
        self.gf   = gf  
        self.ga   = ga  
        self.form = form