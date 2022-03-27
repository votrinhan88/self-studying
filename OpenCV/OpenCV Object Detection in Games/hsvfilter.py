class HsvFilter:
    def __init__(self,
                 h_min = 0,
                 s_min = 0,
                 v_min = 0,
                 h_max = 179,
                 s_max = 255,
                 v_max = 255,
                 s_add = 0,
                 s_sub = 0,
                 v_add = 0,
                 v_sub = 0):
        self.h_min = h_min
        self.s_min = s_min
        self.v_min = v_min
        self.h_max = h_max
        self.s_max = s_max
        self.v_max = v_max
        self.s_add = s_add
        self.s_sub = s_sub
        self.v_add = v_add
        self.v_sub = v_sub
        pass