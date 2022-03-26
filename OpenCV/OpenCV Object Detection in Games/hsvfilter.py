class HsvFilter:
    def __init__(self,
                 h_min = None,
                 s_min = None,
                 v_min = None,
                 h_max = None,
                 s_max = None,
                 v_max = None,
                 s_add = None,
                 s_sub = None,
                 v_add = None,
                 v_sub = None):
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