class CA:
    def __init__(self, rule, width, height, init_state):
        self.rule = rule
        self.width = width
        self.height = height
        self.init_state = init_state
        self.state = init_state
        self.rule = rule
        self.rule_dict = self.get_rule_dict(rule)
        self.state = init_state
        self.states = [init_state]

    def get_rule_dict(self, rule):
        rule_dict = {}
        for i in range(8):
            rule_dict[i] = rule % 2
            rule //= 2
        return rule_dict

    def get_neighborhood(self, i, j):
        neighborhood = 0
        for k in range(-1, 2):
            for l in range(-1, 2):
                neighborhood = neighborhood * 2 + self.state[(i + k) % self.width][(j + l) % self.height]
        return neighborhood

    def step(self):
        new_state = [[0 for _ in range(self.height)] for _ in range(self.width)]
        for i in range(self.width):
            for j in range(self.height):
                new_state[i][j] = self.rule_dict[self.get_neighborhood(i, j)]
        self.state = new_state
        self.states.append(new_state)

    def run(self, n):
        for _ in range(n):
            self.step()

