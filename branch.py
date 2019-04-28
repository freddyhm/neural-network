class Branch:
    def __init__(self, branch_id, weight, delta_weight = 0):
        self._branch_id = branch_id
        self._weight = weight
        self._delta_weight = delta_weight

    @property
    def branch_id(self):
        return self._branch_id

    @branch_id.setter
    def branch_id(self, branch_id):
        self._branch_id = branch_id

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        self._weight = weight

    @property
    def delta_weight(self):
        return self._delta_weight

    @delta_weight.setter
    def delta_weight(self, delta_weight):
        self._delta_weight = delta_weight