class TimeDistributed(MaskedLayer):

    def __init__(self, layer):
        self.layer = layer

    @property
    def input_ndim(self):
        return self.layer.input_ndim + 1

    def build(self):
        input_shape = self.input_shape
        