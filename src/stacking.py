from packages import *


class Stacking:
    def __init__(self, data, local_path, ext_preds):
        self.data = data
        self.local_path = local_path
        self.ext_preds = ext_preds
