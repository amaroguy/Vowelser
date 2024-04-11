from collections import deque
import numpy as np

class MedianStreamer:
    def __init__(self, maxlen):
        self.deque = deque(maxlen=maxlen)

    def insert(self,val):
        self.deque.append(val)

        return np.median(list(self.deque))
