import threading
import numpy as np


class FakeBackend:
    def __init__(self, delay=0.0, slow_word=None):
        self.spoken, self.stops, self.delay, self.slow_word = [], 0, delay, slow_word
        self._evt = threading.Event()

    def speak(self, text):
        self.spoken.append(text)
        if self.delay and (self.slow_word is None or self.slow_word in text):
            self._evt.wait(self.delay)
        self._evt.clear()

    def stop(self):
        self.stops += 1
        self._evt.set()


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt
