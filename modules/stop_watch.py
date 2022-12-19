import time
import sys

class StopWatch:
    def __init__(self):
        self.start_time = time.time()
        self.lap_time = self.start_time
        self.total_time = self.start_time
        print('StopWatch: Initialized')

    def lap(self, title):
        now = time.time()
        self.lap_time = now - self.total_time
        self.total_time = self.total_time + self.lap_time
        total_result = self.total_time - self.start_time
        print(f'{title}: {round(self.lap_time, 2)}sec/{round(total_result, 2)}sec')

    def total(self, title):
        now = time.time()
        self.lap_time = now - self.total_time
        self.total_time = self.total_time + self.lap_time
        total_result = self.total_time - self.start_time
        print(f'{title}: {round(total_result, 2)}sec')
