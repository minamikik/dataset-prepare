import time

class StopWatch:
    def __init__(self):
        self.start_time = time.time()
        self.lap_time = self.start_time
        self.total_time = self.start_time

    def lap(self):
        now = time.time()
        self.lap_time = now - self.total_time
        self.total_time = self.total_time + self.lap_time
        total_result = self.total_time - self.start_time
        return f'({round(self.lap_time, 2)}sec/{round(total_result, 2)}sec)'

    def total(self):
        now = time.time()
        self.lap_time = now - self.total_time
        self.total_time = self.total_time + self.lap_time
        total_result = self.total_time - self.start_time
        return f'({round(total_result, 2)}sec)'
