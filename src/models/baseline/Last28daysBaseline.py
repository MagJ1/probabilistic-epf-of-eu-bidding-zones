from models.baseline.WindowedBaseline import WindowedBaseline

class Last28DaysBaseline(WindowedBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, last_days=28, seasonal_months=0, **kwargs)
