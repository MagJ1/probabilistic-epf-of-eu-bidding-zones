from models.baseline.WindowedBaseline import WindowedBaseline

class Last7Days12MonthsBaseline(WindowedBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, last_days=7, seasonal_months=12, **kwargs)
