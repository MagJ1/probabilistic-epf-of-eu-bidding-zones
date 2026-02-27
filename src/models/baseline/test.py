from models.baseline.WindowedBaseline import WindowedBaseline
from models.baseline.BootstrappedBaseline import BootstrappedBaseline
from pathlib import Path
from models.baseline.Last28daysBaseline import Last28DaysBaseline
from models.baseline.Last7Days12MonthsBaseline import Last7Days12MonthsBaseline

def main():
    train_path = Path("raw_data/Germany_time_zone/all/train_with_all.csv")
    test_path = Path("raw_data/Germany_time_zone/all/test_with_all.csv")
    # train_path = Path("raw_data/transfer_learning/FR_training_data_with_calendar.csv")
    # test_path = Path("raw_data/transfer_learning/FR_test_data_with_calendar.csv")

    last_7_days_12_months_baseline = Last7Days12MonthsBaseline(
        path_to_train_csv = train_path,
        path_to_test_csv = test_path
        )
    
    last_28_days_baseline = Last28DaysBaseline(
        path_to_train_csv = train_path,
        path_to_test_csv = test_path
        )
    
    # df_day = last_7_days_12_months_baseline.evaluate_day_ahead(
    #     anchor_hour=0,
    #     origin_stride_hours=24,   
    #     save_to="outputs/baselines/last7_days_12_months_crps_forecast/crps_per_origin.csv"
    # )

    df_day = last_28_days_baseline.evaluate_day_ahead(
        anchor_hour=0,
        origin_stride_hours=24,   
        save_to="outputs/baselines/last28_day_crps_forecast/crps_per_origin.csv"

    )

    
    print(last_7_days_12_months_baseline.evaluate_dataset())
    print(last_28_days_baseline.evaluate_dataset())

    # point_forecast_errors_bootstrapped_real_price = BootstrappedBaseline(path_to_train_csv = train_path, path_to_test_csv = test_path, point_forecast_lag=24, n_samples=100)
    # print(point_forecast_errors_bootstrapped_real_price.evaluate_dataset())

    # point_forecast_errors_bootstrapped_synth_pric = BootstrappedBaseline(path_to_train_csv = train_path, path_to_test_csv = test_path, point_forecast_lag=24, n_samples=100, ref_col="y", cols_of_interest=["ds","y","synthetic_price"])
    # print(point_forecast_errors_bootstrapped_synth_pric.evaluate_dataset())



if __name__ == "__main__":
    main()