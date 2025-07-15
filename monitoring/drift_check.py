from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
import pandas as pd

def check_drift(current_data_path, reference_data_path):
    current_data = pd.read_csv(current_data_path)
    reference_data = pd.read_csv(reference_data_path)

    report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
    report.run(current_data=current_data, reference_data=reference_data)
    report.save_html("monitoring/drift_report.html")

    print("✅ Drift report generated at monitoring/drift_report.html")

if __name__ == "__main__":
    check_drift("data/processed/latest.csv", "data/processed/reference.csv")