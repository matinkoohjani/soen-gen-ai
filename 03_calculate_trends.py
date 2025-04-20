import pandas as pd
import sys
import config


def calculate_yearly_trends(predictions_csv):
    try:
        df = pd.read_csv(predictions_csv)
    except FileNotFoundError:
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: Error loading prediction CSV {predictions_csv}: {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("No data available to calculate trends.")
        return
    if 'year' not in df.columns or 'prediction' not in df.columns:
        sys.exit(1)

    yearly_trends = df.groupby('year')['prediction'].mean() * 100
    yearly_trends = yearly_trends.rename("percentage_ai_generated")

    print("\n--- AI-Generated Code Percentage Trend ---")
    if not yearly_trends.empty:
        print(yearly_trends.to_string(float_format="%.2f%%"))
    else:
        print("No data available to calculate trends.")


def main():
    calculate_yearly_trends(config.PREDICTIONS_CSV)


if __name__ == "__main__":
    main()
