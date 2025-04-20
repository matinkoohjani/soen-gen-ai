import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import config


def sample_links(links_csv, target_size, output_csv):
    df = pd.read_csv(links_csv)

    if df.empty:
        print("Links data is empty. Cannot sample.")
        return

    n_total = len(df)
    n_sample = min(target_size, n_total)

    if n_sample == 0:
        print("No links available to sample.")
        return

    print(f"Target sample size: {n_sample} (out of {n_total} total links)")

    strata_column = 'project'
    sampled_df = None

    if strata_column in df.columns and df[strata_column].nunique() > 1:
        min_samples_per_stratum = 1
        value_counts = df[strata_column].value_counts()
        valid_strata = value_counts[value_counts >= min_samples_per_stratum].index
        df_stratifiable = df[df[strata_column].isin(valid_strata)]

        if len(df_stratifiable) >= n_sample and df_stratifiable[strata_column].nunique() > 1:
            print(f"Attempting stratified sampling by '{strata_column}'...")
            try:
                split = StratifiedShuffleSplit(n_splits=1, test_size=n_sample / len(df_stratifiable), random_state=42)
                X = df_stratifiable.index.to_numpy().reshape(-1, 1)
                y = df_stratifiable[strata_column].to_numpy()

                for _, test_index in split.split(X, y):
                    sampled_indices = df_stratifiable.iloc[test_index].index
                    sampled_df = df.loc[sampled_indices]
                    break

                if sampled_df is not None:
                    print(f"Successfully performed stratified sampling, got {len(sampled_df)} samples.")
                    if len(sampled_df) > n_sample:
                        sampled_df = sampled_df.sample(n=n_sample, random_state=42)

            except Exception as e:
                print(f"Stratified sampling failed: {e}.")
                sampled_df = None

    if sampled_df is None:
        print("Performing simple random sampling...")
        sampled_df = df.sample(n=n_sample, random_state=42)
        print(f"Selected {len(sampled_df)} samples using simple random sampling.")

    sampled_df['issue_related'] = ''
    sampled_df[
        'issue_type'] = ''
    output_columns = [
        "project",
        "issue_url",
        "code_file_path",
        "code_commit_sha",
        "match_type",
        "issue_related",
        "issue_type",
        "issue_title",
        "code_content_preview",
        "matched_snippet_from_issue",
        "matched_path_in_issue"
    ]
    for col in output_columns:
        if col not in sampled_df.columns:
            sampled_df[col] = ''
    sampled_df = sampled_df[output_columns]

    try:
        sampled_df.to_csv(output_csv, index=False)
        print(f"Sampled links saved to {output_csv}")
    except Exception as e:
        print(f"Failed to save sampled links CSV: {e}")


def main():
    sample_links(config.LINKS_CSV, config.SAMPLE_SIZE, config.SAMPLED_LINKS_CSV)


if __name__ == "__main__":
    main()
