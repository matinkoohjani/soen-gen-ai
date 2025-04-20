import os
import git
import pandas as pd
from datetime import datetime
import sys
import config
from nn_predictor import ProgramClassifierWrapper


def get_commit_for_year(repo, year):
    target_date_str = f"{year}-01-01T00:00:00Z"
    try:
        commits = list(repo.iter_commits('HEAD', max_count=1, until=target_date_str))
        if commits:
            return commits[0]
        else:
            all_commits = list(repo.iter_commits('--all', reverse=True))
            return all_commits[0] if all_commits else None
    except git.GitCommandError as e:
        print(f"ERROR: Error finding commit for year {year} in {repo.working_dir}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error getting commit for year {year}: {e}", file=sys.stderr)
        return None


def analyze_repo_yearly(repo_path, repo_name, start_year, end_year):
    predictions = []
    repo = git.Repo(repo_path)
    original_branch = None
    original_commit = None
    try:
        original_branch = repo.active_branch
    except TypeError:
        original_commit = repo.head.commit

    for year in range(start_year, end_year):
        commit_to_checkout = get_commit_for_year(repo, year + 1)

        if not commit_to_checkout:
            continue

        temp_branch_name = f"analysis_{year}_{commit_to_checkout.hexsha[:8]}"
        checkout_successful = False
        repo.git.checkout(commit_to_checkout.hexsha, b=temp_branch_name)
        checkout_successful = True

        for root, _, files in os.walk(repo_path):
            if '.git' in root.split(os.sep):
                continue

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, repo_path)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        prediction_tensor = ProgramClassifierWrapper.predict(content)
                        prediction_class = prediction_tensor.item()

                        predictions.append({
                            "project": repo_name,
                            "file_path": relative_path,
                            "content_preview": content[:200] + ("..." if len(content) > 200 else ""),
                            "year": year,
                            "commit_sha": commit_to_checkout.hexsha,
                            "commit_date": commit_to_checkout.committed_datetime.strftime('%Y-%m-%d'),
                            "prediction": prediction_class
                        })
                    except FileNotFoundError:
                        pass
                    except Exception as e:
                        print(f"ERROR: Error processing file {file_path}: {e}", file=sys.stderr)


        try:
            if original_branch:
                repo.git.checkout(str(original_branch))
            elif original_commit:
                repo.git.checkout(original_commit.hexsha)
            else:
                try:
                    repo.git.checkout('master')
                except git.GitCommandError:
                    try:
                        repo.git.checkout('main')
                    except git.GitCommandError:
                        print(f"Warning: Could not checkout original/master/main in {repo_name}.",
                              file=sys.stderr)

            if checkout_successful:
                try:
                    repo.git.branch('-D', temp_branch_name)
                except git.GitCommandError:
                    pass

        except Exception as e:
            print(f"ERROR: Unexpected error during cleanup for {repo_name}, year {year}: {e}", file=sys.stderr)

    return predictions


def main():
    try:
        with open(config.REPOS_FILE, 'r') as f:
            repo_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except Exception as e:
        print(f"CRITICAL ERROR: Error reading repository list file {config.REPOS_FILE}: {e}", file=sys.stderr)
        sys.exit(1)

    all_repo_predictions = []
    for repo_name in repo_names:
        repo_folder_name = repo_name.replace('/', '_')
        repo_path = os.path.join(config.CLONE_DIR, repo_folder_name)
        if os.path.isdir(repo_path):
            repo_predictions = analyze_repo_yearly(repo_path, repo_name, config.START_YEAR, config.END_YEAR)
            all_repo_predictions.extend(repo_predictions)
        else:
            pass

    if not all_repo_predictions:
        df_predictions = pd.DataFrame(columns=[
            "project", "year", "file_path", "prediction",
            "commit_sha", "commit_date", "content_preview"
        ])
    else:
        df_predictions = pd.DataFrame(all_repo_predictions)
        df_predictions = df_predictions[[
            "project", "year", "file_path", "prediction",
            "commit_sha", "commit_date", "content_preview"
        ]]

    df_predictions.to_csv(config.PREDICTIONS_CSV, index=False)


if __name__ == "__main__":
    main()
