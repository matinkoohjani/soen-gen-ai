import os
import subprocess
import sys
import config


def clone_repo(repo_name, target_dir):
    repo_url = f"https://github.com/{repo_name}.git"
    repo_folder_name = repo_name.replace('/', '_')
    target_path = os.path.join(target_dir, repo_folder_name)

    if os.path.exists(target_path):
        return target_path
    else:
        try:
            subprocess.run(
                ['git', 'clone', '--quiet', repo_url, target_path],
                check=True,
                capture_output=True,
                stderr=subprocess.PIPE
            )
            return target_path
        except Exception as e:
            print(f"ERROR: An unexpected error occurred cloning {repo_name}: {e}", file=sys.stderr)
            return None


def main():
    if not os.path.exists(config.REPOS_FILE):
        sys.exit(1)

    if not os.path.exists(config.CLONE_DIR):
        try:
            os.makedirs(config.CLONE_DIR)
        except OSError as e:
            sys.exit(1)

    try:
        with open(config.REPOS_FILE, 'r') as f:
            repo_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except Exception as e:
        sys.exit(1)

    failed_count = 0
    for repo_name in repo_names:
        result = clone_repo(repo_name, config.CLONE_DIR)
        if not result:
            failed_count += 1

    if failed_count > 0:
        print(f"Warning: Failed to clone {failed_count} repositories. Check stderr for details.", file=sys.stderr)


if __name__ == "__main__":
    main()
