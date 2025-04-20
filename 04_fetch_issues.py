import os
import pandas as pd
from github import Github
import re
import time
import config

CODE_BLOCK_REGEX = re.compile(r'```(?:[a-zA-Z0-9]+)?\n(.*?)```', re.DOTALL)


def extract_code_snippets(text):
    if not text:
        return ""
    snippets = CODE_BLOCK_REGEX.findall(text)
    return "\n---\n".join(snippets)


def fetch_issues_for_repo(g, repo_name):
    issues_data = []
    try:
        repo = g.get_repo(repo_name)
        print(f"Fetching issues for {repo_name}...")
        issues = repo.get_issues(state='all')
        count = 0
        for issue in issues:
            if count % 100 == 0 and count > 0:
                print(f"  ... fetched {count} issues for {repo_name}")
                rate_limit = g.get_rate_limit()
                remaining = rate_limit.core.remaining
                print(f"  GitHub API Rate Limit Remaining: {remaining}")
                if remaining < 50:
                    time.sleep(60)

            body = issue.body if issue.body else ""
            code_snippets = extract_code_snippets(body)

            issues_data.append({
                "project": repo_name,
                "issue_number": issue.number,
                "title": issue.title,
                "url": issue.html_url,
                "state": issue.state,
                "creator": issue.user.login,
                "created_at": issue.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                "closed_at": issue.closed_at.strftime('%Y-%m-%d %H:%M:%S') if issue.closed_at else None,
                "body": body,
                "code_snippet_extracted": code_snippets,
                "comments_count": issue.comments
            })
            count += 1
        print(f"Fetched a total of {count} issues for {repo_name}.")

    except Exception as e:
        print(f"An error occurred fetching issues for {repo_name}: {e}")

    return issues_data


def main():
    github_token = config.get_github_token()
    g = Github(github_token)

    with open(config.REPOS_FILE, 'r') as f:
        repo_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    all_issues_data = []
    for repo_name in repo_names:
        try:
            repo_issues = fetch_issues_for_repo(g, repo_name)
            all_issues_data.extend(repo_issues)
        except Exception as e:
            print(f"Continuing after error in processing {repo_name}: {e}")

    df_issues = pd.DataFrame(all_issues_data)

    # Reorder columns
    df_issues = df_issues[[
        "project", "issue_number", "url", "state", "title", "creator",
        "created_at", "closed_at", "comments_count", "body", "code_snippet_extracted"
    ]]

    df_issues.to_csv(config.ISSUES_CSV, index=False)


if __name__ == "__main__":
    main()
