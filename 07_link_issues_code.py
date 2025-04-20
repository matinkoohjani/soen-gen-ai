import lucene
import os
import pandas as pd
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser, ParseException
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search.phrase import PhraseQuery
import config
import re


try:
    if not lucene.getVMEnv():
        lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        print("Lucene VM initialized successfully for linking.")
except Exception as e:
    print(f"Failed to initialize Lucene VM: {e}")
    exit()


def search_index(searcher, analyzer, field, query_text, max_hits=10):
    escaped_query_text = QueryParser.escape(query_text)

    if not escaped_query_text.strip():
        return []

    parser = QueryParser(field, analyzer)
    try:
        if ' ' in escaped_query_text or '/' in escaped_query_text or '\n' in escaped_query_text:
            try:
                query = parser.parse(f'"{escaped_query_text}"')
            except ParseException:
                query = parser.parse(escaped_query_text)
        else:
            query = parser.parse(escaped_query_text)

    except ParseException as e:
        print(f"Error parsing query '{escaped_query_text}' for field '{field}': {e}")
        return []
    except Exception as e:
        print(f"Unexpected error during query parsing or creation for '{escaped_query_text}': {e}")
        return []

    try:
        score_docs = searcher.search(query, max_hits).scoreDocs
        hits = []
        for score_doc in score_docs:
            doc = searcher.doc(score_doc.doc)
            hits.append(doc)
        return hits
    except Exception as e:
        print(f"Error during search execution for query '{query_text}': {e}")
        return []


def find_links(issues_csv, predictions_csv, code_index_dir, issue_index_dir):
    try:
        df_issues = pd.read_csv(issues_csv)
        df_issues = df_issues.fillna('')
        print(f"Loaded {len(df_issues)} issues from {issues_csv}")

        df_predictions = pd.read_csv(predictions_csv)
        df_predictions = df_predictions.fillna('')
        print(f"Loaded {len(df_predictions)} predictions from {predictions_csv}")

        target_year = config.END_YEAR - 1
        df_aigc_code = df_predictions[
            (df_predictions['year'] == target_year) & (df_predictions['prediction'] == 1)
            ].copy()
        print(f"Filtered to {len(df_aigc_code)} AI-generated code files from year {target_year}.")

    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return []

    try:
        code_directory = FSDirectory.open(Paths.get(code_index_dir))
        code_reader = DirectoryReader.open(code_directory)
        code_searcher = IndexSearcher(code_reader)

        issue_directory = FSDirectory.open(Paths.get(issue_index_dir))
        issue_reader = DirectoryReader.open(issue_directory)
        issue_searcher = IndexSearcher(issue_reader)

        analyzer = StandardAnalyzer()
    except Exception as e:
        print(f"Error opening Lucene indices: {e}")
        return []

    found_links = []
    processed_links = set()

    print("Starting Condition 1 search: Issue code snippet -> Code Index")
    issues_with_snippets = df_issues[df_issues['code_snippet_extracted'].str.strip().astype(bool)]
    print(f"Found {len(issues_with_snippets)} issues with extracted code snippets.")

    for index, issue_row in issues_with_snippets.iterrows():
        issue_project = issue_row['project']
        issue_url = issue_row['url']
        snippet = issue_row['code_snippet_extracted']

        if len(snippet) < 10:
            continue

        print(f"Searching code index for snippet from issue {issue_url}...")
        code_hits = search_index(code_searcher, analyzer, "content", snippet, max_hits=5)

        for code_hit_doc in code_hits:
            code_project = code_hit_doc.get("project")
            code_path = code_hit_doc.get("path")
            code_commit = code_hit_doc.get("commit_sha")
            code_content_preview = code_hit_doc.get("content")[:200]

            if code_project == issue_project:
                link_tuple = (issue_url, code_path)
                if link_tuple not in processed_links:
                    print(
                        f"  Found Link (Issue Snippet -> Code): {issue_url} <-> {code_project}/{code_path} (Commit: {code_commit})")
                    found_links.append({
                        "project": issue_project,
                        "issue_url": issue_url,
                        "issue_title": issue_row['title'],
                        "code_file_path": code_path,
                        "code_commit_sha": code_commit,
                        "code_content_preview": code_content_preview,
                        "matched_snippet_from_issue": snippet[:200] + "...",
                        "match_type": "snippet_in_code"
                    })
                    processed_links.add(link_tuple)

    print("Starting Condition 2 search: Code file path -> Issue Index")
    for index, code_row in df_aigc_code.iterrows():
        code_project = code_row['project']
        code_path = code_row['file_path']
        code_commit = code_row['commit_sha']
        if not code_path: continue

        print(f"Searching issue index for path '{code_path}' from project {code_project}...")
        issue_hits = search_index(issue_searcher, analyzer, "content", code_path, max_hits=5)

        for issue_hit_doc in issue_hits:
            issue_project = issue_hit_doc.get("project")
            issue_url = issue_hit_doc.get("url")
            issue_content_preview = issue_hit_doc.get("content")[:200]

            if issue_project == code_project:
                link_tuple = (issue_url, code_path)
                if link_tuple not in processed_links:
                    issue_title = df_issues[df_issues['url'] == issue_url]['title'].iloc[0] if any(
                        df_issues['url'] == issue_url) else "Title not found"

                    print(
                        f"  Found Link (Code Path -> Issue): {issue_url} <-> {code_project}/{code_path} (Commit: {code_commit})")
                    found_links.append({
                        "project": code_project,
                        "issue_url": issue_url,
                        "issue_title": issue_title,
                        "code_file_path": code_path,
                        "code_commit_sha": code_commit,
                        "code_content_preview": df_predictions[
                            (df_predictions['project'] == code_project) &
                            (df_predictions['file_path'] == code_path) &
                            (df_predictions['commit_sha'] == code_commit)
                            ]['content_preview'].iloc[0] if not df_predictions[
                            (df_predictions['project'] == code_project) &
                            (df_predictions['file_path'] == code_path) &
                            (df_predictions['commit_sha'] == code_commit)
                            ].empty else "",
                        "matched_path_in_issue": issue_content_preview + "...",
                        "match_type": "path_in_issue"
                    })
                    processed_links.add(link_tuple)

    code_reader.close()
    issue_reader.close()

    print(f"Found {len(found_links)} potential links between issues and AI-generated code.")
    return found_links


def main():
    links = find_links(config.ISSUES_CSV, config.PREDICTIONS_CSV, config.CODE_INDEX_DIR, config.ISSUE_INDEX_DIR)

    if links:
        df_links = pd.DataFrame(links)
        df_links = df_links[[
            "project", "match_type", "issue_url", "issue_title",
            "code_file_path", "code_commit_sha", "code_content_preview",
            "matched_snippet_from_issue", "matched_path_in_issue"
        ]]
        try:
            df_links.to_csv(config.LINKS_CSV, index=False)
            print(f"Links saved to {config.LINKS_CSV}")
        except Exception as e:
            print(f"Failed to save links CSV: {e}")


if __name__ == "__main__":
    if not lucene.getVMEnv():
        print("Initializing Lucene VM for linking...")
        try:
            lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        except Exception as e:
            print(f"Failed to initialize Lucene VM: {e}. Exiting.")
            exit()
    main()
