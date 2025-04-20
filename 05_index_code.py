import lucene
import os
import git
import pandas as pd
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, StringField, TextField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import FSDirectory
import config


try:
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
except Exception as e:
    print(f"Failed to initialize Lucene VM: {e}")
    print("Ensure JAVA_HOME is set and PyLucene is installed correctly.")
    exit()


def create_code_index(predictions_csv, index_dir):
    df = pd.read_csv(predictions_csv)

    target_year = config.END_YEAR - 1
    df_filtered = df[(df['year'] == target_year) & (df['prediction'] == 1)].copy()

    if df_filtered.empty:
        return

    retrieved_content = []
    cloned_repo_paths = {
        repo_name.replace('/', '_'): os.path.join(config.CLONE_DIR, repo_name.replace('/', '_'))
        for repo_name in df_filtered['project'].unique()
    }

    df_filtered = df_filtered.fillna('')

    for index, row in df_filtered.iterrows():
        repo_folder_name = row['project'].replace('/', '_')
        repo_path = cloned_repo_paths.get(repo_folder_name)
        file_rel_path = row['file_path']
        commit_sha = row['commit_sha']

        if not repo_path or not os.path.exists(repo_path) or not commit_sha or not file_rel_path:
            retrieved_content.append(None)
            continue

        repo = git.Repo(repo_path)
        full_content = repo.git.show(f'{commit_sha}:{file_rel_path}')
        retrieved_content.append(full_content)

    df_filtered['full_content'] = retrieved_content
    df_filtered = df_filtered.dropna(subset=['full_content'])

    if not os.path.exists(index_dir):
        os.makedirs(index_dir)

    directory = FSDirectory.open(Paths.get(index_dir))
    analyzer = StandardAnalyzer()
    config_writer = IndexWriterConfig(analyzer)
    config_writer.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    writer = IndexWriter(directory, config_writer)


    for index, row in df_filtered.iterrows():
        doc = Document()
        doc.add(StringField("project", row['project'], Field.Store.YES))
        doc.add(StringField("path", row['file_path'], Field.Store.YES))
        doc.add(StringField("commit_sha", row['commit_sha'], Field.Store.YES))

        doc.add(TextField("content", row['full_content'], Field.Store.YES))
        writer.addDocument(doc)

    writer.commit()
    writer.close()


def main():
    create_code_index(config.PREDICTIONS_CSV, config.CODE_INDEX_DIR)


if __name__ == "__main__":
    if not lucene.getVMEnv():
        try:
            lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        except Exception as e:
            print(f"Failed to initialize Lucene VM: {e}. Exiting.")
            exit()
    main()
