import lucene
import os
import pandas as pd
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, StringField, TextField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import FSDirectory
import config

try:
    if not lucene.getVMEnv():
        lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        print("Lucene VM initialized successfully for issue indexing.")
except Exception as e:
    print(f"Failed to initialize Lucene VM: {e}")
    exit()


def create_issue_index(issues_csv, index_dir):
    df = pd.read_csv(issues_csv)

    df = df.fillna({
        'title': '',
        'body': '',
        'code_snippet_extracted': ''
    })

    if df.empty:
        return

    if not os.path.exists(index_dir):
        os.makedirs(index_dir)

    directory = FSDirectory.open(Paths.get(index_dir))
    analyzer = StandardAnalyzer()
    config_writer = IndexWriterConfig(analyzer)
    config_writer.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    writer = IndexWriter(directory, config_writer)

    for index, row in df.iterrows():
        doc = Document()
        doc.add(StringField("project", str(row['project']), Field.Store.YES))
        doc.add(StringField("url", str(row['url']), Field.Store.YES))
        doc.add(StringField("issue_number", str(row['issue_number']), Field.Store.YES))

        full_content = f"{row['title']} \n {row['body']}"
        doc.add(TextField("content", full_content, Field.Store.YES))

        doc.add(TextField("code_snippet", str(row['code_snippet_extracted']), Field.Store.YES))

        writer.addDocument(doc)

    writer.commit()
    writer.close()


def main():
    create_issue_index(config.ISSUES_CSV, config.ISSUE_INDEX_DIR)


if __name__ == "__main__":
    if not lucene.getVMEnv():
        try:
            lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        except Exception as e:
            print(f"Failed to initialize Lucene VM: {e}. Exiting.")
            exit()
    main()
