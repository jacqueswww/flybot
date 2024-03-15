import httpx
import asyncio
import os
import chromadb

from typing import Any, List
from langchain_community.document_loaders import (
    AsyncHtmlLoader,
    TextLoader,
)
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema.document import Document
from unstructured.partition.html import partition_html
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.vectorstores import Chroma


async def get_page_count(url, limit):
    async with httpx.AsyncClient() as client:
        resp = await client.get(url + f'?limit={limit}')
        res = resp.json()
        return res['page_count']


async def get_article_urls():
    url = 'https://flysafair.zendesk.com/api/v2/help_center/en-us/articles.json'
    limit = 20
    total_pages = await get_page_count(url, limit)

    for current_page in range(total_pages):
        async with httpx.AsyncClient() as client:
            resp = await client.get(url + f'?limit={limit}&page={current_page}')
            res = resp.json()
            for article in res['articles']:
                yield article


async def download_all_articles():
    docs = []
    async for article in get_article_urls():
        async with httpx.AsyncClient() as client:
            resp = await client.get(article['url'])
            body = resp.json()['article']['body']
            path = 'texts/fa_zendesk_{}.html'.format(article['id'])
            open(path, 'w').write(body)
            docs.append(path)
            print('.', end='', flush=True)
    return docs


def populate_vector_store(documents):

    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    # texts = [doc.page_content for doc in docs]

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # db = SQLiteVSS.from_texts(
    #     texts=texts,
    #     embedding=embedding_function,
    #     table="fa_zendesk_articles",
    #     db_file="/tmp/vss.db",
    # )
    persistent_client = chromadb.PersistentClient()
    collection = persistent_client.get_or_create_collection("fa_zendesk_articles")
    collection.add(
        ids=[
            doc.metadata['source']
            for doc in documents
        ],
        documents=[
            doc.page_content
            for doc in documents
        ]
    )
    db = Chroma(
        client=persistent_client,
        collection_name="fa_zendesk_articles",
        embedding_function=embedding_function,
        persist_directory="./chroma_db"
    )
    db = Chroma.from_documents(
        docs,
        embedding_function,
        persist_directory="./chroma_db"
    )
    return db


def get_zendesk_vector_db_connection():
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    # connection = SQLiteVSS.create_connection(
    #     db_file="/tmp/vss.db"
    # )
    # db = SQLiteVSS(
    #     table="fa_zendesk_articles",
    #     embedding=embedding_function,
    #     connection=connection
    # )
    persistent_client = chromadb.PersistentClient()
    db = Chroma(
        client=persistent_client,
        collection_name="fa_zendesk_articles",
        embedding_function=embedding_function,
        persist_directory="./chroma_db"
    )
    return db


async def main_populate():
    # docs = await download_all_articles()
    # print(f'\n {len(docs)} articles downloaded.')

    paths = os.listdir('./texts/')
    docs = []
    for path in paths:
        loader = UnstructuredHTMLLoader('./texts/' + path)
        docs.append(
            loader.load()[0]
        )

    populate_vector_store(docs)

    # # query it
    # query = "What do I need to fly to zanzibar?"
    # data = db.similarity_search(query)

    # # print results
    # data[0].page_content


if __name__ == '__main__':
    asyncio.run(main_populate())
