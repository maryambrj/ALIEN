import faiss
import argparse
import pandas as pd
from dotenv import load_dotenv
from uuid import uuid4
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

load_dotenv("../.env")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create FAISS retrievers.")
    parser.add_argument(
        "--df_path",
        type=str,
        required=True,
        help="Path of the dataframe.",
    )
    parser.add_argument(
        "--faiss_embeddings_model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI model for embeddings.",
    )
    parser.add_argument(
        "--faiss_retriever_k",
        type=int,
        default=5,
        help="Number of documents to retrieve.",
    )
    parser.add_argument(
        "--faiss_index_save_path",
        type=str,
        default="faiss_index",
        help="Path to save the FAISS index."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Load the dataframe
    df = pd.read_csv(args.df_path)
    # validated the df to have sentence, entity_1, entity_2, and label columns
    required_columns = [
        "sentence", "entity_1", "entity_2", "relation"
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Create FAISS embeddings
    embedding_model = OpenAIEmbeddings(model=args.faiss_embeddings_model)

    # Create FAISS index
    faiss_index = faiss.IndexFlatL2(
        len(embedding_model.embed_query("hello world"))
    )

    # Create FAISS vector store
    faiss_vectorstore = FAISS(
        embedding_function=embedding_model,
        index=faiss_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    all_documents = [
        Document(
            page_content=row["sentence"],
            metadata=row.to_dict(),
        )
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating documents")
    ]
    uuids = [str(uuid4()) for _ in range(len(all_documents))]
    faiss_vectorstore.add_documents(documents=all_documents, ids=uuids)

    faiss_vectorstore.save_local(args.faiss_index_save_path)
    print("FAISS index saved to: {}".format(args.faiss_index_save_path))
