import os

from tqdm import tqdm
from upstash_vector import Index
from upstash_vector.types import Data


def main():
    with open("assets/stretchlab_document.txt", "r") as f:
        document = f.read()
        documents = document.split("---")

    vector_db = []
    for i in tqdm(range(len(documents))):
        current_document = documents[i]
        current_document = current_document.replace("\n", "...")
        data = Data(id=i, data=current_document, metadata={"text": current_document})
        vector_db.append(data)

    index = Index(
        url=os.environ["STRETCHLAB_VECTOR_DB"],
        token=os.environ["STRETCHLAB_VECTOR_TOKEN"],
    )
    index.reset()
    index.upsert(vectors=vector_db)


if __name__ == "__main__":
    main()
