import os

from upstash_vector import Index


def query(text: str, threshold=0.85):
    index = Index(
        url=os.environ["STRETCHLAB_VECTOR_DB"],
        token=os.environ["STRETCHLAB_VECTOR_TOKEN"],
    )

    response = index.query(
        data=text,
        top_k=5,
        include_metadata=True,
        include_data=True,
    )

    response = [res.data for res in response if res.score >= threshold]

    return "...".join(response)
