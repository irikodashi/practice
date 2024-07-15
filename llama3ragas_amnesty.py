
from datasets import load_dataset

# loading the V2 dataset
amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")

amnesty_subset = amnesty_qa["eval"].select(range(2))
amnesty_subset.to_pandas()

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)


from langchain_community.chat_models import ChatOllama
from ragas import evaluate
from langchain_community.embeddings import OllamaEmbeddings

langchain_llm = ChatOllama(model="llama3")
langchain_embeddings = OllamaEmbeddings(model="llama3")

result = evaluate(
    amnesty_subset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall
    ],
    llm=langchain_llm,
    embeddings=langchain_embeddings
)
print(result)
