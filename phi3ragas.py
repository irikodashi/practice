
from datasets import Dataset 

data_samples = {
    'question': ['倉敷の名産品は？', '吉備津神社と吉備津彦神社の違いは？'],
    'answer': ['倉敷の名産品は、「倉敷デニム」です。倉敷は日本でも有数のデニム生産地であり、高品質なデニム製品が多く生産されています。', '吉備津神社と吉備津彦神社の違いは、「吉備津神社は商業施設で、吉備津彦神社は学校です」。これは誤った情報です。'],
    'contexts' : [['倉敷の名物は、「倉敷デニム」です。','もう一つの名物は「倉敷の和菓子」です。特に「むらすずめ」という和菓子'], 
    ['吉備津神社と吉備津彦神社は共に岡山県。','吉備津神社は桃太郎伝説に関連する神社で、特に「鳴釜神事」']],
    'ground_truth': ['倉敷の名産品は、「倉敷デニム」です。', '吉備津神社と吉備津彦神社は共に岡山県にありますが、異なる神社です。吉備津神社は桃太郎伝説に関連する神社で、特に「鳴釜神事」が有名です。']
}
dataset = Dataset.from_dict(data_samples)

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_similarity,
    answer_correctness
)

from langchain_community.chat_models import ChatOllama
from ragas import evaluate
from langchain_community.embeddings import OllamaEmbeddings

# モデルのロード
try:
    langchain_llm = ChatOllama(model="phi3:latest")
    langchain_embeddings = OllamaEmbeddings(model="phi3:latest")
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

result = evaluate(
    dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        answer_similarity,
        answer_correctness
    ],
    llm=langchain_llm,
    embeddings=langchain_embeddings
)

# 評価結果をExcelファイルに出力
result_df = result.to_pandas()
result_df.to_excel("evaluation_results.xlsx", index=False)
