import warnings
warnings.filterwarnings("ignore")

from scripts.model.embeddings import embedding_pipeline

if __name__ == "__main__":
    summaries = embedding_pipeline()
    print("-----------Summaries:-----------")
    print(len(summaries))
    print(summaries)
