import os
import yaml
import pandas as pd
import requests
import time
from typing import List, Dict, Any
import numpy as np
from langchain_ollama import OllamaLLM
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
    ContextRelevance, 
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from ragas.metrics.base import Metric
from datasets import Dataset
import numpy as np
from typing import List
from ragas.embeddings.base import BaseRagasEmbeddings  

class CustomOllama(OllamaLLM):
    """
    A custom Ollama LLM wrapper that removes the 'temperature' parameter 
    from kwargs before making the request, to support older servers.
    """
    def _create_generate_stream(self, prompt: str, stop: list[str] | None = None, **kwargs: Any):
        kwargs.pop("temperature", None)
        return super()._create_generate_stream(prompt, stop, **kwargs)
    
class DirectAnswerRelevancy(Metric):
    """
    Calculates the direct cosine similarity between the question and the answer embeddings.
    This metric does not use an LLM and relies solely on the embedding model.
    It's adapted for the new Ragas API.
    """
    name: str = "direct_answer_relevancy"
    _required_columns: List[str] = ["question", "answer"]
    embeddings: BaseRagasEmbeddings

    def init(self):
        """
        This method is called by Ragas to initialize the metric's dependencies.
        """
        super().init()

    def _score_batch(self, dataset: Dataset) -> List[float]:
        questions = dataset["question"]
        answers = dataset["answer"]

        q_embeddings = self.embeddings.embed_documents(questions)
        a_embeddings = self.embeddings.embed_documents(answers)

        scores = []
        for q_emb, a_emb in zip(q_embeddings, a_embeddings):
            q_emb = np.array(q_emb)
            a_emb = np.array(a_emb)
            
            if np.linalg.norm(q_emb) == 0 or np.linalg.norm(a_emb) == 0:
                scores.append(0.0)
                continue

            similarity = np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb))
            scores.append(float(similarity))
            
        return scores

def load_config(config_path: str = "config.yml") -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class RAGServiceClient:
    def __init__(self, config: Dict[str, Any]):
        print("در حال آماده‌سازی کلاینت سرویس RAG...")
        self.server_config = config['server']
        self.document_path = config['source_document_path']
        self.vs_id = None
        self._setup_session_on_server()
        print(f"کلاینت آماده است. با شناسه VS: {self.vs_id} کار می‌کند.")

    def _setup_session_on_server(self):
        print("در حال ایجاد یک جلسه (VS) خالی روی سرور...")
        create_url = f"{self.server_config['base_url']}{self.server_config['create_empty_vs_endpoint']}"
        response = requests.post(create_url, data={'vector_store_strategy': 'faiss'})
        response.raise_for_status()
        self.vs_id = response.json()['vs_id']
        print(f"VS خالی با شناسه {self.vs_id} ایجاد شد.")

        doc_files = [f for f in os.listdir(self.document_path) if f.endswith(('.txt', '.pdf', '.docx'))]
        if not doc_files:
            raise ValueError(f"هیچ سندی در مسیر {self.document_path} یافت نشد.")

        add_url = f"{self.server_config['base_url']}{self.server_config['add_document_endpoint'].format(vs_id=self.vs_id)}"
        for doc_file in doc_files:
            print(f"در حال افزودن سند: {doc_file}")
            file_path = os.path.join(self.document_path, doc_file)
            with open(file_path, 'rb') as f:
                files = {'file': (doc_file, f)}
                response = requests.post(add_url, files=files)
                response.raise_for_status()
        print(f"تمام اسناد با موفقیت به VS با شناسه {self.vs_id} اضافه شدند.")
        time.sleep(2)

    def ask(self, query: str) -> Dict[str, Any]:
        ask_url = f"{self.server_config['base_url']}{self.server_config['ask_endpoint'].format(vs_id=self.vs_id)}"
        payload = {"query": query}
        response = requests.post(ask_url, json=payload)
        response.raise_for_status()
        return response.json()

    def cleanup(self):
        if self.vs_id:
            print(f"در حال پاک‌سازی VS با شناسه {self.vs_id} از روی سرور...")
            delete_url = f"{self.server_config['base_url']}{self.server_config['delete_vs_endpoint'].format(vs_id=self.vs_id)}"
            try:
                requests.delete(delete_url)
                print("VS با موفقیت پاک‌سازی شد.")
            except requests.RequestException as e:
                print(f"هشدار: پاک‌سازی VS با خطا مواجه شد. خطا: {e}")



def run_evaluation(config: Dict[str, Any]):
    rag_client = RAGServiceClient(config)
    
    try:
        df = pd.read_csv(config['qa_dataset_path'], delimiter=",")
        if 'reference' in df.columns:
            df.rename(columns={'reference': 'ground_truth'}, inplace=True)
        
        original_df = df.copy()

        questions = df["question"].tolist()
        ground_truths = df["ground_truth"].tolist()

        print("در حال تولید پاسخ‌ها از سرویس RAG...")
        answers = []
        contexts = []
        for q in questions:
            response_json = rag_client.ask(q)
            answers.append(response_json['answer'])
            retrieved_contexts = [chunk['content'] for ref in response_json.get('references', []) for chunk in ref.get('chunks', [])]
            contexts.append(retrieved_contexts)
            print(f"سوال پردازش شد: {q[:50]}...")
        
        original_df["answer"] = answers
        original_df["contexts"] = contexts

        dataset = Dataset.from_pandas(original_df)

        ragas_config = config['ragas']
        
        ragas_llm = LangchainLLMWrapper(
            CustomOllama(
                model=ragas_config['llm_model'],
                base_url=ragas_config['ollama_base_url'],
                timeout=600
            )
        )
        
        ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=ragas_config['embedding_model']))

        faithfulness = Faithfulness(llm=ragas_llm)
        context_recall = ContextRecall(llm=ragas_llm)
        context_precision = ContextPrecision(llm=ragas_llm)
        context_relevance = ContextRelevance(llm=ragas_llm)
        
        metrics = [
            context_relevance,
            context_precision,
            faithfulness,
            context_recall,
        ]

        print("در حال شروع ارزیابی Ragas (فقط متریک‌های استاندارد)...")
        
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            embeddings=ragas_embeddings
        )
        
        print("ارزیابی Ragas به پایان رسید.")
        
        ragas_scores_df = result.to_pandas()
        original_df = original_df.drop(columns=[m.name for m in metrics if m.name in original_df.columns], errors='ignore')
        final_df = pd.concat([original_df, ragas_scores_df], axis=1)

        return final_df, ragas_embeddings

    finally:
        rag_client.cleanup()


def calculate_direct_relevancy(results_df: pd.DataFrame, embeddings: LangchainEmbeddingsWrapper) -> pd.DataFrame:
    """محاسبه دستی امتیاز شباهت مستقیم سوال و پاسخ و اضافه کردن آن به دیتافریم نتایج."""
    questions = results_df["question"].tolist()
    answers = results_df["answer"].tolist()
    
    q_embeddings = embeddings.embed_documents(questions)
    a_embeddings = embeddings.embed_documents(answers)
    
    scores = []
    for q_emb, a_emb in zip(q_embeddings, a_embeddings):
        q_emb = np.array(q_emb)
        a_emb = np.array(a_emb)
        
        if np.linalg.norm(q_emb) == 0 or np.linalg.norm(a_emb) == 0:
            scores.append(0.0)
            continue
            
        similarity = np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb))
        scores.append(float(similarity))
        
    results_df["direct_answer_relevancy"] = scores
    return results_df

from matplotlib.colors import LinearSegmentedColormap
def visualize_results(result_df: pd.DataFrame): 
    import seaborn as sns
    import matplotlib.pyplot as plt
    cmap = LinearSegmentedColormap.from_list(
    "green_red", ["#E74C3C", "#2ECC71"] 
)
    df = result_df
    metric_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    question_col = 'question' if 'question' in df.columns else None
    
    plt.figure(figsize=(10, len(df) * 0.6))
    sns.heatmap(
        df[metric_columns].astype(float),
        annot=True, fmt=".2f", linewidths=.5, cmap=cmap,
        yticklabels=df[question_col].str.slice(0, 40) if question_col else False
    )
    plt.xticks(rotation=45, ha="right")
    if question_col:
        plt.yticks(rotation=0)
    plt.title("RAG Evaluation Metrics Heatmap")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    config = load_config()
    
    df_with_ragas_scores, ragas_embeddings = run_evaluation(config)
    
    print("\nدر حال محاسبه امتیاز Direct Answer Relevancy...")
    final_df_with_all_scores = calculate_direct_relevancy(df_with_ragas_scores, ragas_embeddings)
    
    print("\n--- نتایج نهایی ارزیابی (با متریک سفارشی) ---")
    print(final_df_with_all_scores)

    output_path = "rag_evaluation_results.csv"
    try:
        final_df_with_all_scores.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ نتایج با موفقیت در فایل '{output_path}' ذخیره شد.")
    except Exception as e:
        print(f"\n❌ خطا در ذخیره‌سازی فایل CSV: {e}")
    
    print("\nدر حال بصری‌سازی نتایج...")
    visualize_results(final_df_with_all_scores)