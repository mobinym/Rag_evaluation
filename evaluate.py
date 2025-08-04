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
    ContextRelevance, # ✅ نام صحیح برای نسخه 0.1.7
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




class CustomOllama(OllamaLLM):
    """
    A custom Ollama LLM wrapper that removes the 'temperature' parameter 
    from kwargs before making the request, to support older servers.
    """
    def _create_generate_stream(self, prompt: str, stop: list[str] | None = None, **kwargs: Any):
        # قبل از ارسال درخواست، پارامتر دما را از آن حذف کن
        kwargs.pop("temperature", None)
        # حالا متد اصلی کلاس پدر را با پارامترهای اصلاح شده فراخوانی کن
        return super()._create_generate_stream(prompt, stop, **kwargs)



# --- ۱. بارگذاری تنظیمات ---
def load_config(config_path: str = "config.yml") -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# --- ۲. کلاینت سرویس RAG ---
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

# --- ۳. منطق ارزیابی ---
# این کد را جایگزین کل تابع run_evaluation خود کنید

def run_evaluation(config: Dict[str, Any]):
    rag_client = RAGServiceClient(config)
    
    try:
        df = pd.read_csv(config['qa_dataset_path'], delimiter=",")
        questions = df["question"].tolist()
        ground_truths = df["ground_truth"].tolist()

        print("در حال تولید پاسخ‌ها از سرویس RAG برای مجموعه داده ارزیابی...")
        answers = []
        contexts = []
        for q in questions:
            response_json = rag_client.ask(q)
            answers.append(response_json['answer'])
            retrieved_contexts = [chunk['content'] for ref in response_json.get('references', []) for chunk in ref.get('chunks', [])]
            contexts.append(retrieved_contexts)
            print(f"سوال پردازش شد: {q[:50]}...")

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths
        })

        ragas_config = config['ragas']
        
        # ✅ تغییر اول: اضافه کردن temperature=None برای سازگاری با سرور شما
        ragas_llm = LangchainLLMWrapper(
            CustomOllama(
                model=ragas_config['llm_model'],
                base_url=ragas_config['ollama_base_url'],
                timeout=600
            )
        )

        # این متغیر اینجا تعریف می‌شود
        ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=ragas_config['embedding_model']))

        faithfulness = Faithfulness(llm=ragas_llm)
        # و اینجا در متریک استفاده می‌شود
        answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
        context_recall = ContextRecall(llm=ragas_llm)
        context_precision = ContextPrecision(llm=ragas_llm)
        context_relevancy = ContextRelevance(llm=ragas_llm)

        metrics = [
            context_relevancy, context_precision,
            faithfulness, answer_relevancy, context_recall,
        ]

        print("در حال شروع ارزیابی Ragas...")
        
        # ✅ تغییر دوم: معرفی کردن embeddings به تابع اصلی evaluate
        # و مطمئن شوید که نام متغیر ragas_embeddings اینجا درست تایپ شده
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            embeddings=ragas_embeddings
        )
        
        print("ارزیابی به پایان رسید.")
        return result

    finally:
        rag_client.cleanup()

# --- ۴. بصری‌سازی ---
def visualize_results(result_dataset: Dataset):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    df = result_dataset.to_pandas()
    metric_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    question_col = 'question' if 'question' in df.columns else None
    
    plt.figure(figsize=(10, len(df) * 0.6))
    sns.heatmap(
        df[metric_columns].astype(float),
        annot=True, fmt=".2f", linewidths=.5, cmap="coolwarm",
        yticklabels=df[question_col].str.slice(0, 40) if question_col else False
    )
    plt.xticks(rotation=45, ha="right")
    if question_col:
        plt.yticks(rotation=0)
    plt.title("RAG Evaluation Metrics Heatmap")
    plt.tight_layout()
    plt.show()

# --- اجرای اصلی ---
if __name__ == "__main__":
    config = load_config()
    evaluation_result = run_evaluation(config)
    
    print("\n--- نتایج ارزیابی ---")
    df_results = evaluation_result.to_pandas()
    print(df_results)
    
    print("\nدر حال بصری‌سازی نتایج...")
    visualize_results(evaluation_result)