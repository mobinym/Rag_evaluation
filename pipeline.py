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

# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
# from langchain_community.document_loaders import DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_core.prompts import PromptTemplate
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser
from ragas.metrics.base import Metric
from datasets import Dataset
import numpy as np
from typing import List
from ragas.embeddings.base import BaseRagasEmbeddings  






from langchain_core.embeddings import Embeddings
import requests
from typing import List

class APICallEmbeddings(Embeddings):
    """
    A custom embedding class that calls a remote API endpoint.
    It expects the API to take a JSON with a 'texts' key and return
    a JSON with a 'vectors' key.
    """
    def __init__(self, api_url: str):
        self.api_url = api_url

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Helper function to call the API."""
        try:
            response = requests.post(self.api_url, json={"texts": texts}, timeout=60) 
            response.raise_for_status()
            data = response.json()
            
            if "vectors" not in data or not isinstance(data["vectors"], list):
                raise ValueError(f"API response is not in the expected format. Expected key 'vectors', but got keys: {data.keys()}")
            
            return data["vectors"]

        except requests.RequestException as e:
            raise IOError(f"Failed to get a valid response from embedding API: {e}") from e
        except ValueError as e:
            raise ValueError(f"Could not parse embedding API response: {e}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents by calling the API."""
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query by calling the API."""
        result = self._embed([text])
        return result[0] if result else []
    
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



def run_evaluation(config: Dict[str, Any], status_updater=None):
    rag_client = RAGServiceClient(config)
    try:
        if status_updater: status_updater("در حال خواندن فایل پرسش و پاسخ...")
        df = pd.read_csv(config['qa_dataset_path'])
        if 'reference' in df.columns:
            df.rename(columns={'reference': 'ground_truth'}, inplace=True)
        original_df = df.copy()
        questions, ground_truths = df["question"].tolist(), df["ground_truth"].tolist()
        
        if status_updater: status_updater("در حال تولید پاسخ‌ها از سرویس RAG...")
        answers, contexts = [], []
        total_questions = len(questions)
        for i, q in enumerate(questions):
            if status_updater: status_updater(f"در حال پردازش سوال {i+1}/{total_questions}...")
            response_json = rag_client.ask(q)
            answers.append(response_json['answer'])
            retrieved_contexts = [chunk['content'] for ref in response_json.get('references', []) for chunk in ref.get('chunks', [])]
            contexts.append(retrieved_contexts)
        
        original_df["answer"], original_df["contexts"] = answers, contexts
        dataset = Dataset.from_pandas(original_df)
        
        ragas_config = config['ragas']
        embedding_service_config = config['embedding_service'] # خواندن تنظیمات سرویس امبدینگ
        
        ragas_llm = LangchainLLMWrapper(
            CustomOllama(
                model=ragas_config['llm_model'],
                base_url=ragas_config['ollama_base_url'],
                timeout=600
            )
        )
        
        ragas_embeddings = LangchainEmbeddingsWrapper(
            APICallEmbeddings(api_url=embedding_service_config['api_url'])
        )
        
        metrics = [
            ContextRelevance(llm=ragas_llm), 
            ContextPrecision(llm=ragas_llm), 
            Faithfulness(llm=ragas_llm), 
            ContextRecall(llm=ragas_llm)
        ]
        
        if status_updater: status_updater("در حال اجرای ارزیابی Ragas...")
        result = evaluate(dataset=dataset, metrics=metrics, embeddings=ragas_embeddings)
        
        ragas_scores_df = result.to_pandas()
        final_df = pd.concat([original_df.reset_index(drop=True), ragas_scores_df.reset_index(drop=True)], axis=1)
        
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

# def calculate_overall_score(df: pd.DataFrame) -> float:
#     """محاسبه یک امتیاز کلی بر اساس میانگین متریک‌های کلیدی."""
#     # ما متریک‌های کلیدی را اینجا تعریف می‌کنیم
#     key_metrics = ['context_precision', 'faithfulness', 'context_recall', 'direct_answer_relevancy']
    
#     # فقط متریک‌هایی که در دیتافریم وجود دارند را در نظر می‌گیریم
#     existing_key_metrics = [m for m in key_metrics if m in df.columns]
#     print(f"[INFO] متریک‌های کلیدی برای محاسبه امتیاز نهایی: {existing_key_metrics}")
    
#     # میانگین هر ستون را محاسبه کرده و مقادیر NaN را نادیده می‌گیریم
#     metric_averages = df[existing_key_metrics].mean(numeric_only=True, skipna=True)
    
#     # میانگین این میانگین‌ها را به عنوان امتیاز کلی برمی‌گردانیم
#     overall_score = metric_averages.mean()
    
#     return overall_score
# این تابع را با نسخه فعلی خود جایگزین کنید

def calculate_overall_score(df: pd.DataFrame) -> tuple[float, pd.Series]:
    """
    یک امتیاز کلی بر اساس میانگین وزنی محاسبه کرده و آن را به همراه
    میانگین‌های هر یک از متریک‌های کلیدی برمی‌گرداند.
    """
    weights = {
        'faithfulness': 3.0,
        'direct_answer_relevancy': 2.0,
        'context_recall': 1.5,
        'context_precision': 1.0
    }
    
    existing_key_metrics = [m for m in weights.keys() if m in df.columns]
    active_weights = {k: v for k, v in weights.items() if k in existing_key_metrics}
    
    metric_averages = df[existing_key_metrics].mean(skipna=True)
    

    weighted_sum = (metric_averages * pd.Series(active_weights)).sum()
    total_weight = sum(active_weights.values())
    
    if total_weight == 0:
        return 0.0, pd.Series(dtype=float) 
        
    overall_score = weighted_sum / total_weight
    
    return overall_score, metric_averages


from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib.font_manager import FontProperties

def save_heatmap(result_df: pd.DataFrame, output_path: str):
    print(f"[INFO] در حال ذخیره نمودار هیت‌مپ در {output_path}...")
    
    font_path = 'Vazirmatn-Regular.ttf' 
    try:
        persian_font = FontProperties(fname=font_path)
    except FileNotFoundError:
        print(f"[WARNING] فونت در مسیر '{font_path}' یافت نشد. از فونت پیش‌فرض استفاده می‌شود.")
        persian_font = FontProperties()
    # -----------------------------------------

    metric_columns = result_df.select_dtypes(include=np.number).columns.tolist()
    question_col = 'question' if 'question' in result_df.columns else None
    
    if question_col:
        original_labels = result_df[question_col].str.slice(0, 50)
        reshaped_labels = [get_display(arabic_reshaper.reshape(label)) for label in original_labels]
    else:
        reshaped_labels = False
    # --------------------------------

    plt.figure(figsize=(12, len(result_df) * 0.6))
    cmap = LinearSegmentedColormap.from_list(
    "green_red", ["#E74C3C", "#2ECC71"] 
    )

    ax = sns.heatmap(
        result_df[metric_columns].astype(float),
        annot=True, fmt=".2f", linewidths=.5, cmap=cmap,
        yticklabels=reshaped_labels 
    )
    
    ax.set_title("RAG Evaluation Metrics Heatmap", fontproperties=persian_font)
    plt.xticks(rotation=45, ha="right", fontproperties=persian_font)
    if question_col:
        ax.set_yticklabels(ax.get_yticklabels(), fontproperties=persian_font, rotation=0, ha="right")

    plt.xlabel("")
    plt.ylabel("") 
    # ------------------------------------------------

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print("[INFO] نمودار با موفقیت و با پشتیبانی از زبان فارسی ذخیره شد.")



import sys
def main():
    """تابع اصلی برای اجرای کل فرآیند ارزیابی."""
    try:
        config = load_config()
        ci_config = config.get('ci_cd', {})
        artifacts_dir = ci_config.get('artifacts_dir', 'evaluation_artifacts')
        pass_threshold = ci_config.get('pass_threshold', 0.75)

        os.makedirs(artifacts_dir, exist_ok=True)
        csv_path = os.path.join(artifacts_dir, "rag_evaluation_results.csv")
        heatmap_path = os.path.join(artifacts_dir, "rag_evaluation_heatmap.png")
        summary_path = os.path.join(artifacts_dir, "evaluation_summary.csv") # ✅ مسیر فایل خلاصه

        df_ragas, embeddings = run_evaluation(config)
        print("[INFO] در حال محاسبه امتیاز Direct Answer Relevancy...")
        df_final = calculate_direct_relevancy(df_ragas, embeddings)
        
        print("\n--- نتایج نهایی ارزیابی ---")
        print(df_final)

        df_final.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n[INFO] نتایج کامل با موفقیت در فایل '{csv_path}' ذخیره شد.")
        save_heatmap(df_final, heatmap_path)

        
        print("\n" + "="*50)
        print("[CI/CD] در حال بررسی نتایج برای قبولی...")
        
        overall_score, metric_averages = calculate_overall_score(df_final)
        
        print("\n--- میانگین امتیازات متریک‌های کلیدی ---")
        if not metric_averages.empty:
            for metric_name, avg_score in metric_averages.items():
                print(f"[RESULT] میانگین {metric_name:<25}: {avg_score:.4f}")
        print("-----------------------------------------")
        
        print(f"[RESULT] امتیاز کلی محاسبه شده (وزنی): {overall_score:.4f}")
        print(f"[CONFIG] آستانه قبولی تنظیم شده: {pass_threshold:.4f}")

        try:
            summary_df = metric_averages.to_frame(name='Average Score')
            summary_df.loc['OVERALL_SCORE_WEIGHTED'] = overall_score
            summary_df.to_csv(summary_path, encoding='utf-8-sig')
            print(f"[INFO] خلاصه امتیازات با موفقیت در '{summary_path}' ذخیره شد.")
        except Exception as e:
            print(f"[WARNING] خطا در ذخیره فایل خلاصه امتیازات: {e}")

        if overall_score >= pass_threshold:
            print("\n[CI/CD CHECK PASSED] ✔️ امتیاز کلی بالاتر از آستانه است. بیلد موفقیت‌آمیز بود.")
            sys.exit(0)
        else:
            print(f"\n[CI/CD CHECK FAILED] ❌ امتیاز کلی ({overall_score:.4f}) پایین‌تر از آستانه ({pass_threshold:.4f}) است. بیلد ناموفق بود.")
            sys.exit(1)

    except Exception as e:
        print(f"\n[FATAL ERROR] یک خطای پیش‌بینی‌نشده در حین اجرای اسکریپت رخ داد: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()