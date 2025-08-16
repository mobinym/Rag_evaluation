import os
import uuid
import tempfile
import pandas as pd
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Body
from typing import List, Dict, Any
import numpy as np


from pipeline import (
    load_config,
    run_evaluation,
    calculate_direct_relevancy,
    calculate_overall_score
)

app = FastAPI(
    title="RAG Evaluation Service",
    description="یک سرویس RESTful برای ارزیابی خودکار سیستم‌های RAG",
    version="1.0.0"
)


tasks_db: Dict[str, Dict[str, Any]] = {}



def run_evaluation_task(task_id: str, docs_data: List[tuple[str, bytes]], qa_file_data: bytes):
    """این تابع کل پایپ‌لاین ارزیابی را در پس‌زمینه اجرا می‌کند."""
    try:
        tasks_db[task_id] = {"status": "running", "progress": "آماده‌سازی فایل‌ها..."}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_path = os.path.join(temp_dir, "docs")
            os.makedirs(docs_path)

            for name, content in docs_data:
                with open(os.path.join(docs_path, name), "wb") as f:
                    f.write(content)
            
            qa_path = os.path.join(temp_dir, "qa.csv")
            with open(qa_path, "wb") as f:
                f.write(qa_file_data)

            config = load_config()
            config['source_document_path'] = docs_path
            config['qa_dataset_path'] = qa_path

            def update_status(message):
                tasks_db[task_id]["progress"] = message

            df_ragas, embeddings = run_evaluation(config, status_updater=update_status)
            update_status("محاسبه امتیاز Direct Answer Relevancy...")
            df_final = calculate_direct_relevancy(df_ragas, embeddings)
            update_status("محاسبه امتیاز کلی...")
            overall_score, metric_averages = calculate_overall_score(df_final)


            df_final_json_safe = df_final.replace({np.nan: None})
            
            summary_scores = metric_averages.to_dict()
            summary_scores['OVERALL_SCORE_WEIGHTED'] = overall_score
            
      
            cleaned_summary_scores = {}
            for key, value in summary_scores.items():
                cleaned_summary_scores[key] = None if pd.isna(value) else value
            
            tasks_db[task_id] = {
                "status": "completed",
                "result": {
                    "summary_scores": cleaned_summary_scores,
                    "detailed_results": df_final_json_safe.to_dict(orient='records')
                }
            }

    except Exception as e:
        import traceback
        tasks_db[task_id] = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.post("/evaluate", status_code=202)
async def start_evaluation_endpoint(
    background_tasks: BackgroundTasks,
    documents: List[UploadFile] = File(..., description="یک یا چند فایل دانش (PDF, TXT, DOCX)"),
    qa_file: UploadFile = File(..., description="فایل CSV شامل ستون‌های question و reference/ground_truth")
):
    """
    یک وظیفه ارزیابی جدید را در پس‌زمینه آغاز می‌کند.
    بلافاصله یک شناسه وظیفه (task_id) برمی‌گرداند که می‌توانید با آن وضعیت را پیگیری کنید.
    """
    task_id = str(uuid.uuid4())
    tasks_db[task_id] = {"status": "pending"}

    docs_data = [(doc.filename, await doc.read()) for doc in documents]
    qa_file_data = await qa_file.read()

    background_tasks.add_task(run_evaluation_task, task_id, docs_data, qa_file_data)
    
    return {
        "message": "Evaluation started in the background.",
        "task_id": task_id,
        "status_url": f"/evaluate/status/{task_id}"
    }

@app.get("/evaluate/status/{task_id}")
async def get_evaluation_status_endpoint(task_id: str):
    """
    وضعیت یک وظیفه ارزیابی را با استفاده از task_id آن برمی‌گرداند.
    """
    task = tasks_db.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task