import streamlit as st
import pandas as pd
import os
import tempfile
import time

# Import all logic from the pipeline file
from pipeline import (
    load_config,
    run_evaluation,
    calculate_direct_relevancy,
    calculate_overall_score,
    save_heatmap
)

# Initial page settings
st.set_page_config(layout="wide", page_title="RAG Evaluation Dashboard")

# --- User Interface ---
st.title("📊 Automatic RAG Evaluation Dashboard")
st.markdown("This dashboard allows you to evaluate the performance of your RAG system using your own files.")

# Layout for inputs and results
col1, col2 = st.columns([1, 2])

with col1:
    st.header("⚙️ Input Settings")
    
    uploaded_docs = st.file_uploader(
        "1. Upload knowledge files (PDF, TXT, DOCX)",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True
    )
    
    uploaded_qa_file = st.file_uploader(
        "2. Upload Q&A file (CSV)",
        type=['csv']
    )
    
    threshold = st.slider("3. Pass Threshold", 0.0, 1.0, 0.75, 0.05)
    
    start_button = st.button("🚀 Start Evaluation", type="primary", use_container_width=True)

with col2:
    st.header("📈 Evaluation Results")

    if start_button:
        if not uploaded_docs or not uploaded_qa_file:
            st.warning("لطفاً هم فایل دانش و هم فایل پرسش و پاسخ را آپلود کنید.")
        else:
            with st.spinner("لطفاً صبر کنید، فرآیند ارزیابی در حال انجام است... این کار ممکن است چند دقیقه طول بکشد."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    docs_path = os.path.join(temp_dir, "docs")
                    os.makedirs(docs_path)
                    
                    for doc in uploaded_docs:
                        with open(os.path.join(docs_path, doc.name), "wb") as f:
                            f.write(doc.getbuffer())
                            
                    qa_path = os.path.join(temp_dir, "qa.csv")
                    with open(qa_path, "wb") as f:
                        f.write(uploaded_qa_file.getbuffer())

                    config = load_config()
                    config['source_document_path'] = docs_path
                    config['qa_dataset_path'] = qa_path
                    config['ci_cd']['pass_threshold'] = threshold
                    
                    status_placeholder = st.empty()
                    def update_status(message):
                        status_placeholder.info(message)

                    try:
                        update_status("شروع فرآیند ارزیابی...")
                        df_ragas, embeddings = run_evaluation(config, status_updater=update_status)
                        
                        update_status("محاسبه امتیاز Direct Answer Relevancy...")
                        df_final = calculate_direct_relevancy(df_ragas, embeddings)
                        
                        update_status("محاسبه امتیاز کلی...")
             
                        overall_score, metric_averages = calculate_overall_score(df_final)
 
                        st.session_state['results_df'] = df_final
                        st.session_state['overall_score'] = overall_score
                        st.session_state['metric_averages'] = metric_averages 
                        st.session_state['threshold'] = threshold
                        
                        status_placeholder.empty()
                        st.rerun()

                    except Exception as e:
                        st.error(f"یک خطای پیش‌بینی‌نشده رخ داد: {e}")
                        st.exception(e)



    if 'results_df' in st.session_state:
        results_df = st.session_state['results_df']
        score = st.session_state['overall_score']
        metric_averages = st.session_state['metric_averages']
        current_threshold = st.session_state['threshold']


        if score >= current_threshold:
            st.success(f"✔️ ارزیابی با موفقیت انجام شد (امتیاز کلی: {score:.2f})")
        else:
            st.error(f"❌ ارزیابی ناموفق بود (امتیاز کلی: {score:.2f} کمتر از آستانه {current_threshold:.2f})")

        score_col, avg_col = st.columns(2)
        
        with score_col:
            st.metric(label="امتیاز کلی (میانگین وزنی)", value=f"{score:.2%}")
            
        with avg_col:

            st.subheader("میانگین امتیازات متریک‌ها")
            st.dataframe(metric_averages.to_frame(name='میانگین امتیاز'))



        tab1, tab2, tab3 = st.tabs(["📊 نمودار هیت‌مپ", "📄 نتایج کامل", "✍️ بررسی دستی"])

        with tab1:
            artifacts_dir = "evaluation_artifacts"
            os.makedirs(artifacts_dir, exist_ok=True)
            heatmap_path = os.path.join(artifacts_dir, f"heatmap_{int(time.time())}.png")
            save_heatmap(results_df, heatmap_path)
            st.image(heatmap_path)

        with tab2:
            st.dataframe(results_df)

        with tab3:
            comparison_df = results_df[['question', 'answer', 'ground_truth']].rename(columns={
                'question': 'سوال پرسیده شده',
                'answer': 'پاسخ RAG',
                'ground_truth': 'پاسخ مورد انتظار'
            })
            st.dataframe(comparison_df)
    else:
        st.info("برای شروع، فایل‌های مورد نیاز را از منوی سمت راست آپلود کرده و روی دکمه «شروع ارزیابی» کلیک کنید.")