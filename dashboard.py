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

# Main section to display results
with col2:
    st.header("📈 Evaluation Results")

    # If button is clicked, start evaluation
    if start_button:
        if not uploaded_docs or not uploaded_qa_file:
            st.warning("Please upload both knowledge files and a Q&A file.")
        else:
            with st.spinner("Please wait, evaluation is in progress... This may take a few minutes."):
                # Create a temporary folder to store uploaded files
                with tempfile.TemporaryDirectory() as temp_dir:
                    docs_path = os.path.join(temp_dir, "docs")
                    os.makedirs(docs_path)
                    
                    # Save knowledge files in the temporary folder
                    for doc in uploaded_docs:
                        with open(os.path.join(docs_path, doc.name), "wb") as f:
                            f.write(doc.getbuffer())
                            
                    # Save Q&A file in the temporary folder
                    qa_path = os.path.join(temp_dir, "qa.csv")
                    with open(qa_path, "wb") as f:
                        f.write(uploaded_qa_file.getbuffer())

                    # Load main config and update paths
                    config = load_config()
                    config['source_document_path'] = docs_path
                    config['qa_dataset_path'] = qa_path
                    config['ci_cd']['pass_threshold'] = threshold
                    
                    # Placeholder for logs during execution
                    status_placeholder = st.empty()
                    def update_status(message):
                        status_placeholder.info(message)

                    try:
                        # Run full pipeline
                        update_status("Starting evaluation process...")
                        df_ragas, embeddings = run_evaluation(config, status_updater=update_status)
                        
                        update_status("Calculating Direct Answer Relevancy score...")
                        df_final = calculate_direct_relevancy(df_ragas, embeddings)
                        
                        update_status("Calculating overall score...")
                        overall_score = calculate_overall_score(df_final)
                        
                        # Save results in session state for later display
                        st.session_state['results_df'] = df_final
                        st.session_state['overall_score'] = overall_score
                        st.session_state['threshold'] = threshold
                        
                        status_placeholder.empty()  # Clear final status message
                        st.rerun()  # Rerun to display results

                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                        st.exception(e)


    # If results exist in session, display them
    if 'results_df' in st.session_state:
        results_df = st.session_state['results_df']
        score = st.session_state['overall_score']
        current_threshold = st.session_state['threshold']

        # Display overall score and pass/fail status
        if score >= current_threshold:
            st.success(f"✔️ Evaluation successful (Overall Score: {score:.2f})")
        else:
            st.error(f"❌ Evaluation failed (Overall Score: {score:.2f} is less than threshold {current_threshold:.2f})")
        
        st.metric(label="Overall Score (Weighted Average)", value=f"{score:.2%}")

        # Create tabs for different results views
        tab1, tab2, tab3 = st.tabs(["📊 Heatmap Chart", "📄 Full Results", "✍️ Manual Review"])

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
                'question': 'Asked Question',
                'answer': 'RAG Answer',
                'ground_truth': 'Expected Answer'
            })
            st.dataframe(comparison_df)
    else:
        st.info("To get started, upload the required files from the left menu and click the 'Start Evaluation' button.")
