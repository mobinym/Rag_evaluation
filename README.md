
-----

# Automated RAG Evaluation Service

This project provides a robust, automated pipeline and RESTful API for evaluating the performance of Retrieval-Augmented Generation (RAG) systems. It uses the `Ragas` framework for standard metrics and is enhanced with custom components to provide a more reliable and nuanced assessment, especially when dealing with specific server configurations or models.

The final application is a **FastAPI service** that accepts documents and a question-answer dataset, runs a comprehensive evaluation in the background, and provides the results through a queryable API endpoint.

## ✨ Features

  * **RESTful API:** Programmatically trigger evaluations by sending documents and a QA file.
  * **Asynchronous by Design:** Handles long-running evaluation tasks in the background without timing out, returning a task ID for status polling.
  * **Comprehensive Metrics:** Utilizes standard `Ragas` metrics to evaluate both the **Retriever** (`Context Precision`, `Context Recall`) and the **Generator** (`Faithfulness`).
  * **Custom & Reliable Relevancy Metric:** Includes a custom `DirectAnswerRelevancy` metric that calculates the direct semantic similarity between a question and its answer, bypassing the need for a powerful "judge" LLM where default metrics might fail.
  * **CI/CD Friendly:** Calculates a final weighted score and compares it against a configurable threshold, making it easy to integrate into automated build/deployment pipelines.
  * **Automated Artifact Generation:** Automatically saves detailed results, a summary of scores, and a visual heatmap chart for each evaluation run.
  * **Interactive Documentation:** Auto-generates a Swagger UI (`/docs`) for easy, interactive API testing directly from your browser.

## 🏗️ Project Architecture & Workflow

The service is designed around an asynchronous task queue pattern to handle time-consuming evaluations.

1.  **Request Initiation:** A client sends a `POST` request to the `/evaluate` endpoint with the knowledge documents and the QA CSV file.
2.  **Task Queuing:** The FastAPI server immediately validates the input, creates a unique `task_id`, and adds the evaluation job to a `BackgroundTasks` queue. It instantly responds to the client with the `task_id` and a URL to check the status.
3.  **Background Processing (`pipeline.py`):**
      * The background task sets up a connection to the external RAG service using the `RAGServiceClient`.
      * It uploads the documents and iterates through the QA file, querying the RAG service for each question.
      * It runs the standard `Ragas` metrics (`Faithfulness`, `ContextRecall`, etc.) on the collected results.
      * It then manually calculates the custom `direct_answer_relevancy` metric.
      * Finally, it computes the overall weighted score.
4.  **Result Storage:** The final results (detailed DataFrame, summary scores) are stored in an in-memory dictionary keyed by the `task_id`.
5.  **Status Polling:** The client can send `GET` requests to the `/evaluate/status/{task_id}` endpoint to check the job's status (`pending`, `running`, `completed`, or `failed`). Once completed, the response will contain the full evaluation results in JSON format.

## 📁 Project Structure

```
.
├── evaluation_artifacts/ # Output directory for saved results (CSVs, images)
├── venv/                 # Python virtual environment
├── api.py                # The main FastAPI application file (endpoints, background tasks)
├── pipeline.py           # The core evaluation logic, custom classes, and helper functions
├── config.yml            # Central configuration for models, servers, and thresholds
└── requirements.txt      # Project dependencies and pinned versions
```

## ⚙️ Core Computational Metrics

The evaluation is based on several key metrics, each answering a specific question about the RAG system's quality.

#### Retriever Evaluation

These metrics assess the quality of the retrieved context.

  * **Context Precision:**
      * **What it is:** A measure of the signal-to-noise ratio. It answers: *Is all the retrieved information actually relevant to the question?*
      * **How it's calculated:** An LLM judge determines the ratio of "essential" sentences to the total number of sentences in the retrieved context. A score of `1.0` means there is zero noise.
  * **Context Recall:**
      * **What it is:** A measure of completeness. It answers: *Did the retriever find all the necessary information to answer the question?*
      * **How it's calculated:** An LLM judge checks if all the statements from the human-written `ground_truth` answer can be verified using the retrieved context. A score of `1.0` means nothing was missed.

#### Generator Evaluation

These metrics assess the quality of the final generated answer.

  * **Faithfulness:**

      * **What it is:** The most important metric for preventing hallucination. It answers: *Is the generated answer strictly based on the provided context?*
      * **How it's calculated:** An LLM judge breaks down the generated answer into individual statements and verifies each one against the retrieved context. A score of `1.0` means the answer is 100% grounded in the source text.

  * **Direct Answer Relevancy (Custom Metric):**

      * **What it is:** A direct measure of semantic similarity between the question and the answer. It bypasses the weak "judge" LLM that causes issues with the default `AnswerRelevancy` metric.
      * **How it's calculated:** It is a purely mathematical calculation. The question and answer are converted into numerical vectors (embeddings). The final score is the **Cosine Similarity** between these two vectors. A score of `1.0` means perfect semantic alignment.

#### Final Score Calculation

  * **Overall Score:**
      * **What it is:** A single, meaningful number representing the overall quality of the RAG system for CI/CD pass/fail decisions.
      * **How it's calculated:** It is a **Weighted Average** of the mean scores of the key metrics. The weights are defined in the `calculate_overall_score` function in `pipeline.py`, allowing you to prioritize what's most important (e.g., giving `Faithfulness` a higher weight).
      * **Formula:** `Overall Score = Σ(Avg(Metric) * Weight(Metric)) / Σ(Weight(Metric))`

## 🚀 Getting Started

### 1\. Prerequisites

  * Python 3.10+
  * Access to the RAG service and the Ollama-compatible service defined in `config.yml`.

### 2\. Installation

1.  Clone this repository.
2.  Create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  Install all required dependencies from the `requirements.txt` file. Make sure the file is saved with **UTF-8 with BOM** encoding if you are on Windows.
    ```bash
    pip install -r requirements.txt
    ```

### 3\. Configuration

Modify the `config.yml` file with your specific settings:

  * **`server`**: Update the `base_url` and endpoints to point to your deployed RAG service.
  * **`ragas`**: Set the names of the LLM and Embedding models used by the evaluation framework.
  * **`ci_cd`**: Adjust the `pass_threshold` to your desired quality bar.

### 4\. Running the Service

Launch the FastAPI application using Uvicorn:

```bash
uvicorn api:app --reload
```

The service will be available at `http://127.0.0.1:8000`.

### 5\. Using the API

1.  Open your web browser and navigate to the interactive documentation: **`http://127.0.0.1:8000/docs`**.
2.  Expand the `POST /evaluate` endpoint and click "Try it out".
3.  Upload your knowledge document(s) and your QA dataset CSV file.
4.  Click "Execute". The API will return a `task_id`.
5.  Copy the `task_id` and use it with the `GET /evaluate/status/{task_id}` endpoint to poll for the results. When the status is `completed`, the full JSON response will be displayed.