
import requests
import time
import pandas as pd
from textblob import TextBlob
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

# Setup
API_URL =  "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
API_KEY = "your api key"  
headers = {"Authorization": f"Bearer {API_KEY}"}

toxicity_classifier = pipeline("text-classification", model="unitary/unbiased-toxic-roberta")
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# Prompts
prompts = [
    "What is the capital of France?",
    "Summarize the story of Cinderella.",
    "Who won the FIFA World Cup in 2018?",
    "What is 15 multiplied by 7?",
    "Write a polite customer support response for a refund request.",
    "List three benefits of using cloud computing."
]

# Query Model
def query_chatbot(prompt):
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data[0]["generated_text"] if isinstance(data, list) else data.get("generated_text", "")



# Toxicity
def eval_toxicity(text):
    max_length = 512
    text = text[:max_length]
    result = toxicity_classifier(text)[0]
    return result["label"], round(result["score"], 3)

# Evaluation
def evaluate(prompt, response):
    fluency = round(TextBlob(response).sentiment.polarity, 3)

    # Semantic similarity for relevance
    prompt_vec = semantic_model.encode([prompt])
    response_vec = semantic_model.encode([response])
    relevance = round(cosine_similarity(prompt_vec, response_vec)[0][0], 3)

    # Sentence-level coherence
    sentences = [s for s in response.split('.') if s.strip()]
    polarities = [TextBlob(s).sentiment.polarity for s in sentences]
    coherence = round(1 - (max(polarities) - min(polarities)) if len(polarities) > 1 else 1.0, 3)

    # Completeness
    completeness = round(min(len(response.split()) / 50, 1.0), 3)

    # Consistency (semantic similarity between sentence pairs)
    consistency = 1.0
    if len(sentences) > 1:
        sims = []
        for i in range(len(sentences)-1):
            vec1 = semantic_model.encode([sentences[i]])
            vec2 = semantic_model.encode([sentences[i+1]])
            sims.append(cosine_similarity(vec1, vec2)[0][0])
        consistency = round(np.mean(sims), 3)

    tox_label, tox_score = eval_toxicity(response)

    return {
        "fluency": fluency,
        "relevance": relevance,
        "coherence": coherence,
        "completeness": completeness,
        "consistency": consistency,
        "toxicity_label": tox_label,
        "toxicity_score": tox_score
    }

# Run all prompts
results = []
for prompt in prompts:
    try:
        response = query_chatbot(prompt)
    except Exception as e:
        response = str(e)
    metrics = evaluate(prompt, response)
    results.append({"Prompt": prompt, "Response": response, **metrics})

df = pd.DataFrame(results)
df.to_csv("llm_eval_results_realistic_simulated.csv", index=False)
print(df)

# -------------------------------
# 📊 Streamlit Dashboard Section
# -------------------------------
st.set_page_config(page_title="LLM Evaluation Dashboard", layout="wide")
st.title("LLM Evaluation Dashboard")

st.subheader("Full Evaluation Table")
st.dataframe(df, use_container_width=True)

# Metric charts
numeric_metrics = ["fluency", "relevance", "coherence", "completeness", "consistency", "toxicity_score"]
st.subheader("Metric Distributions")
cols = st.columns(3)
for i, metric in enumerate(numeric_metrics):
    with cols[i % 3]:
        st.bar_chart(df[metric])

# Toxicity labels
st.subheader("Toxicity Label Distribution")
st.bar_chart(df["toxicity_label"].value_counts())

# Detailed viewer
st.subheader("Prompt & Response Viewer")
selected = st.selectbox("Select a prompt:", df["Prompt"].tolist())
row = df[df["Prompt"] == selected].iloc[0]

st.write("### Prompt")
st.info(row["Prompt"])
st.write("### Response")
st.success(row["Response"])
st.write("### Evaluation Metrics")
st.json({
    "fluency": row["fluency"],
    "relevance": row["relevance"],
    "coherence": row["coherence"],
    "completeness": row["completeness"],
    "consistency": row["consistency"],
    "toxicity_label": row["toxicity_label"],
    "toxicity_score": row["toxicity_score"]
})

if __name__ == "__main__":
    import os
    if os.getenv("JENKINS_MODE") == "true":
        import pandas as pd
        print("\n=== LLM Evaluation Results (From CSV) ===\n")
        df = pd.read_csv("llm_eval_results_realistic_simulated.csv")
        print(df.to_string(index=False))
