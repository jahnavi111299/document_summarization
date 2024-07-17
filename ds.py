import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from rouge_score import rouge_scorer
import base64
import os
import torch 

# Model and tokenizer loading
model_path = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(model_path)
base_model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=False, device_map='auto', torch_dtype=torch.float32)

# File loader and preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts += text.page_content
    return final_texts

# LLM pipeline
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50
    )
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

# Evaluate ROUGE scores
def evaluate_rouge(generated_summary, reference_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(generated_summary, reference_summary)
    return scores

# Function to display the PDF of a given file
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit code
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization using LaMini FlanT5")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            filepath = os.path.join("data", uploaded_file.name)
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded File")
                displayPDF(filepath)

            with col2:
                summary = llm_pipeline(filepath)
                st.info("Summarization Complete")
                st.success(summary)

                # Example reference summary
                reference_summary = "Your reference summary here."
                rouge_scores = evaluate_rouge(summary, reference_summary)
                st.info("ROUGE Scores:")
                for metric, score in rouge_scores.items():
                    st.write(f"{metric}: {score}")

if __name__ == "__main__":
    main()
