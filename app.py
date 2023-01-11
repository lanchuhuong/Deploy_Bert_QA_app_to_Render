from typing import Dict

import pandas as pd
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, BertForQuestionAnswering, pipeline

from io import BytesIO

st.title("Question-Answering Webapp")


@st.cache(allow_output_mutation=True)
def extract_text_from_pdfs(pdf_files):
    # Create an empty data frame
    df = pd.DataFrame(columns=["file", "text"])
    # Iterate over the PDF files
    for pdf_file in pdf_files:
        # Open the PDF file
        # with open(pdf_file.read(), "rb") as f:
        with BytesIO(pdf_file.read()) as f:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(f)
            # Get the number of pages in the PDF
            num_pages = len(pdf_reader.pages)
            # Initialize a string to store the text from the PDF
            text = ""
            # Iterate over all the pages
            for page_num in range(num_pages):
                # Get the page object
                page = pdf_reader.pages[page_num]
                # Extract the text from the page
                page_text = page.extract_text()
                # Add the page text to the overall text
                text += page_text
            # Add the file name and the text to the data frame
            df = df.append({"file": pdf_file.name, "text": text}, ignore_index=True)
    # Return the data frame
    return df


def preprocess_text(text_list):
    # Initialize a empty list to store the pre-processed text
    processed_text = []
    # Iterate over the text in the list
    for text in text_list:
        num_words = len(text.split(" "))
        if num_words > 10:  # only include sentences with length >10
            processed_text.append(text)
    # Return the pre-processed text
    return processed_text


def remove_short_sentences(df):
    df["sentences"] = df["sentences"].apply(preprocess_text)
    return df


@st.cache(allow_output_mutation=True)
def get_relevant_texts(df, topic):
    model_embedding = SentenceTransformer("all-MiniLM-L6-v2")
    model_embedding.save("all-MiniLM-L6-v2")
    cosine_threshold = 0.3  # set threshold for cosine similarity value
    queries = topic  # search query
    results = []
    for i, document in enumerate(df["sentences"]):
        sentence_embeddings = model_embedding.encode(document)
        query_embedding = model_embedding.encode(queries)
        for j, sentence_embedding in enumerate(sentence_embeddings):
            distance = cosine_similarity(
                sentence_embedding.reshape((1, -1)), query_embedding.reshape((1, -1))
            )[0][0]
            sentence = df["sentences"].iloc[i][j]
            results += [(i, sentence, distance)]
    results = sorted(results, key=lambda x: x[2], reverse=True)
    del model_embedding

    texts = []
    for idx, sentence, distance in results:
        if distance > cosine_threshold:
            text = sentence
            texts.append(text)
    # turn the list to string
    context = "".join(texts)
    return context


@st.cache(allow_output_mutation=True)
def get_pipeline():
    modelname = "deepset/bert-base-cased-squad2"
    model_qa = BertForQuestionAnswering.from_pretrained(modelname)
    # model_qa.save_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained("tokenizer-deepset")
    # tokenizer.save_pretrained("tokenizer-" + modelname)
    qa = pipeline("question-answering", model=model_qa, tokenizer=tokenizer)
    return qa


def answer_question(pipeline, question: str, context: str) -> Dict:
    input = {"question": question, "context": context}
    return pipeline(input)


@st.cache(allow_output_mutation=True)
def create_context(df):
    # path = "data/"
    # files = Path(path).glob("WHR+22.pdf")
    # df = extract_text_from_pdfs(files)
    df["sentences"] = df["text"].apply(
        lambda long_str: long_str.replace("\n", " ").split(".")
    )
    df = remove_short_sentences(df)

    context = get_relevant_texts(df, topic)
    return context


@st.cache(allow_output_mutation=True)
def start_app():
    with st.spinner("Loading model. Please hold..."):
        # context = create_context()
        pipeline = get_pipeline()
    return pipeline


pdf_files = st.file_uploader(
    "Upload pdf files", type=["pdf"], accept_multiple_files=True
)

if pdf_files:
    with st.spinner("processing pdf..."):
        df = extract_text_from_pdfs(pdf_files)
        # context = create_context(df)
        # del df
    topic = st.text_input("Enter the topic you want to ask here")
    question = st.text_input("Enter your questions here...")

    if question != "":
        # pipeline = get_pipeline()
        with st.spinner("Searching. Please hold..."):
            context = create_context(df)
            qa_pipeline = start_app()
            answer = answer_question(qa_pipeline, question, context)
            st.write(answer)
        del qa_pipeline
        del context
