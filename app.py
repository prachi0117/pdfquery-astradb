from langchain_openai import OpenAI
import streamlit as st
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.embeddings import OpenAIEmbeddings  # You'll replace this with Google embeddings if available
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import cassio
import os
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

# Define your Google API credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# AstraDB Connection Information
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:BOZjWDqbcSfNutpRBmCNHXTd:1a575c878f4006f30bdec818b6480a02dbf57f8dcbb7b6cece5ebbda7154ab31"
ASTRA_DB_ID = "8f3592d9-37fd-4785-b5ce-cfde84a3b750"

# Initialize Astra DB connection


# Placeholder for Google LLM and Embedding (implement using available Google services)
# llm = GoogleLLM(api_key=GOOGLE_API_KEY)
# embedding = GoogleEmbeddings(api_key=GOOGLE_API_KEY)

# Sample PdfReader code
pdfreader = PdfReader('Attention.pdf')

# Read text from PDF
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content
        
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,
)
# Text splitting
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)



astra_vector_store.add_texts(texts[:50])

st.write("Inserted %i headlines." % len(texts[:50]))

astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Streamlit interface
st.title("PDF Q&A Application")

first_question = True

while True:
    if first_question:
        query_text = st.text_input("Enter your question (or type 'quit' to exit):")
    else:
        query_text = st.text_input("What's your next question (or type 'quit' to exit):")

    if query_text.lower() == "quit":
        break

    if query_text == "":
        continue

    first_question = False

    st.write("QUESTION: \"%s\"" % query_text)
    answer = astra_vector_index.query(query_text, llm=None).strip()  # Replace with Google LLM
    st.write("ANSWER: \"%s\"" % answer)

    st.write("FIRST DOCUMENTS BY RELEVANCE:")
    for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
        st.write("[%0.4f] \"%s ...\"" % (score, doc.page_content[:84]))
