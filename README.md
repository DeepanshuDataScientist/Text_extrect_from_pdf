# Text_extrect_from_pdf

This project provides a system to upload PDF documents, process them into searchable text chunks, and use a conversational AI model to answer questions based on the content of the uploaded documents.

## Features

- **PDF Text Extraction**: Extracts text from uploaded PDF files.
- **Text Splitting**: Splits extracted text into manageable chunks for processing.
- **Vector Store Creation**: Creates and saves a vector store of text chunks using embeddings.
- **Conversational AI**: Uses a pre-trained conversational AI model to answer questions based on the processed text.
- **Interactive Widgets**: Provides a user interface for uploading PDFs and asking questions.

## Requirements

- Python 3.7+
- Required Python libraries:
  - `os`
  - `io`
  - `asyncio`
  - `nest_asyncio`
  - `PyPDF2`
  - `langchain`
  - `langchain_google_genai`
  - `dotenv`
  - `google`
  - `ipywidgets`
  - `IPython`

## Installation

1. Clone the repository or download the script files.
2. Install the required Python libraries using pip:
   ```bash
   pip install PyPDF2 nest_asyncio langchain dotenv google ipywidgets
Ensure you have access to the custom libraries langchain and langchain_google_genai. Place them in your Python environment if they are not available on PyPI.
Setup
Google API Key: Obtain a Google API key and store it in a .env file in the project directory:
plaintext
Copy code
GOOGLE_API_KEY=your_google_api_key
Environment Variables: Load environment variables from the .env file.
Usage
Run the Script: Execute the Python script in a Jupyter Notebook or any environment that supports ipywidgets.
Upload PDFs: Use the file upload widget to select and upload PDF files.
Process PDFs: Click the "Submit & Process" button to extract text and create a vector store.
Ask Questions: Enter a question in the text input field and click the "Ask Question" button to get a response based on the content of the uploaded PDFs.
Example Code
python
Copy code
# Import necessary libraries
import os
import io
import asyncio
import nest_asyncio
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import ipywidgets as widgets
from IPython.display import display

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Async function to process user input and generate response
async def async_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Function to handle user input and display response
def user_input(user_question):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    nest_asyncio.apply()
    response = loop.run_until_complete(async_user_input(user_question))
    return response

# Widgets for file upload and user input
pdf_uploader = widgets.FileUpload(accept=".pdf", multiple=True)
question_input = widgets.Text(description="Question:")
output_area = widgets.Output()
process_button = widgets.Button(description="Submit & Process")
ask_button = widgets.Button(description="Ask Question")

# Function to process PDFs and create vector store
def process_pdfs(change):
    with output_area:
        output_area.clear_output()
        files = pdf_uploader.value
        pdf_docs = [io.BytesIO(file['content']) for file in files]
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        print("Processing done.")

# Function to handle question asking
def ask_question(change):
    with output_area:
        output_area.clear_output()
        user_question = question_input.value
        if user_question:
            response = user_input(user_question)
            print(f"Reply: {response}")

# Attach functions to buttons
process_button.on_click(process_pdfs)
ask_button.on_click(ask_question)

# Display widgets
display(widgets.VBox([pdf_uploader, process_button, question_input, ask_button, output_area]))
License
This project is licensed under the MIT License. See the LICENSE file for details.
