import os
import json
#from reportlab.platypus import SimpleDocTemplate, Paragraph
#from reportlab.lib.styles import getSampleStyleSheet
from pathlib import Path
import pymupdf4llm
from langchain.text_splitter import MarkdownTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import FAISS

### Ingest PDF files
def ingest_data(file_with_path):
    try:
        md_text = pymupdf4llm.to_markdown(file_with_path)
    except Exception as e:
        print(f"Skipping {file_with_path.name}: {e}")
    return md_text

### Create chunks with some overlap
def chunk_data(md_text):
    splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([md_text])
    return chunks

### Create embedding and vector store from the specified embedding model
def embed_data(chunks, embedding_model):
    embeddings = OllamaEmbeddings(model=embedding_model)
    vectore_store = FAISS.from_documents(chunks, embeddings)
    return vectore_store, embeddings

### Persist vectore in local fine system, so that for the subsequent run we don't 
### have to create vector store and can be loaded for retriever; hence improving performance significantly
def persist_vector_store(vectore_store, local_vector_store_path):
    vectore_store.save_local(local_vector_store_path)
    print(f"FAISS vector store saved to: {local_vector_store_path}")

### Create retriever from RAG that will be needed to create final prompt for LLM
def get_RAG_retriever(vectore_store):
    retriever = vectore_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    return retriever

### Get the professionally built resume as mode response
def fetch_model_response(model_name, prompt):
    llm = ChatOllama(model=model_name, temperature=0.3, stream=True)
    response = llm.invoke(prompt)
    return response

### Save model respose (i.e. resume) locally
def save_model_response(folder_path: str, file_name: str, content: str):
    """
    Saves text content into a file inside the specified folder.
    Creates the folder if it doesn't exist.
    """
    try:
     # Ensure the folder exists (creates intermediate dirs too)
     os.makedirs(folder_path, exist_ok=True)

     # Construct the full file path
     file_path = os.path.join(folder_path, file_name)
     # Use context manager for safe writing
     with open(file_path, "w", encoding="utf-8") as f:
         f.write(content)
     print(f"File saved successfully: {file_path}")
    
    except Exception as e:
        print(f"Could not save file : {file_path}: {e}")
    return file_path

###--------------------------------------------------------------

##### main function #####
if __name__ == "__main__":

    #### 1) ingest the pdfs from local folder
    data_path = "./data/"
    ext = "*.pdf"

    # get the pdf file's location
    data_folder = Path(data_path)
    if not data_folder.exists() or not data_folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {data_path}")
    files = list(data_folder.glob(f"*{ext}"))

    # main loop to iterate over all 10 resumes
    for file_with_path in files:
        
        print(f"Ingesting file: {file_with_path} --- started")
        md_text = ingest_data(file_with_path)
        print(f"Ingesting file: {file_with_path} --- finished") 
        
        #### 2) chunk each pdf
        print(f"Chunking file --- started")
        chunks = chunk_data(md_text)
        print(f"Chunking file --- finished")

        #### 3) embed each pdf
        print(f"Embedding text --- started")
        embedding_model = "nomic-embed-text"
        vectore_store, embeddings = embed_data(chunks, embedding_model)
        print(f"Embedding text --- finished")

        #### 4) persist embedding locally for faster access

        print(f"Persisting vector store locally --- started")

        LOCAL_VECTOR_STORE_PATH = "vector_store_" + file_with_path.stem

        # check if local vector store exists for this pdf, if yes, then load it from there else save it there
        if os.path.exists(LOCAL_VECTOR_STORE_PATH) and os.listdir(LOCAL_VECTOR_STORE_PATH):
            print(f"Local vector store found at: {LOCAL_VECTOR_STORE_PATH}")
            vectore_store = FAISS.load_local(
                LOCAL_VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            print(f"Local vector store not found at: {LOCAL_VECTOR_STORE_PATH}")
            persist_vector_store(vectore_store, LOCAL_VECTOR_STORE_PATH)

        print(f"Persisting vector store locally --- finished")

        #### 5) buid a question for the prompt
        question = "Extract name, contact, education, experience, skills, achievements etc... from the resume."    

        #### 6) get retriever for each pdf from the RAG, for the given question/prompt
        print(f"Fetching RAG etriever doc  --- started")

        retriever = get_RAG_retriever(vectore_store)
        retrieved_docs = retriever.invoke(question)

        print(f"Fetching RAG etriever doc  --- finished")

        #### 7) create prompt template

        print(f"Creating 1st prompt template and invoking LLM model with retriever doc  --- started")

        first_prompt = PromptTemplate(
            template="""
                You are a helpful AI assistant.
                Answer ONLY from the provided pdf context.
                If the context is insufficient, just say you don't know.
                {context}
                Question: {question}
            """,
            input_variables=["context", "question"],
        )

        #### 8) call model with prompt containg RAG context (i.e retrieved_docs) and original question
        model_name = "gemma3"
        first_prompt_invoked = first_prompt.invoke({"context": retrieved_docs, "question": question})
        first_prompt_model_response = fetch_model_response(model_name, first_prompt_invoked)

        print(f"Creating 1st prompt template and invoking LLM model with retriever doc  --- finished")
        

        #### 9) fetch LLM response for user provided job posting

        print(f"Fetch LLM response for user provided job posting  --- started")

        second_job_posting_prompt = "Looking for experienced JAVA Software Engineer with Bachelor’s Degree in Computer Science."
        model_name="llama3.2"
        second_job_posting_prompt_response = fetch_model_response(model_name, second_job_posting_prompt)
        json_job_posting_obj = json.dumps(second_job_posting_prompt_response.content)

        #### 10) revised prompt template for choosing right resume based on job posting
        third_prompt = PromptTemplate(
            template="""
                Answer ONLY from the provided context.
                If the context is insufficient, just say you don't know.
                {context1}
                {context2}
                Question: {question}
            """,
            input_variables=["context1", "context2", "question"],
        )
        model_name = "llama3.2"
        third_prompt_invoked = third_prompt.invoke({"context1": first_prompt_model_response, "context2": json_job_posting_obj, "question": question})
        third_prompt_model_response = fetch_model_response(model_name, third_prompt_invoked)
        print(third_prompt_model_response.content)

        print(f"Fetch LLM response for user provided job posting  --- finished")

        
        #### 11) save revised resume as model final output into 'output' dir as pdf

        print(f"Save revised resume as per the job posting in pdf format --- started")
        output_folder = "output"
        output_file_ext = ".txt"
        output_file_path = save_model_response(output_folder, file_with_path.stem + output_file_ext, third_prompt_model_response.content)
        print("model resonse location:-", output_file_path)

        print(f"Save revised resume as per the job posting in pdf format --- finished")
