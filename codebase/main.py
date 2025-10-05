
import os
from pathlib import Path
import pymupdf4llm
from langchain.text_splitter import MarkdownTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import FAISS

def ingest_data(file_with_path):
  try:
   md_text = pymupdf4llm.to_markdown(file_with_path) 
  except Exception as e:
    print(f"Skipping {file_with_path.name}: {e}")
  #print ("md_text:-", md_text)   
  return md_text   


def chunk_data(md_text):
 splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
 chunks = splitter.create_documents([md_text])
 #print("len(chunks):-", len(chunks))
 #print("chunks:-", chunks)
 return chunks


def embed_data(chunks, embedding_model):
 embeddings = OllamaEmbeddings(model=embedding_model)
 vectore_store = FAISS.from_documents(chunks, embeddings)
 return vectore_store, embeddings

def persist_vector_store(vectore_store, local_vector_store_path):
  vectore_store.save_local(local_vector_store_path)
  print(f"FAISS vector store saved to: {local_vector_store_path}")

def get_RAG_retriever(vectore_store):
  retriever = vectore_store.as_retriever(search_type="similarity", search_kwargs={"k":4})
  return retriever  

def fetch_model_response(model_name, final_prompt):
   llm = ChatOllama(model=model_name, temperature=0.7, stream=True)
   response = llm.invoke(final_prompt)
   return response

def save_model_response(folder_path: str, file_name: str, content: str):
    """
    Saves text content into a file inside the specified folder.
    Creates the folder if it doesn't exist.
    """
    # Ensure the folder exists (creates intermediate dirs too)
    os.makedirs(folder_path, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(folder_path, file_name)

    # Use context manager for safe writing
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"File saved successfully: {file_path}")
    return file_path


##### main function #####
if __name__ == "__main__":
 
 # 1) ingest the pdfs from local folder
 data_path = "./data/"
 ext = "*.pdf"

 data_folder = Path(data_path)
 if not data_folder.exists() or not data_folder.is_dir():
  raise FileNotFoundError(f"Folder not found: {data_path}")
 files = list(data_folder.glob(f"*{ext}"))

 #main loop to iterate over all 10 resumes
 
 for file_with_path in files:
   md_text = ingest_data(file_with_path)
   
   # 2) chunk each pdf
   chunks = chunk_data(md_text)
   
   # 3) embed each pdf
   embedding_model = "nomic-embed-text"
   vectore_store, embeddings = embed_data(chunks, embedding_model)
   #print("vectore_store.index_to_docstore_id:-", vectore_store.index_to_docstore_id)
   
   # 4) persist embedding locally for faster access
  
   LOCAL_VECTOR_STORE_PATH = "vector_store_" + file_with_path.stem

   # check if local vectore store exists for this pdf, then load it from there else save it there
   if(os.path.exists(LOCAL_VECTOR_STORE_PATH) and os.listdir(LOCAL_VECTOR_STORE_PATH)):
     print(f"Local vector store found at: {LOCAL_VECTOR_STORE_PATH}")
     vectore_store = FAISS.load_local(LOCAL_VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True) 
   else:
     print(f"Local vector store not found at: {LOCAL_VECTOR_STORE_PATH}") 
     persist_vector_store(vectore_store, LOCAL_VECTOR_STORE_PATH)

   # 5) buid a question for the prompt
   question = "Please create a professional resume."

   # 5) get retriever each pdf from the RAG, for the given question/prompt
   retriever = get_RAG_retriever(vectore_store)
   retrieved_docs = retriever.invoke(question)
   
   # 6) create prompt template
   prompt = PromptTemplate(
     template="""
     You are a helpful AI assistant.
     Answer ONLY from the provided pdf context.
     If the context is insufficient, just say you don't know.
     {context}
     Question: {question}
     """,
     input_variables=["context", "question"]
    )
   
   # 6) call model with prompt containg RAG context (i.e retrieved_docs) and original question
   
   model_name="gemma3"
   final_prompt = prompt.invoke({"context": retrieved_docs, "question": question})
   
   model_response = fetch_model_response(model_name,final_prompt) 
 
 
   # 6) /save professional resume as model output into 'output' dir
   output_folder = "output"
   output_file_ext = ".txt"
   output_file_path = save_model_response(output_folder, file_with_path.stem + output_file_ext, model_response.content)
   print("model output_file_path:-", output_file_path)
