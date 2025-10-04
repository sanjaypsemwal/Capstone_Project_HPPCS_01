from pathlib import Path
#import pymupdf, re
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import unstructured_inference

def extract_pdf_text(path):
   
   # 1) Ingest data


   pdf_loader = UnstructuredPDFLoader(file_path=path) 
   pdf_data = pdf_loader.load()
   print("pdf_data", pdf_data) 

  
   #page_list = pages.to_raw_data()

   #pages_final = " ".join(chunk["text"] for chunk in pages)
   #print(pages_final)
       
   # 2) chunking - splitter

   #splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
   #chunks = splitter.create_documents(pages_final)
   #print(len(chunks))
   #print(chunks[0:100])

   # 3) Embedding
   
   #embeddings = OllamaEmbeddings(model="nomic-embed-text") # model nomic-embed-text is for embedding only
   #vectore_store = FAISS.from_documents(chunks, embeddings) 

   #print("Vectore_store", vectore_store)

   #print(vectore_store.index_to_docstore_id)
   #save embedding locally
   #embedding_folder_path="cv_faiss_vector_store_index"
   #vectore_store.save_local(embedding_folder_path)

   #now load the inxed
   #new_vector_store = FAISS.load_local(
   # embedding_folder_path, 
   # embeddings, 
   # allow_dangerous_deserialization=True
   # ) 
   
   #print("new_vector_store", new_vector_store.index_to_docstore_id)

   #print(new_vector_store.get_by_ids(['80284677-cd0c-47f9-9ac3-0640dfeef0e3']))
  

if __name__ == "__main__":
    local_file_path = Path("./data/Ram_Lal.pdf")
    pages = extract_pdf_text(local_file_path)
    #print(pages)

