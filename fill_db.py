#imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

#Paths
DATA_PATH = r"C:\Users\mah19\RAG\Data Folder\VH021.pdf"
CHROMA_PATH = r"chroma_db"

#ChromaDB setup
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="growing_vegetables")

#Load Document
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

#Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
 chunk_size=350,
 chunk_overlap=100,
 length_function=len,
 is_separator_regex=False,
 )
chunks = text_splitter.split_documents(raw_documents)

#Prepare for ChromaDB
documents = [chunk.page_content for chunk in chunks]
metadata = [chunk.metadata for chunk in chunks]
ids = ["ID"+str(i) for i in range(len(chunks))]

#Store in ChromaDB
collection.upsert(
 documents=documents,
 metadatas=metadata,
 ids=ids
)

