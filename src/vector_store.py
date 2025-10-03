import os
from pathlib import Path
from typing import List, Optional
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()


class VectorStoreManager:
    """
    Class defining the behaviour for the Vector Store Manager 
    (for vector embeddings and retrieval operations using ChromaDB)
    """
    
    def __init__(self, persist_directory: str = "data/processed/chroma_db"):
        """
        Constructor to initialize the vector store manager with persistence configuration
        
        @params:
            persist_directory: Path where ChromaDB will store its database files
        
        @attributes:
            persist_directory: Resolved path to the persistence directory
            embeddings: OpenAI embedding model instance for text-to-vector conversion
            vectorstore: ChromaDB vector store instance, None until loaded or created
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # initialize OpenAI embeddings model (text-embedding-ada-002 produces 1536-dim vectors)
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        )
        
        self.vectorstore: Optional[Chroma] = None
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Creates a new vector store from document collection with batch processing
        
        @params:
            documents: List of LangChain Document objects to embed
            
        @returns:
            Initialized ChromaDB vector store with all documents embedded
        """
        print("\n" + "="*60)
        print("CREATING VECTOR STORE")
        print("="*60)
        print(f"Embedding {len(documents)} documents...")
        print("This may take a few minutes...")
        
        try:
            batch_size = 100
            
            if len(documents) > batch_size:
                print(f"Processing in batches of {batch_size} to avoid API limits...")
                
                # initialize vector store with first batch
                first_batch = documents[:batch_size]
                total_batches = (len(documents) + batch_size - 1) // batch_size
                print(f"  Processing batch 1/{total_batches}...")
                
                self.vectorstore = Chroma.from_documents(
                    documents=first_batch,
                    embedding=self.embeddings,
                    persist_directory=str(self.persist_directory),
                    collection_name="automotive_reports"
                )
                
                for i in range(batch_size, len(documents), batch_size):
                    batch_num = (i // batch_size) + 1
                    print(f"  Processing batch {batch_num}/{total_batches}...")
                    
                    batch = documents[i:i + batch_size]
                    self.vectorstore.add_documents(batch)
                
            else:
                # small dataset, process all documents at once
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=str(self.persist_directory),
                    collection_name="automotive_reports"
                )
            
            print(f" Vector store created successfully")
            print(f" Persisted to: {self.persist_directory}")
            
            return self.vectorstore
            
        except Exception as e:
            print(f" Error creating vector store: {str(e)}")
            raise
    
    def load_vectorstore(self) -> Chroma:
        """
        Loads an existing vector store from disk; initializes ChromaDB with previously persisted embeddings

        @returns:
            Loaded ChromaDB vector store ready for search operations
        """
        print("\n" + "="*60)
        print("LOADING VECTOR STORE")
        print("="*60)
        
        try:
            # initialize ChromaDB client pointing to persisted directory
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
                collection_name="automotive_reports"
            )
            
            # verify store contains documents
            collection = self.vectorstore._collection
            count = collection.count()
            
            if count == 0:
                raise ValueError("Vector store is empty. Please create a new one.")
            
            print(f" Vector store loaded successfully")
            print(f" Contains {count} document chunks")
            
            return self.vectorstore
            
        except Exception as e:
            print(f" Error loading vector store: {str(e)}")
            raise
    
    def vectorstore_exists(self) -> bool:
        """
        Checks if a persisted vector store exists on disk; 
        looks for the ChromaDB SQLite database file which indicates a previously
        created and persisted vector store
        
        @returns:
            bool: True if chroma.sqlite3 file exists in persist directory, else False
        """
        chroma_db_path = self.persist_directory / "chroma.sqlite3"
        return chroma_db_path.exists()
    
    def get_or_create_vectorstore(self, documents: Optional[List[Document]] = None) -> Chroma:
        """
        Loads existing vector store if available, otherwise creates new one

        @params:
            documents: Documents to embed if creating new store
            
        @returns:
            Loaded or newly created ChromaDB vector store
        """
        if self.vectorstore_exists():
            print("Found existing vector store.")
            return self.load_vectorstore()
        else:
            if documents is None:
                raise ValueError("No existing vector store found and no documents provided to create one.")
            print("No existing vector store found. Creating new one...")
            return self.create_vectorstore(documents)
    
    def search(self, query: str, k: int = 4, filter_dict: Optional[dict] = None) -> List[Document]:
        """
        Searches for documents semantically similar to the query; 
        converts query to embedding and finds top-k most similar document chunks
        
        @params:
            query: Natural language search query
            k: Number of top results to return
            filter_dict: Metadata filters to apply
            
        @returns:
            results: List of k most similar Document objects, ordered by descending similarity
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call load_vectorstore() or create_vectorstore() first.")
        
        if filter_dict:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vectorstore.similarity_search(query=query, k=k)
        
        return results
    
    def search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Searches for documents with similarity scores included
        
        @params:
            query: Natural language search query
            k: Number of top results to return
            
        Returns:
            results: List of (Document, score) tuples, where score is a float representing cosine similarity
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
        
        results = self.vectorstore.similarity_search_with_score(query=query, k=k)
        return results
    
    def delete_vectorstore(self):
        """
        Deletes the persisted vector store from disk
        """
        import shutil
        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
            print(f" Deleted vector store at {self.persist_directory}")
        self.vectorstore = None


if __name__ == "__main__":
    """
    Test script to verify vector store functionality
    """
    from document_processor import DocumentProcessor
    
    print("Testing Vector Store...")
    
    vs_manager = VectorStoreManager()
    
    if vs_manager.vectorstore_exists():
        print("\nLoading existing vector store...")
        vectorstore = vs_manager.load_vectorstore()
        
        test_query = "BMW revenue 2023"
        print(f"\nTest search: '{test_query}'")
        results = vs_manager.search(test_query, k=3)
        
        print(f"\nTop {len(results)} results:")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. Company: {doc.metadata.get('company')}, Year: {doc.metadata.get('year')}")
            print(f"   Content preview: {doc.page_content[:200]}...")
    else:
        print("\nNo existing vector store found.")
        print("Run setup.py first to create the vector store.")