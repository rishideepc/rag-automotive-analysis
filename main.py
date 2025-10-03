import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vector_store import VectorStoreManager
from rag_engine import RAGEngine
from chat_interface import ChatInterface


def main():
    """
    Initialize and run the RAG application
    
    Orchestrates the startup sequence: -
    1. Loads the pre-built vector store from disk
    2. Initializes the RAG engine with the vector store
    3. Launches the interactive chat interface
    
    The vector store must already exist (created by running setup.py);
    if the vector store is not found, displays an error message and exits
    """
    try:
        # load pre-built vector store from disk
        print("Loading vector store...")
        vs_manager = VectorStoreManager()
        vs_manager.load_vectorstore()
        
        # initialize RAG engine with loaded vector store
        print("Initializing RAG engine...")
        rag = RAGEngine(vs_manager)
        
        # create and run interactive chat interface
        chat = ChatInterface(rag)
        chat.run()
        
    except FileNotFoundError:
        print("Error: Vector store not found.")
        print("Please run 'python setup.py' first to create the vector store.")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()