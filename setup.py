import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))

from document_processor import DocumentProcessor
from vector_store import VectorStoreManager


def verify_environment():
    """
    Verifies that all required environment variables are properly configured
    """
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("="*60)
        print("ERROR: OpenAI API Key Not Set")
        print("="*60)
        print("Please set your OPENAI_API_KEY in the .env file")
        sys.exit(1)
    
    print("Environment variables verified")


def verify_data_directory():
    """
    Verifies that the data directory exists and contains PDF files
    """
    data_dir = Path("data/raw")
    
    # check if base data directory exists
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    # scan each company directory for PDFs
    companies = ["BMW", "Tesla", "Ford"]
    pdf_count = 0
    
    for company in companies:
        company_dir = data_dir / company
        if company_dir.exists():
            pdfs = list(company_dir.glob("*.pdf"))
            pdf_count += len(pdfs)
            if pdfs:
                print(f" Found {len(pdfs)} PDF(s) for {company}")
    
    if pdf_count == 0:
        print("\n No PDF files found!")
        sys.exit(1)
    
    print(f"\n Total PDFs found: {pdf_count}")


def main():
    print("="*60)
    print("RAG AUTOMOTIVE ANALYSIS - SETUP")
    print("="*60)
    
    print("\nStep 1: Verifying environment...")
    verify_environment()
    
    print("\nStep 2: Verifying data directory...")
    verify_data_directory()
    
    print("\nStep 3: Processing documents with table extraction...")
    try:
        # initialize processor with optimized chunking parameters
        processor = DocumentProcessor(
            chunk_size=1500,
            chunk_overlap=300
        )
        chunks = processor.process_all_documents()
        
        # prnt processing statistics
        stats = processor.get_document_stats(chunks)
        print("\n" + "="*60)
        print("DOCUMENT STATISTICS")
        print("="*60)
        print(f"Total chunks: {stats['total_documents']}")
        print(f"\nBy Company:")
        for company, count in stats['by_company'].items():
            print(f"  {company}: {count} chunks")
        print(f"\nBy Year:")
        for year, count in sorted(stats['by_year'].items()):
            print(f"  {year}: {count} chunks")
        
    except Exception as e:
        print(f" Error processing documents: {str(e)}")
        sys.exit(1)
    
    print("\nStep 4: Creating vector store...")
    try:
        vs_manager = VectorStoreManager()
        
        if vs_manager.vectorstore_exists():
            response = input("\nVector store already exists. Recreate? (y/N): ")
            if response.lower() != 'y':
                print("Keeping existing vector store.")
                print("\n" + "="*60)
                print("SETUP COMPLETE")
                print("="*60)
                return
            
            # delete existing store before recreating
            vs_manager.delete_vectorstore()
        
        vectorstore = vs_manager.create_vectorstore(chunks)
        
    except Exception as e:
        print(f" Error creating vector store: {str(e)}")
        sys.exit(1)
    
    print("\nStep 5: Verifying vector store...")
    try:
        test_queries = [
            "BMW revenue 2023",
            "Tesla profit 2023",
            "Ford financial performance 2022"
        ]
        
        print("Running test queries...")
        for query in test_queries:
            results = vs_manager.search(query, k=1)
            if results:
                print(f"   '{query}' - Found results")
        
    except Exception as e:
        print(f" Warning: Error during verification: {str(e)}")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print("\nYour RAG system is ready!")
    print("\nTo start using the system:")
    print("\n  python main.py")
    print("\nTo test with sample queries:")
    print("\n  python test_queries.py --mode test")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()