import os
from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm import tqdm
import pdfplumber
import re


class DocumentProcessor:

    """
    Class defining the behaviour of the Document processor object;
    Splits documents into overlapping chunks to maintain semantic coherence for RAG retrieval
    """
    
    def __init__(self, data_dir: str = "data/raw", chunk_size: int = 1500, chunk_overlap: int = 300):
        """
        Constructor to initialize the document processor object with chunking parameters
        
        @params:
            data_dir: Path to directory containing company subdirectories with PDFs
            chunk_size: Max size of each text chunk in characters 
            chunk_overlap: Num of overlapping characters between consecutive chunks
        
        @attributes:
            data_dir: Resolved path to the data directory
            chunk_size: Configured chunk size
            chunk_overlap: Configured overlap size
            text_splitter: LangChain text splitter
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # initialize recursive text splitter with hierarchical separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Hierarchical splitting strategy
        )
        

    def extract_text_with_pdfplumber(self, pdf_path: Path) -> List[Dict]:
        """
        Extracts text and tables from PDF using pdfplumber library; 
        (tables are converted to pipe-separated text format and marked with [TABLE] tags
        for easier identification by the LLM)
   
        @params:
            pdf_path: Path object pointing to the PDF file to extract
            
        @returns:
            pages_data: 
                List of dictionaries, one per page, each containing: -
                    - 'page_content' (str): Combined text and table content
                    - 'page_number' (int): 1-indexed page number
        """
        pages_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # extract regular text content from page
                    text = page.extract_text() or ""
                    
                    # extract tables and convert to readable text format
                    tables = page.extract_tables()
                    table_text = ""
                    
                    if tables:
                        for table in tables:
                            # mark table boundaries for LLM recognition
                            table_text += "\n\n[TABLE]\n"
                            for row in table:
                                if row:
                                    # filter out None/empty cells and join with pipe separator
                                    row_text = " | ".join([str(cell) for cell in row if cell])
                                    table_text += row_text + "\n"
                            table_text += "[/TABLE]\n\n"
                    
                    # combine regular text with extracted table text
                    combined_text = text + table_text
                    
                    pages_data.append({
                        "page_content": combined_text,
                        "page_number": page_num + 1  # 1-indexed for human readability
                    })
                    
        except Exception as e:
            print(f"Error with pdfplumber on {pdf_path.name}: {str(e)}")
            # return None to trigger PyPDF fallback in calling function
            return None
        
        return pages_data
    

    def load_pdfs_from_directory(self, company: str) -> List[Document]:
        """
        Loads all PDF files from a specific company's directory
        
        @params:
            company: Company name matching the subdirectory name (e.g., "BMW", "Tesla", "Ford")
            
        @returns:
            documents: List of LangChain Document objects, one per page from all PDFs in the company directory
        """
        company_dir = self.data_dir / company
        documents = []
        
        # check if company directory exists
        if not company_dir.exists():
            print(f"Warning: Directory {company_dir} does not exist.")
            return documents
        
        # find all PDF files in the company directory
        pdf_files = list(company_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"Warning: No PDF files found in {company_dir}")
            return documents
        
        print(f"\nLoading PDFs for {company}...")
        for pdf_path in tqdm(pdf_files, desc=f"Processing {company}"):
            try:
                pages_data = self.extract_text_with_pdfplumber(pdf_path)
                
                if pages_data:
                    # create Document objects from pdfplumber extraction
                    for page_data in pages_data:
                        doc = Document(
                            page_content=page_data["page_content"],
                            metadata={
                                "company": company,
                                "source_file": pdf_path.name,
                                "year": self._extract_year_from_filename(pdf_path.name),
                                "page": page_data["page_number"]
                            }
                        )
                        documents.append(doc)
                    
                    print(f"  Loaded {pdf_path.name}: {len(pages_data)} pages (with tables)")
                else:
                    loader = PyPDFLoader(str(pdf_path))
                    docs = loader.load()
                    
                    # add metadata to PyPDF documents
                    for doc in docs:
                        doc.metadata.update({
                            "company": company,
                            "source_file": pdf_path.name,
                            "year": self._extract_year_from_filename(pdf_path.name)
                        })
                    
                    documents.extend(docs)
                    print(f"   Loaded {pdf_path.name}: {len(docs)} pages (fallback)")
                
            except Exception as e:
                print(f"   Error loading {pdf_path.name}: {str(e)}")
        
        return documents


    def load_all_documents(self) -> List[Document]:
        """
        Loads PDF documents from all company directories; processes files sequentially
        
        @returns:
            all_documents: Combined list of Document objects from all companies
        """
        all_documents = []
        companies = ["BMW", "Tesla", "Ford"]
        
        print("="*60)
        print("LOADING ANNUAL REPORTS")
        print("="*60)
        
        # process each company's PDFs sequentially
        for company in companies:
            docs = self.load_pdfs_from_directory(company)
            all_documents.extend(docs)
        
        print(f"\n Total documents loaded: {len(all_documents)}")
        return all_documents
    

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents into smaller overlapping chunks for better retrieval
        
        @params:
            documents: List of Document objects to split
            
        @returns:
            chunks: List of chunked Document objects; each chunk maintains the metadata from its parent document
        """
        print("\n" + "="*60)
        print("CHUNKING DOCUMENTS")
        print("="*60)
        print(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        
        chunks = []
        for doc in tqdm(documents, desc="Chunking documents"):
            # split each document into chunks
            doc_chunks = self.text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
        
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    

    def process_all_documents(self) -> List[Document]:
        """
        Executes the complete document processing pipeline for RAG   

        @returns:
            chuncks: List of chunked Document objects ready for embedding
        """
        # load all documents from all company directories
        documents = self.load_all_documents()
        
        if not documents:
            raise ValueError("No documents were loaded. Please check your data directory.")
        
        # split documents into chunks
        chunks = self.chunk_documents(documents)
        return chunks
    

    def _extract_year_from_filename(self, filename: str) -> str:
        """
        Extracts year from filename using regex pattern matching; searches for a 4-digit year pattern 
                                                                  starting with "20" (e.g., 2020-2099).
        
        @params:
            filename: Name of the PDF file
            
        @returns:
            Extracted year as string
        """
        match = re.search(r'20\d{2}', filename)
        return match.group(0) if match else "Unknown"
    

    def get_document_stats(self, documents: List[Document]) -> Dict:
        """
        Calculates statistics about the loaded documents
                
        @params:
            documents: List of Document objects to analyze
            
        @returns:
            stats: 
                Dictionary containing: -
                    - 'total_documents' (int): Total number of documents
                    - 'by_company' (Dict[str, int]): Document count per company
                    - 'by_year' (Dict[str, int]): Document count per year
        """
        stats = {
            "total_documents": len(documents),
            "by_company": {},
            "by_year": {}
        }
        
        # aggregate counts by company and year
        for doc in documents:
            company = doc.metadata.get("company", "Unknown")
            year = doc.metadata.get("year", "Unknown")
            
            stats["by_company"][company] = stats["by_company"].get(company, 0) + 1
            stats["by_year"][year] = stats["by_year"].get(year, 0) + 1
        
        return stats


if __name__ == "__main__":
    """
    Test script to verify document processor functionality
    """
    # initialize processor with default settings
    processor = DocumentProcessor()
    
    try:
        # run full processing pipeline
        chunks = processor.process_all_documents()
        stats = processor.get_document_stats(chunks)
        
        # display processing results
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
        print(f"Error: {str(e)}")