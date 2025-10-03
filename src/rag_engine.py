import os
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

from vector_store import VectorStoreManager

load_dotenv()


class RAGEngine:
    """
    Class defining the behaviour of the RAG Engine
    """
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        """
        Constructor to initialize the RAG engine with vector store and LLM components
        
        @params:
            vector_store_manager: Initialized vector store manager with loaded vector store
        
        @attributes:
            vs_manager: Vector store manager for retrieval
            llm: Language model for answer generation
            memory: Stores conversation history for context
            qa_prompt: Custom prompt template for financial queries
            qa_chain: Complete RAG pipeline chain
        """
        self.vs_manager = vector_store_manager
        
        if self.vs_manager.vectorstore is None:
            raise ValueError("Vector store must be loaded before initializing RAG engine")
        
        # initialize LLM with deterministic settings for factual accuracy
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0,  # Deterministic responses for consistent financial data
        )
        
        # initialize conversation memory to support follow-up questions
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True, 
            output_key="answer"  
        )
        
        self.qa_prompt = self._create_qa_prompt()
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vs_manager.vectorstore.as_retriever(
                search_kwargs={"k": 15}                     # retrieve top 15 most relevant chunks
            ),
            memory=self.memory,
            return_source_documents=True,                   # include source docs for attribution
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            verbose=False
        )
    
    def _create_qa_prompt(self) -> PromptTemplate:
        """
        Creates specialized prompt template optimized for financial data extraction
    
        @returns:
            PromptTemplate: LangChain prompt template with context and question variables
        """
        template = """You are an expert financial analyst specializing in the automotive industry. You are analyzing annual reports from BMW, Tesla, and Ford.

Context from annual reports (including tables marked with [TABLE] tags):
{context}

CRITICAL INSTRUCTIONS:
1. Look for financial data in TABLES (marked with [TABLE]...[/TABLE]) as well as in regular text
2. Revenue and profit figures are often in tables - examine them carefully
3. When you find relevant financial data, always include:
   - The exact number/amount
   - The currency (e.g., millions, billions, EUR, USD)
   - The company name
   - The year
4. For comparison questions, retrieve data for ALL companies mentioned
5. If you find partial data, provide what you have and specify what's missing
6. Only say "I don't have that information" if you've thoroughly checked all context and found nothing relevant
7. Tables may use abbreviations: m = million, bn = billion, € = EUR, $ = USD

Common financial terms to look for:
- Revenue: "Total revenue", "Revenues", "Net sales", "Total net sales"
- Profit: "Net income", "Net profit", "Profit attributable", "EBIT", "EBITDA", "Earnings"
- Growth: "Change", "Increase", "Decrease", "Growth rate", "% change"

Question: {question}

Provide a clear, specific answer with exact figures when available:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def query(self, question: str) -> Dict[str, any]:
        """
        Executes a query using multi-strategy retrieval and answer generation
        
        @params:
            question: User's natural language question
            
        @returns:
            results: 
                Dictionary containing: -
                    - 'answer' (str): Generated answer text
                    - 'source_documents' (List[Document]): Retrieved document chunks used
                    - 'success' (bool): Check whether query executed successfully
        """
        results = self._multi_strategy_query(question)
        
        return results
    
    def _multi_strategy_query(self, question: str) -> Dict[str, any]:
        """
        Tries multiple query strategies to maximize retrieval success
        
        Strategy 1: Try the original query unmodified
        Strategy 2: Else, expand query with financial terminology synonyms
        
        @params:
            question: Original user question
            
        @returns:
            Query result dictionary with answer, sources, and success flag
        """
        try:
            result = self.qa_chain({"question": question})
            
            answer = result["answer"].lower()
            if "don't have that information" not in answer or len(result.get("source_documents", [])) > 0:
                return {
                    "answer": result["answer"],
                    "source_documents": result.get("source_documents", []),
                    "success": True
                }
            
            expanded_question = self._expand_financial_query(question)
            if expanded_question != question:
                print(f" Trying expanded query...")
                result = self.qa_chain({"question": expanded_question})
                
                return {
                    "answer": result["answer"],
                    "source_documents": result.get("source_documents", []),
                    "success": True
                }
            
            return {
                "answer": result["answer"],
                "source_documents": result.get("source_documents", []),
                "success": True
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "source_documents": [],
                "success": False
            }
    
    def _expand_financial_query(self, question: str) -> str:
        """
        Expands financial queries by appending synonym alternatives
        
        @params:
            question: Original user question
            
        @returns:
            expanded: Expanded question with synonyms appended (or original if no expansion needed)
        """
        expansions = {
            "revenue": "revenue total sales net sales turnover",
            "profit": "profit net income earnings EBIT EBITDA net profit",
            "growth": "growth increase change trend performance",
        }
        
        expanded = question
        for term, alternatives in expansions.items():
            if term in question.lower():
                expanded = question + f" (including {alternatives})"
                break
        
        return expanded
    
    def query_with_filter(self, question: str, company: Optional[str] = None, 
                         year: Optional[str] = None) -> Dict[str, any]:
        """
        Executes a query with metadata filtering to narrow search scope
        
        @params:
            question: User's natural language question
            company: Filter to specific company 
            year: Filter to specific year
            
        @returns:
            Query result dictionary with answer, sources, success flag
        """
        # build metadata filter dictionary
        filter_dict = {}
        if company:
            filter_dict["company"] = company
        if year:
            filter_dict["year"] = year
        
        if filter_dict:
            original_search_kwargs = self.qa_chain.retriever.search_kwargs
            self.qa_chain.retriever.search_kwargs = {
                "k": 15,
                "filter": filter_dict
            }
        
        try:
            result = self.qa_chain({"question": question})
            return {
                "answer": result["answer"],
                "source_documents": result.get("source_documents", []),
                "success": True
            }
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "source_documents": [],
                "success": False
            }
        finally:
            if filter_dict:
                self.qa_chain.retriever.search_kwargs = original_search_kwargs
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Retrieve the current conversation history
        
        @returns:
            history: 
                List of message dictionaries, each containing: -
                    - 'role' (str): Message role ('human' or 'ai')
                    - 'content' (str): Message content text
        """
        messages = self.memory.chat_memory.messages
        history = []
        
        # convert message objects to dictionaries
        for msg in messages:
            if hasattr(msg, 'type'):
                history.append({
                    "role": msg.type,
                    "content": msg.content
                })
        
        return history
    
    def clear_history(self):
        """
        Clear the conversation history from memory
        """
        self.memory.clear()
    
    def format_sources(self, source_documents: List[Document]) -> str:
        """
        Formats source documents into a readable attribution string; groups source chunks by company and year
        
        @params:
            source_documents: List of retrieved Document objects
            
        @returns:
            sources_text: Formatted string displaying sources by company, year, file, chunk count
        """
        if not source_documents:
            return "No sources available"
        
        sources_text = "\n\nSources:\n"
        sources_text += "=" * 60 + "\n"
        
        sources_by_company = {}
        for doc in source_documents:
            company = doc.metadata.get("company", "Unknown")
            year = doc.metadata.get("year", "Unknown")
            source = doc.metadata.get("source_file", "Unknown")
            
            key = f"{company} - {year}"
            if key not in sources_by_company:
                sources_by_company[key] = {
                    "company": company,
                    "year": year,
                    "source": source,
                    "count": 0
                }
            sources_by_company[key]["count"] += 1
        
        for i, (key, info) in enumerate(sources_by_company.items(), 1):
            sources_text += f"{i}. {info['company']} Annual Report {info['year']}\n"
            sources_text += f"   File: {info['source']}\n"
            sources_text += f"   Chunks referenced: {info['count']}\n"
        
        return sources_text
    
    def analyze_query_intent(self, question: str) -> Dict[str, any]:
        """
        Analyzes user query to extract entities and classify intent type
        
        @params:
            question: User's natural language question
            
        @returns:
            analysis: 
                Dictionary containing: -
                    - 'companies' (List[str]): Detected company names (uppercase)
                    - 'years' (List[str]): Detected years as strings
                    - 'metrics' (List[str]): Detected metric types
                    - 'query_type' (str): Classified query type
        """
        question_lower = question.lower()
        
        analysis = {
            "companies": [],
            "years": [],
            "metrics": [],
            "query_type": "general"
        }
        
        companies = ["bmw", "tesla", "ford"]
        for company in companies:
            if company in question_lower:
                analysis["companies"].append(company.upper())
        
        import re
        years = re.findall(r'20[2-4][0-9]', question)
        analysis["years"] = list(set(years))  
        
        metrics = {
            "revenue": ["revenue", "sales", "turnover"],
            "profit": ["profit", "earnings", "net income", "ebitda", "ebit"],
            "growth": ["growth", "increase", "trend"],
            "performance": ["performance", "results"]
        }
        
        for metric_name, keywords in metrics.items():
            if any(keyword in question_lower for keyword in keywords):
                analysis["metrics"].append(metric_name)
        
        # classify query type based on keywords and entity patterns
        if "compare" in question_lower or "between" in question_lower or "versus" in question_lower:
            analysis["query_type"] = "comparison"
        elif "trend" in question_lower or "over" in question_lower or "growth" in question_lower:
            analysis["query_type"] = "trend"
        elif "summary" in question_lower or "overview" in question_lower:
            analysis["query_type"] = "summary"
        elif len(analysis["companies"]) == 1 and len(analysis["years"]) == 1:
            analysis["query_type"] = "specific"
        
        return analysis


if __name__ == "__main__":
    """
    Test script to verify RAG engine functionality
    """
    print("Testing RAG Engine...")
    
    try:
        vs_manager = VectorStoreManager()
        vs_manager.load_vectorstore()
        
        rag = RAGEngine(vs_manager)
        
        test_question = "What was BMW's revenue in 2023?"
        print(f"\nQuestion: {test_question}")
        print("-" * 60)
        
        result = rag.query(test_question)
        
        if result["success"]:
            print(f"Answer: {result['answer']}")
            print(rag.format_sources(result['source_documents']))
        else:
            print(f"Error: {result['answer']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nMake sure to run setup.py first to create the vector store.")