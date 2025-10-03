import sys
from pathlib import Path
from colorama import Fore, Style, init

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vector_store import VectorStoreManager
from rag_engine import RAGEngine
from chat_interface import ChatInterface

init(autoreset=True)


def run_test_queries():
    """
    Runs all test queries (from the sample questions provided)
    """
    test_questions = [
        "What was BMW's total revenue in 2023?",
        "How much revenue did Tesla generate in 2023?",
        "What was Ford's revenue for the year 2020?",
        "Can you provide the revenue figures for BMW in 2017?",
        "What key economic factors influenced Ford's performance in 2021?",
        "Which Tesla product is currently in the development stage?",
        "What were BMW's profit figures for 2020 and 2023?",
        "Between Tesla and Ford, which company achieved higher profits in 2022?",
        "What were Tesla's profit numbers for 2022 and 2023?",
        "Which company recorded better profitability in 2022 overall?",
        "Provide a summary of revenue figures for Tesla, BMW, and Ford over the past three years.",
        "What were the growth trends for BMW's financial performance from 2020 to 2023?",
    ]
    
    try:
        print(Fore.CYAN + "="*70)
        print(Fore.CYAN + "RAG AUTOMOTIVE ANALYSIS - TEST QUERIES")
        print(Fore.CYAN + "="*70)
        print()
        
        print(Fore.YELLOW + "Initializing RAG system...")
        vs_manager = VectorStoreManager()
        vs_manager.load_vectorstore()
        rag = RAGEngine(vs_manager)
        chat = ChatInterface(rag)
        
        print(Fore.GREEN + " System initialized successfully\n")
        print(Fore.CYAN + "="*70)
        print(f"Running {len(test_questions)} test queries...")
        print(Fore.CYAN + "="*70 + "\n")
        
        # execute each test query and collect results
        results = []
        for i, question in enumerate(test_questions, 1):
            print(f"\n{Fore.CYAN}{'='*70}")
            print(f"{Fore.CYAN}Query {i}/{len(test_questions)}")
            print(f"{Fore.CYAN}{'='*70}\n")
            
            # run query and display results
            chat.run_single_query(question, show_sources=True)
            
            # store result metrics for summary
            result = rag.query(question)
            results.append({
                "question": question,
                "success": result["success"],
                "answer_length": len(result["answer"]) if result["success"] else 0,
                "has_data": "don't have that information" not in result["answer"].lower()
            })
            
            print()
        
        # print summary statistics
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}TEST SUMMARY")
        print(f"{Fore.CYAN}{'='*70}\n")
        
        successful = sum(1 for r in results if r["success"])
        with_data = sum(1 for r in results if r["has_data"])
        failed = len(results) - successful
        
        print(f"{Fore.GREEN}Successful queries: {successful}/{len(results)}")
        print(f"{Fore.GREEN}Queries with actual data: {with_data}/{len(results)}")
        if failed > 0:
            print(f"{Fore.RED}Failed queries: {failed}/{len(results)}")
        
        avg_answer_length = sum(r["answer_length"] for r in results if r["success"]) / successful if successful > 0 else 0
        print(f"\n{Fore.YELLOW}Average answer length: {avg_answer_length:.0f} characters")
        
        print(f"\n{Fore.CYAN}{'='*70}\n")
        
    except Exception as e:
        print(f"\n{Fore.RED}Error: {str(e)}")
        print(f"{Fore.YELLOW}Make sure you have run 'python setup.py' first.")
        sys.exit(1)


def run_interactive_demo():
    """
    Run a quick interactive demo showcasing conversation context
    """
    demo_questions = [
        "What was BMW's revenue in 2023?",
        "How does that compare to Tesla?",  # follow-up question using context
        "What were the main factors affecting automotive industry in 2023?",
    ]
    
    try:
        print(Fore.CYAN + "="*70)
        print(Fore.CYAN + "RAG AUTOMOTIVE ANALYSIS - INTERACTIVE DEMO")
        print(Fore.CYAN + "="*70)
        print()
        
        # initialize RAG system components
        print(Fore.YELLOW + "Initializing RAG system...")
        vs_manager = VectorStoreManager()
        vs_manager.load_vectorstore()
        rag = RAGEngine(vs_manager)
        chat = ChatInterface(rag)
        
        print(Fore.GREEN + " System initialized successfully\n")
        print(Fore.CYAN + "="*70)
        print("Running interactive demo...")
        print(Fore.CYAN + "="*70 + "\n")
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n{Fore.CYAN}Question {i}: {Fore.WHITE}{question}")
            print(Fore.YELLOW + "Thinking...")
            
            result = rag.query(question)
            
            if result["success"]:
                print(f"\n{Fore.GREEN}Answer: {Fore.WHITE}{result['answer']}")
                sources = rag.format_sources(result["source_documents"])
                print(f"{Fore.CYAN}{sources}")
            else:
                print(f"\n{Fore.RED}Error: {result['answer']}")
            
            print()
        
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.GREEN}Demo completed!")
        print(f"{Fore.YELLOW}Notice how the second question used context from the first.")
        print(f"{Fore.CYAN}{'='*70}\n")
        
    except Exception as e:
        print(f"\n{Fore.RED}Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    """
    Command-line entry point with mode selection; parses command-line arguments to determine which test mode to run
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the RAG Automotive Analysis system")
    parser.add_argument(
        "--mode",
        choices=["test", "demo"],
        default="test",
        help="Test mode: 'test' runs all questions, 'demo' runs interactive demo"
    )
    
    args = parser.parse_args()
    
    if args.mode == "test":
        run_test_queries()
    else:
        run_interactive_demo()