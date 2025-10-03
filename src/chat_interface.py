import os
import sys
from typing import Union
from colorama import Fore, Style, init

try:
    from rag_engine import RAGEngine
except ImportError:
    RAGEngine = None

from vector_store import VectorStoreManager

init(autoreset=True)



class ChatInterface:

    """
    Class defining the behaviour and the UI for the RAG system; terminal-based chat
    """

    def __init__(self, rag_engine: RAGEngine): # type: ignore
        """
        Constructor to initialize the chat interface
        
        @params:
            rag_engine: Initialized instance of the RAG engine with loaded vector store
        
        @attributes:
            rag: RAG engine used for query processing based on provided context
            running: Boolean flag to check whether chat loop is active
        """
        self.rag = rag_engine
        self.running = False
    

    def print_header(self):
        """
        Prints application header information; plus commands list
        """
        os.system('clear' if os.name == 'posix' else 'cls')
        print(Fore.CYAN + "="*70)
        print(Fore.CYAN + "  RAG AUTOMOTIVE ANALYSIS SYSTEM")
        print(Fore.CYAN + "  Query BMW, Tesla, and Ford Annual Reports")
        print(Fore.CYAN + "="*70)
        print()
        print(Fore.YELLOW + "Commands:")
        print(Fore.YELLOW + "  • Type your question and press Enter")
        print(Fore.YELLOW + "  • 'exit', 'quit', or 'q' to close")
        print(Fore.YELLOW + "  • 'clear' to clear conversation history")
        print(Fore.YELLOW + "  • 'examples' to see example questions")
        print(Fore.YELLOW + "  • 'help' for more information")
        print(Fore.CYAN + "="*70)
        print()
    

    def print_examples(self):
        """
        Prints example questions; grouped by different query types (for user guidance)
        """
        print(Fore.CYAN + "\n" + "="*70)
        print(Fore.CYAN + "EXAMPLE QUESTIONS")
        print(Fore.CYAN + "="*70)
        
        examples = [
            ("Simple Queries", [
                "What was BMW's total revenue in 2023?",
                "How much revenue did Tesla generate in 2023?",
                "What was Ford's revenue for the year 2020?",
            ]),
            ("Comparison Queries", [
                "Between Tesla and Ford, which company achieved higher profits in 2022?",
                "Compare BMW and Tesla revenue in 2023",
                "Which company had better profitability in 2022?",
            ]),
            ("Trend & Summary Queries", [
                "What were the growth trends for BMW from 2020 to 2023?",
                "Provide a summary of revenue for all companies over the past three years",
                "How has Tesla's profit changed from 2022 to 2023?",
            ]),
            ("Qualitative Queries", [
                "What key economic factors influenced Ford's performance in 2021?",
                "Which Tesla product is currently in development?",
                "What are BMW's strategic priorities for 2023?",
            ])
        ]
        
        for category, questions in examples:
            print(f"\n{Fore.GREEN}{category}:")
            for i, q in enumerate(questions, 1):
                print(f"  {i}. {q}")
        
        print(Fore.CYAN + "\n" + "="*70 + "\n")
    

    def print_help(self):
        """
        Prints support information and query tips (for user guidance)
        """
        print(Fore.CYAN + "\n" + "="*70)
        print(Fore.CYAN + "HELP & TIPS")
        print(Fore.CYAN + "="*70)
        print(f"""
            {Fore.GREEN}How to use:
            • Ask questions in natural language
            • Be specific about company names and years
            • You can ask follow-up questions based on previous answers
            
            {Fore.GREEN}Supported companies:
            • BMW (Reports: 2021, 2022, 2023)
            • Tesla (Reports: 2022, 2023)
            • Ford (Reports: 2021, 2022, 2023)

            {Fore.GREEN}Types of questions you can ask:
            • Financial metrics (revenue, profit, EBITDA, etc.)
            • Comparisons between companies or years
            • Trends and growth analysis
            • Qualitative information (strategies, risks, products)
            
            {Fore.GREEN}Tips for best results:
            • Mention specific years when asking about financial data
            • Use company names (BMW, Tesla, Ford) clearly
            • For comparisons, specify both companies and the year
            • Ask one question at a time for clearer answers

            {Fore.YELLOW}Note: Answers are based solely on the annual reports provided.
        """)
        print(Fore.CYAN + "="*70 + "\n")
    

    def format_answer(self, answer: str, sources: str) -> str:
        """
        Formats the model-generated answer for terminal display
        
        @params:
            answer: Final query answer text from the RAG engine
            sources: Source information string
        
        @returns:
            output: Final output string with answer and referenced sources
        """
        output = f"\n{Fore.GREEN}Answer:\n"
        output += f"{Fore.WHITE}{answer}\n"
        output += f"{Fore.CYAN}{sources}\n"
        return output
    

    def process_query(self, user_input: str) -> bool:
        """
        Processes user input (for RAG pipeline) and handles different command scenarios
        
        @params:
            user_input: Raw user input string parsed from terminal
        
        @returns:
            bool: True to continue the chat, False to exit
        """
        user_input = user_input.strip()
        
        # handle exit commands
        if user_input.lower() in ['exit', 'quit', 'q']:
            print(f"\n{Fore.YELLOW}Thank you for using RAG Automotive Analysis. Goodbye!")
            return False
        
        # handle clear command
        if user_input.lower() == 'clear':
            self.rag.clear_history()
            self.print_header()
            print(f"{Fore.GREEN} Conversation history cleared\n")
            return True
        
        # handle examples command
        if user_input.lower() == 'examples':
            self.print_examples()
            return True
        
        # handle help command
        if user_input.lower() == 'help':
            self.print_help()
            return True
        
        # handle empty input
        if not user_input:
            return True
        
        # process the query through RAG engine
        print(f"{Fore.YELLOW}Searching and analyzing...")
        
        result = self.rag.query(user_input)
        
        if result["success"]:
            sources = self.rag.format_sources(result["source_documents"])
            print(self.format_answer(result["answer"], sources))
        else:
            print(f"\n{Fore.RED}Error: {result['answer']}\n")
        
        return True
    

    def run(self):
        """
        Initializes the main interactive chat thread and runs it; continuously accepts user input in iterations
        """
        self.running = True
        self.print_header()
        
        print(f"{Fore.GREEN}Ready! Ask me anything about BMW, Tesla, or Ford.\n")
        print(f"{Fore.YELLOW}Type 'examples' to see sample questions or 'help' for more info.\n")
        
        while self.running:
            try:
                # fetch user input
                user_input = input(f"{Fore.CYAN}You: {Style.RESET_ALL}")
                
                # process the input
                should_continue = self.process_query(user_input)
                
                if not should_continue:
                    self.running = False
                    break
                
            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}Interrupted. Type 'exit' to quit or continue asking questions.\n")
                continue
            
            except Exception as e:
                print(f"\n{Fore.RED}Unexpected error: {str(e)}\n")
                continue
    

    def run_single_query(self, question: str, show_sources: bool = True):
        """
        Run a single query (without entering the interactive chat thread); for testing/demo purposes
        
        @params:
            question: Questions to ask the RAG system
            show_sources: Boolean to check Whether to display source documents; defaults to True
        """
        print(f"{Fore.CYAN}Question: {Fore.WHITE}{question}")
        print(Fore.YELLOW + "Searching and analyzing...")
        
        result = self.rag.query(question)
        
        if result["success"]:
            print(f"\n{Fore.GREEN}Answer:")
            print(f"{Fore.WHITE}{result['answer']}")
            
            if show_sources:
                sources = self.rag.format_sources(result["source_documents"])
                print(f"{Fore.CYAN}{sources}")
        else:
            print(f"\n{Fore.RED}Error: {result['answer']}")




def main():
    """
    Main function to initialize and run the chat interface; handles setup and other errors
    """
    try:
        # load vector store
        print("Loading vector store...")
        vs_manager = VectorStoreManager()
        vs_manager.load_vectorstore()
        
        # initialize RAG engine
        print("Initializing RAG engine...")
        
        if RAGEngine:
            rag = RAGEngine(vs_manager)
            print(f"{Fore.GREEN} Using RAG Engine")
        else:
            raise ImportError("RAG engine not available")
        
        # create and run chat interface
        chat = ChatInterface(rag)
        chat.run()
        
    except FileNotFoundError:
        print(f"{Fore.RED}Error: Vector store not found.")
        print(f"{Fore.YELLOW}Please run 'python setup.py' first.")
        sys.exit(1)
        
    except Exception as e:
        print(f"{Fore.RED}Error initializing application: {str(e)}")
        sys.exit(1)



if __name__ == "__main__":
    main()