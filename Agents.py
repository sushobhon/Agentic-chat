import sys
import io

from crewai.tools import BaseTool

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Configuration --- #
VECTORSTORE_DIR = "hr_policy_vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- 1. RAG Agent with Crew.ai --- #
# Initialize embeddings and vector store once to be used by the tool
print("Loading embeddings and vector store for RAG tool...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=VECTORSTORE_DIR,
    embedding_function=embeddings
)
retiver = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 results

# Defining RAG Tools
class RetrieveHRPolicy(BaseTool):
    name: str = "RetrieveHRPolicy"
    description: str = "Retrives relevant HR policy information based on the query."

    def _run(self, query: str) -> str:
        """Run the retrieval tool to get relevant HR policy information."""
        # print(f"Retrieving information for query: {query}")
        results = retiver.invoke(query)

        if not results:
            return "No relevant information found in the vector database."
        
        context = "\n".join([doc.page_content for doc in results])
        # print("\t--- RAG Tool Finished ---")
        return f"Retrieved context: \n{context}"

# ## Checking Retriver Tool
# retrival_tool = RetrieveHRPolicy()
# print(retrival_tool._run("What is the process for approval to request time off?")) # Test the tool

class CodingAgent(BaseTool):
    name: str = "CodingAgent"
    description: str = """A coding agent that can write and execute Python code based on user queries.
    It returns the output of the code if successful, or an error message if an error occurs.
    The query must be a valid Python code block.
    """

    def _run(self, query: str) -> str:
        """Run the coding agent to write and execute Python code based on the query."""
        
        # We need to capture the output, so we redirect stdout to a string buffer.
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        try:
            # Execute the user's query as a code block.
            # The 'exec' function can run multi-line statements, including imports and function definitions.
            exec(query)
            
            # Get the output from the redirected buffer.
            output = redirected_output.getvalue()
            
            # Restore the original stdout.
            sys.stdout = old_stdout
            
            # If there's no output, we return a message indicating success.
            if not output.strip():
                return "Code executed successfully with no output."
            
            # Return the captured output.
            return output
            
        except Exception as e:
            # If any error occurs during execution, capture it and return the error message.
            # Restore the original stdout first.
            sys.stdout = old_stdout
            return f"An error occurred while executing the code: {e}"

# ## Checking Coding tools
# coding_tool = CodingAgent()
# print(coding_tool._run("print('Hello')"))
