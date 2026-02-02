"""
Example usage of AI-RAG system
"""

import os
from dotenv import load_dotenv
from ai_rag import RAG

# Load environment variables
load_dotenv()


def main():
    """Main example function"""

    print("AI-RAG Web Document Q&A System")
    print("=" * 50)

    # Example 1: Programming languages comparison
    print("\n Example 1: Programming Languages")
    print("-" * 50)

    urls = [
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://en.wikipedia.org/wiki/C%2B%2B",
        "https://en.wikipedia.org/wiki/Rust_(programming_language)",
    ]

    # Initialize RAG
    rag = RAG(
        urls,
        model="Qwen/Qwen3-235B-A22B",
        temperature=0.5,
        max_tokens=256,
        chunk_size=512,
        verbose=True,
    )

    print(f"Loaded {rag.get_document_count()} documents")
    print(f"Created {rag.get_node_count()} document chunks")

    # Questions about programming languages
    questions = [
        "Which programming language is faster: Python, C++ or Rust?",
        "When was Python created and by whom?",
        "What are the main features of Rust?",
        "Is Python easy to learn for beginners?",
        "What is C++ mainly used for?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n Question {i}: {question}")
        print(" Answer:", end=" ")
        answer = rag(question)
        print(answer)

    # Example 2: Technical documentation
    print("\n\n Example 2: Technical Documentation")
    print("-" * 50)

    tech_urls = [
        "https://docs.python.org/3/tutorial/index.html",
        "https://fastapi.tiangolo.com/",
    ]

    tech_rag = RAG(
        tech_urls,
        model="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.3,
        max_tokens=512,
        chunk_size=1024,
        verbose=False,
    )

    tech_questions = [
        "How do I define a function in Python?",
        "What is FastAPI and what are its main features?",
    ]

    for i, question in enumerate(tech_questions, 1):
        print(f"\n Question {i}: {question}")
        print(" Answer:", end=" ")
        answer = tech_rag(question)
        print(answer[:200] + "..." if len(answer) > 200 else answer)

    # Example 3: Error handling
    print("\n\n Example 3: Handling Unknown Information")
    print("-" * 50)

    response = rag("What is the capital of Mars?")
    print(f" Question: What is the capital of Mars?")
    print(f" Answer: {response}")


if __name__ == "__main__":
    # Check for token
    if not os.getenv("HUGGING_FACE_TOKEN"):
        print("Warning: HUGGING_FACE_TOKEN not found in environment")
        print("Set it in .env file or export HUGGING_FACE_TOKEN='your-token'")
        print("Using demo mode with limited functionality...")

    main()
