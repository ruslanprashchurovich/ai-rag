"""
Command-line interface for AI-RAG
"""

import argparse
import sys
from dotenv import load_dotenv
from ai_rag import RAG

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="AI-RAG: Web Document Q&A System")

    parser.add_argument("urls", nargs="+", help="Web URLs to process")
    parser.add_argument("--question", "-q", help="Question to ask")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-235B-A22B", help="Hugging Face model name"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="Maximum response tokens"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512, help="Document chunk size"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode"
    )

    args = parser.parse_args()

    # Initialize RAG
    try:
        rag = RAG(
            urls=args.urls,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            chunk_size=args.chunk_size,
        )

        print(f"Loaded {len(args.urls)} URLs")
        print(f"Documents: {rag.get_document_count()}, Chunks: {rag.get_node_count()}")

        # Interactive mode
        if args.interactive:
            print("\nInteractive mode. Type 'quit' to exit.\n")
            while True:
                try:
                    question = input("Question: ").strip()
                    if question.lower() in ["quit", "exit", "q"]:
                        break

                    if question:
                        print(" Answer:", end=" ")
                        answer = rag(question)
                        print(answer)
                        print()
                except KeyboardInterrupt:
                    print("\n Goodbye!")
                    break
        # Single question mode
        elif args.question:
            print(f"\n Question: {args.question}")
            print("Answer:", end=" ")
            answer = rag(args.question)
            print(answer)
        else:
            parser.print_help()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
