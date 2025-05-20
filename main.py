import argparse
import os
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path)
from chatbot.server import chat_cli, chat_gradio

def main() -> None:
    
    parser = argparse.ArgumentParser(description="Korea YouthPolicy Chatbot")
    parser.add_argument("--cli", action="store_true", help="CLI 모드 실행")
    parser.add_argument("--host", default="localhost", help="Gradio host (기본 localhost)")
    parser.add_argument("--port", default=7860, type=int, help="Gradio port (기본 7860)")
    args = parser.parse_args()

    if args.gradio:
        chat_gradio(host=args.host, port=args.port)
    else:
        chat_cli()


if __name__ == "__main__":
    main()
