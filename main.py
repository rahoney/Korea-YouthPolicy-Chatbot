import argparse
from chatbot.server import chat_cli, chat_gradio


def main() -> None:
    parser = argparse.ArgumentParser(description="Korea YouthPolicy Chatbot")
    parser.add_argument("--gradio", action="store_true", help="Gradio 웹 UI 실행")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio host (기본 0.0.0.0)")
    parser.add_argument("--port", default=7860, type=int, help="Gradio port (기본 7860)")
    args = parser.parse_args()

    if args.gradio:
        chat_gradio(host=args.host, port=args.port)
    else:
        chat_cli()


if __name__ == "__main__":
    main()
