# chatbot/server.py
from __future__ import annotations
import argparse, os

from chatbot.llm_chain import build_chain

# LLM + Retriever 체인 (top-k=5)
qa_chain = build_chain(k=5)

# ────────────────────────────────────────────────
# 1) CLI 모드
# ----------------------------------------------
def chat_cli() -> None:
    print("Korea YouthPolicy Chatbot ─ Exit input with 'q'")
    while True:
        q = input("질문> ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break
        print("\n", qa_chain.run(q), "\n")


# ────────────────────────────────────────────────
# 2) Gradio Web UI
# ----------------------------------------------
def chat_gradio(host: str = "0.0.0.0", port: int = 7860) -> None:
    import gradio as gr

    def _gr_answer(msg: str) -> str:
        return qa_chain.run(msg)

    with gr.Blocks(title="Korea YouthPolicy Chatbot") as demo:
        gr.Markdown("# 한국 청년정책 챗봇")
        chat = gr.ChatInterface(fn=_gr_answer)
    demo.queue().launch(server_name=host, server_port=port, share=False)


# ────────────────────────────────────────────────
# 3) entrypoint
# ----------------------------------------------
def serve() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradio", action="store_true", help="웹 UI 실행")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()

    if args.gradio:
        chat_gradio(host=args.host, port=args.port)
    else:
        chat_cli()


if __name__ == "__main__":
    serve()
