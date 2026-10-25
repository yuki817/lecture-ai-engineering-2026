"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask
    uv run python -m app.ask "LoRAとは何ですか？"  # 最初の質問を指定して対話する

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def print_usage(response) -> None:
    usage = response.usage_metadata
    if usage is None:
        print("[usage] トークン数を取得できませんでした")
    else:
        print(f"[入力トークン数] {usage.prompt_token_count}")
        print(f"[出力トークン数] {usage.candidates_token_count}")


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)
    print_usage(response)
    return response.text


def chat(first_question: str | None = None) -> None:
    client = genai.Client(api_key=get_api_key())
    conversation = client.chats.create(model=MODEL)
    print("対話モードです。終了するには /exit と入力してください。")
    question_to_send = first_question

    while True:
        if question_to_send is None:
            try:
                question = input("あなた> ").strip()
            except EOFError:
                print()
                return
        else:
            question = question_to_send
            question_to_send = None
            print(f"あなた> {question}")

        if question == "/exit":
            return
        if not question:
            continue

        response = conversation.send_message(question)
        print(f"\n[回答]\n{response.text}\n")
        print_usage(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMと対話する")
    parser.add_argument("question", nargs="?")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        chat(args.question)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
