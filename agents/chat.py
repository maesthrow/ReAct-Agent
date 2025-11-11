import os
from datetime import date
from dotenv import load_dotenv
from langchain_gigachat.chat_models import GigaChat

load_dotenv()
GIGA_AUTH_DATA = os.getenv("GIGA_AUTH_DATA")

llm = GigaChat(credentials=GIGA_AUTH_DATA, verify_ssl_certs=False)

today = date.today().strftime("%d.%m.%Y")  # DD.MM.YYYY
system_prompt = f"Сегодня {today}. Ты полезный ассистент. Вежливо, кратко и по делу отвечай на вопросы."


def run(query: str):
    # просто передаём список сообщений
    resp = llm.invoke([("system", system_prompt), ("user", query)])
    print(resp.content)


if __name__ == "__main__":
    print("⚡ Простой LLM-чат готов. Пустая строка — выход.")
    try:
        while True:
            q = input("> ").strip()
            if not q:
                break
            run(q)
    except (KeyboardInterrupt, EOFError):
        pass
