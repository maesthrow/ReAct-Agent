import os
from pathlib import Path
from datetime import datetime, date

from dotenv import load_dotenv
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain.tools import tool

from langchain_gigachat.chat_models import GigaChat
from ddgs import DDGS

# ──────────────────────────────────────────────────────────────
DATA_DIR = Path("agent_data")
DATA_DIR.mkdir(exist_ok=True)

load_dotenv()
GIGA_AUTH_DATA = os.getenv("GIGA_AUTH_DATA")

llm = GigaChat(credentials=GIGA_AUTH_DATA, verify_ssl_certs=False)


# ── search_web ──────────────────────────────────────────────

@tool("search_web", description="Ищет в DuckDuckGo (RU, неделя, 5 ссылок)")
def search_web(query: str, max_results: int = 5) -> str:
    with DDGS() as ddgs:
        hits = ddgs.text(query, region="ru-ru", time="w", max_results=max_results)
        return "\n".join(f"{hit['title']}: {hit['body']} "
                         f"-- {hit['href']}" for hit in hits[:max_results])


# ── append_to_file ──────────────────────────────────────────

class AppendArgs(BaseModel):
    query: str = Field(..., description="Текст поискового запроса")
    content: str = Field(..., description="Найденный контент для записи в файл")


@tool(
    args_schema=AppendArgs,
    description="Добавить строку в локальный текстовый файл"
)
def append_to_file(query: str, content: str) -> str:
    path = DATA_DIR / Path('agent.log')
    stamp = datetime.now().strftime("[%d.%m.%Y %H:%M] ")  # DD.MM.YYYY
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{stamp} {query}: {content.rstrip()}\n")
    return f"✅ Записано в {path.name}"


# ── агент ───────────────────────────────────────────────────────

today = date.today().strftime("%d.%m.%Y")  # DD.MM.YYYY
system_prompt = (
    f"Сегодня {today}. Ты полезный ассистент.\n"
    "ВСЕГДА действуй по шагам:\n"
    "1) используй инструмент search_web, чтобы найти актуальные сведения;\n"
    "2) составь краткий конспект (1–4 предложения) + 3–5 лучших ссылок;\n"
    "3) вызови append_to_file(query='<текст поискового запроса>', content='<конспект и ссылки>');\n"
    "4) затем выдай пользователю четкий финальный ответ + найденные ссылки.\n"
    "Если поиск ничего не дал — так и скажи в своем финальном ответе."
)

agent = create_agent(
    model=llm,
    tools=[search_web, append_to_file],
    system_prompt=system_prompt,
)


# ── REPL ───────────────────────────────────────────────────

def run(query: str):
    resp = agent.invoke({"messages": [("user", query)]})
    # берем последний AI-ответ
    messages = resp["messages"]
    final = next((m for m in reversed(messages) if m.type == "ai"), messages[-1])
    print(final.content)


if __name__ == "__main__":
    print("⚡ Мини-агент готов (поиск + запись). Пустая строка -- выход.")
    try:
        while True:
            q = input("> ").strip()
            if not q:
                break
            run(q)
    except (KeyboardInterrupt, EOFError):
        pass
