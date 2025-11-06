from __future__ import annotations
import os
from typing import Dict, Any, List
import aiohttp
from telegram import Update
from telegram.ext import ContextTypes, Application
from notion_service import NotionService
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

# 以簡單的 in-memory 狀態記錄對話（可換成 Redis/DB）
CONV: Dict[int, Dict[str, Any]] = {}

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def inject_services(app: Application, notion: NotionService):
    """把外部服務放到 application.bot_data"""
    app.bot_data["notion"] = notion

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("您好！輸入 /ask 客戶名 問題")

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    args = context.args or []
    if len(args) < 2:
        await update.message.reply_text("指令用法：/ask [客戶名稱] [問題]")
        return

    customer_name = args[0]
    user_question = " ".join(args[1:])

    notion: NotionService = context.application.bot_data["notion"]
    hits = notion.find_customer_pages_by_title(customer_name)

    if not hits:
        await update.message.reply_text(f"找不到名稱含「{customer_name}」的客戶頁面。")
        return
    if len(hits) > 1:
        # 列出候選
        names = [h.get("_title") for h in hits]
        await update.message.reply_text("找到多筆，請更精確：\n- " + "\n- ".join(names))
        return

    page = hits[0]
    page_id = page["id"]
    title = page.get("_title") or "未命名"
    portrait = notion.get_page_portrait_section(page_id)
    print(portrait)

    # 設定對話狀態
    CONV[chat_id] = {
        "customer_title": title,
        "page_id": page_id,
        "portrait": portrait,
        "question": user_question,
    }
    # 呼叫 LLM
    answer = await _answer_with_llm(context, title, portrait, user_question)
    print(answer)

    await update.message.reply_text(f"【{title}】\nQ: {user_question}\n\nA:\n{answer}\n\n（可輸入一般訊息繼續追問，或 /end 結束）")

async def end_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    if chat_id in CONV:
        del CONV[chat_id]
    await update.message.reply_text("會談已結束。")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    if chat_id not in CONV:
        return  # 非會談模式下忽略
    follow_up = update.message.text.strip()
    state = CONV[chat_id]
    title = state["customer_title"]
    portrait = state["portrait"]
    # 將追問附加在原問題後
    prompt = f"{state['question']}\n（追問）{follow_up}"
    answer = await _answer_with_llm(context, title, portrait, prompt)
    await update.message.reply_text(answer)

# ----------------- LLM 協作 -----------------
async def _answer_with_llm(context, title: str, portrait: str, question: str) -> str:
    """
    單一入口：根據 .env 設定選擇 OpenAI 或 Gemini。
    若沒有金鑰，回傳安全退場的規則化答案（不阻斷流程）。
    """
    try:
        return await _call_gemini(title, portrait, question)
    except Exception as e:
        return f"呼叫 Gemini 發生錯誤：{e}"

def _build_user_prompt(title: str, portrait: str, question: str) -> str:
    return (
        f"客戶名稱：{title}\n"
        f"客戶畫像（節錄）：\n{portrait}\n\n"
        f"任務：請根據以上畫像與問題，產出回答。\n"
        f"問題：{question}\n"
        f"輸出格式：\n"
        f"1) 重點摘要（3-6 點）\n"
        f"2) 具體建議（可操作）\n"
    )

async def _call_gemini(title: str, portrait: str, question: str) -> str:
    prompt = _build_user_prompt(title, portrait, question)
    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = prompt,
    )
    return response.text