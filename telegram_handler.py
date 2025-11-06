from __future__ import annotations
import os
from typing import Dict, Any, List
from telegram import Update
from telegram.ext import ContextTypes, Application
from notion_service import NotionService
from generate import answer_question, summarize_conversation, answer_with_grounding

# 以簡單的 in-memory 狀態記錄對話（可換成 Redis/DB）
CONV: Dict[int, Dict[str, Any]] = {}

def inject_services(app: Application, notion: NotionService):
    """把外部服務放到 application.bot_data"""
    app.bot_data["notion"] = notion

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("您好！\n- /ask [客戶名稱] [問題]\n- /ask [一般問題]")

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    args = context.args or []

    if not args:
        await update.message.reply_text("指令用法：\n- /ask [客戶名稱] [問題]\n- /ask [一般問題]")
        return

    # 模式一：特定客戶問答 (參數 >= 2)
    if len(args) >= 2:
        customer_name = args[0]
        question = " ".join(args[1:])
        notion: NotionService = context.application.bot_data["notion"]
        hits = notion.find_customer_pages_by_title(customer_name)

        if hits:
            if len(hits) > 1:
                names = [h.get("_title") for h in hits]
                await update.message.reply_text("找到多筆，請更精確：\n- " + "\n- ".join(names))
                return

            page = hits[0]
            page_id = page["id"]
            title = page.get("_title") or "未命名"
            portrait = notion.get_page_portrait_section(page_id)

            # 設定對話狀態
            CONV[chat_id] = {
                "customer_title": title,
                "page_id": page_id,
                "portrait": portrait,
                "history": [],
            }

            # 呼叫 LLM 並記錄
            answer = await answer_question(title, portrait, question)
            CONV[chat_id]["history"].append({"role": "user", "content": question})
            CONV[chat_id]["history"].append({"role": "assistant", "content": answer})

            await update.message.reply_text(f"【{title}】\nQ: {question}\n\nA:\n{answer}\n\n（可輸入一般訊息繼續追問，或 /end 結束）")
            return

    # 模式二：通用問答 (參數 < 2 或找不到客戶)
    question = " ".join(args)
    await update.message.reply_text(f"正在為您查詢「{question}」，請稍候...")
    answer = await answer_with_grounding(question)
    await update.message.reply_text(answer)

async def end_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    if chat_id not in CONV:
        await update.message.reply_text("目前沒有進行中的會談。")
        return

    await update.message.reply_text("正在為您總結對話並寫入 Notion，請稍候...")

    state = CONV[chat_id]
    page_id = state["page_id"]
    title = state["customer_title"]
    history = state["history"]

    # 1. 產生摘要
    summary_text = await summarize_conversation(title, history)

    # 2. 準備寫入 Notion 的 blocks
    blocks_to_append = [
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": f"與 {update.message.from_user.full_name} 的討論摘要"}}]
            }
        },
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": summary_text}}]
            }
        },
        {
            "object": "block",
            "type": "divider",
            "divider": {}
        }
    ]

    # 3. 寫入 Notion
    try:
        notion: NotionService = context.application.bot_data["notion"]
        notion.append_blocks_to_page(page_id, blocks_to_append)
        await update.message.reply_text(f"已將本次討論摘要寫入 Notion 頁面：【{title}】")
    except Exception as e:
        await update.message.reply_text(f"寫入 Notion 時發生錯誤：{e}")
    finally:
        # 4. 結束會談
        del CONV[chat_id]

async def query_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """處理 /query 指令，結合討論內容進行提問"""
    chat_id = update.message.chat_id
    if chat_id not in CONV:
        await update.message.reply_text("沒有進行中的對話，請先用 /ask [客戶名稱] [問題] 開始。")
        return
    args = context.args or []
    if not args:
        await update.message.reply_text("請輸入問題：/query [你的問題]")
        return
    new_question = " ".join(args)
    state = CONV[chat_id]
    history = state["history"]
    # 找出最後一筆 assistant 的回覆，從那之後的都是新討論
    last_assistant_index = -1
    for i in range(len(history) - 1, -1, -1):
        if history[i]["role"] == "assistant":
            last_assistant_index = i
            break
    recent_discussions = history[last_assistant_index + 1:]
    discussion_texts = [msg["content"] for msg in recent_discussions if msg["role"] == "discussion"]
    # 組合 prompt
    prompt_context = ""
    if discussion_texts:
        discussion_summary = "\n".join(discussion_texts)
        prompt_context = f"請參考以下團隊成員的討論：\n---\n{discussion_summary}\n---\n"
    full_question = f"{prompt_context}基於以上討論，請回答這個問題：{new_question}"
    await update.message.reply_text("正在整合討論內容並為您查詢，請稍候...")
    answer = await answer_with_grounding(full_question)
    state["history"].append({"role": "user", "content": new_question})
    state["history"].append({"role": "assistant", "content": answer})
    await update.message.reply_text(answer)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """處理一般訊息，當作討論內容記錄下來"""
    chat_id = update.message.chat_id
    if chat_id not in CONV:
        return  # 非對話模式下，忽略一般訊息

    # 將訊息存入歷史紀錄，標記為討論
    state = CONV[chat_id]
    user = update.message.from_user
    message_text = update.message.text
    
    state["history"].append({
        "role": "discussion",
        "content": f"{user.full_name}: {message_text}"
    })
    # 不對討論內容做任何回覆