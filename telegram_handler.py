from __future__ import annotations
import os
from typing import Dict, Any, List
from telegram import Update
from telegram.ext import ContextTypes, Application
from notion_service import NotionService
from gcp_service import GcpService
from generate import answer_question, summarize_conversation, answer_with_grounding, summarize_segment

# 以簡單的 in-memory 狀態記錄對話（可換成 Redis/DB）
CONV: Dict[int, Dict[str, Any]] = {}

def inject_services(app: Application, notion: NotionService, gcp_service: GcpService):
    """把外部服務放到 application.bot_data"""
    app.bot_data["notion"] = notion
    app.bot_data["gcp"] = gcp_service

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "您好！這是一個整合 Notion 客戶資料與公司產品知識庫的助理 Bot。\n\n"
        "主要指令：\n"
        "1. `/ask [客戶名稱] [問題]`\n"
        "   - 載入客戶資料並針對您的問題進行回覆。\n\n"
        "2. `/ask [客戶名稱]`\n"
        "   - 僅載入並顯示客戶畫像資料，以開始團隊討論。\n\n"
        "在載入客戶資料後，您可以使用以下指令：\n"
        "- `/query [問題]`: 根據當前討論向 AI 提問。\n"
        "- `/products [問題]`: 根據當前討論搜尋產品資料庫。\n"
        "- `/end`: 結束目前對話，並將討論摘要寫入 Notion。\n\n"
        "您也可以隨時輸入 `/start` 來查看此說明。"
    )

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    args = context.args or []

    if not args:
        await update.message.reply_text("指令用法：\n- /ask [客戶名稱] [問題]\n- /ask [客戶名稱]")
        return

    # --- 新邏輯開始 ---
    # 不論是哪種用法，第一個參數都是客戶名稱
    customer_name = args[0]
    notion: NotionService = context.application.bot_data["notion"]
    
    try:
        hits = notion.find_customer_pages_by_title(customer_name)
    except Exception as e:
        await update.message.reply_text(f"從 Notion 查詢資料時發生錯誤：{e}")
        return

    if not hits:
        await update.message.reply_text(f"在 Notion 中找不到名為「{customer_name}」的客戶。 সন")
        return

    if len(hits) > 1:
        names = [h.get("_title") for h in hits]
        await update.message.reply_text("找到多筆符合的客戶，請更精確地指定名稱：\n- " + "\n- ".join(names))
        return

    page = hits[0]
    page_id = page["id"]
    title = page.get("_title") or "未命名"
    portrait = notion.get_page_portrait_section(page_id)

    # 為兩種情況都預先設定好對話狀態
    CONV[chat_id] = {
        "customer_title": title,
        "page_id": page_id,
        "portrait": portrait,
        "history": [],
    }

    # 根據參數數量決定執行問答還是僅載入資料
    if len(args) >= 2:
        # 用法一：/ask [客戶名稱] [問題]
        question = " ".join(args[1:])
        await update.message.reply_text("正在為您分析客戶畫像並生成回覆，請稍候...")
        
        answer = await answer_question(title, portrait, question)
        
        # 記錄問題和答案到歷史紀錄
        CONV[chat_id]["history"].append({"role": "user", "content": question})
        CONV[chat_id]["history"].append({"role": "assistant", "content": answer})
        
        await update.message.reply_text(
            f"【{title}】\nQ: {question}\n\n"
            f"A:\n{answer}\n\n"
            "（可成員討論，或使用 /query, /products 等指令詢問，或用 /end 結束）"
        )
    else:
        # 用法二：/ask [客戶名稱]
        question = ""
        await update.message.reply_text("正在為您分析客戶畫像並生成回覆，請稍候...")
        
        answer = await answer_question(title, portrait, question)
        
        # 記錄問題和答案到歷史紀錄
        CONV[chat_id]["history"].append({"role": "user", "content": question})
        CONV[chat_id]["history"].append({"role": "assistant", "content": answer})
        
        await update.message.reply_text(
            f"A:\n{answer}\n\n"
            "（可成員討論，或使用 /query, /products 等指令詢問，或用 /end 結束）"
            )

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
    """處理 /query 指令，結合討論內容進行提問，並在事後進行摘要"""
    chat_id = update.message.chat_id
    if chat_id not in CONV:
        await update.message.reply_text("沒有進行中的對話，請先用 /ask [客戶名稱] [問題] 開始。 সন")
        return
    args = context.args or []
    if not args:
        await update.message.reply_text("請輸入問題：/query [你的問題]")
        return
    
    new_question = " ".join(args)
    state = CONV[chat_id]
    history = state["history"]

    # 1. 找出最近的對話片段作為上下文
    last_assistant_index = -1
    for i in range(len(history) - 1, -1, -1):
        if history[i]["role"] == "assistant":
            last_assistant_index = i
            break
    
    recent_discussions = history[last_assistant_index:] if last_assistant_index != -1 else history
    discussion_texts = [msg["content"] for msg in recent_discussions if msg["role"] in ["discussion", "assistant"]]
    
    prompt_context = ""
    if discussion_texts:
        discussion_summary = "\n".join(discussion_texts)
        prompt_context = f"請參考以下團隊成員的討論：\n---\n{discussion_summary}\n---\n"
    
    full_question = f"{prompt_context}基於以上討論，請回答這個問題：{new_question}"
    
    # 2. 產生並回覆答案
    await update.message.reply_text("正在整合討論內容並為您查詢，請稍候...")
    answer = await answer_with_grounding(full_question)
    await update.message.reply_text(answer)

    # 3. 在背景進行滾動式摘要
    segment_to_summarize = (history[last_assistant_index:] if last_assistant_index != -1 else history).copy()
    segment_to_summarize.append({"role": "user", "content": new_question})
    segment_to_summarize.append({"role": "assistant", "content": answer})

    summary_text = await summarize_segment(segment_to_summarize)
    
    # 用摘要取代原始的對話片段
    history_before_segment = history[:last_assistant_index] if last_assistant_index != -1 else []
    CONV[chat_id]["history"] = history_before_segment + [
        {"role": "assistant", "content": summary_text}
    ]
    print(f"DEBUG: Chat {chat_id} history compressed. New length: {len(CONV[chat_id]['history'])}")

async def database_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """處理 /products 指令，參考討論內容搜尋，並在事後進行摘要"""
    chat_id = update.message.chat_id
    if chat_id not in CONV:
        await update.message.reply_text("沒有進行中的對話，請先用 /ask [客戶名稱] [問題] 開始。 সন")
        return
    args = context.args or []
    if not args:
        await update.message.reply_text("請輸入查詢產品資料庫的問題：/products [你的問題]")
        return
    user_query = " ".join(args)
    state = CONV[chat_id]
    history = state["history"]

    # 1. 找出最近的對話片段作為上下文
    last_assistant_index = -1
    for i in range(len(history) - 1, -1, -1):
        if history[i]["role"] == "assistant":
            last_assistant_index = i
            break
    
    recent_discussions = history[last_assistant_index:] if last_assistant_index != -1 else history
    discussion_texts = [msg["content"] for msg in recent_discussions if msg["role"] in ["discussion", "assistant"]]

    context_for_search = ""
    if discussion_texts:
        discussion_summary = "\n".join(discussion_texts)
        context_for_search = f"請參考以下團隊成員的討論摘要：\n---\n{discussion_summary}\n---\n"

    full_query = f"{context_for_search}根據上述討論，請到公司產品資料庫中搜尋並回答：{user_query}"

    # 2. 產生並回覆答案
    await update.message.reply_text("正在參考討論內容，到公司產品資料庫搜尋，請稍候...")
    gcp_service: GcpService = context.application.bot_data["gcp"]
    answer = gcp_service.query_knowledge_base(full_query)
    await update.message.reply_text(answer)

    # 3. 在背景進行滾動式摘要
    segment_to_summarize = (history[last_assistant_index:] if last_assistant_index != -1 else history).copy()
    segment_to_summarize.append({"role": "user", "content": f"/products {user_query}"})
    segment_to_summarize.append({"role": "assistant", "content": answer})
    
    summary_text = await summarize_segment(segment_to_summarize)
    
    history_before_segment = history[:last_assistant_index] if last_assistant_index != -1 else []
    CONV[chat_id]["history"] = history_before_segment + [
        {"role": "assistant", "content": summary_text}
    ]
    print(f"DEBUG: Chat {chat_id} history compressed. New length: {len(CONV[chat_id]['history'])}")

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

async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """處理機器人無法識別的指令"""
    await update.message.reply_text("抱歉，我無法識別這個指令。請確認您的輸入是否正確。 সন")
