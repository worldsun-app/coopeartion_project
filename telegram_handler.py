from __future__ import annotations
import os
import asyncio
from typing import Dict, Any, List
from telegram import Update
from telegram.ext import ContextTypes, Application
from notion_service import NotionService
from gcp_service import GcpService
from generate import answer_question, summarize_conversation, answer_with_grounding, summarize_segment, extract_product_from_query, extract_keywords_from_query, refine_summary

# 串接 Redis 客戶端
from redis_client import get_conv_state, set_conv_state, delete_conv_state

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
        "- `/end`: 結束目前對話，並生成摘要準備寫入Notion。\n"
        "--- `/save`: 儲存目前的摘要到 Notion。\n"
        "- `/cancel`: 中斷目前對話，不儲存任何內容。\n\n"
        "獨立指令：\n"
        "- `/search_db [你的問題]`: 直接查詢公司產品資料庫，可包含產品名稱以聚焦搜尋。\n"
        "您也可以隨時輸入 `/start` 來查看此說明。"
    )

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    args = context.args or []

    # --- 修改：檢查 Redis 中是否已有會談 ---
    if get_conv_state(chat_id):
        state = get_conv_state(chat_id)
        current_customer = state.get("customer_title", "目前的")
        await update.message.reply_text(
            f"您目前正在與「{current_customer}」的會談中。\n"
            "請先使用 /end 或 /cancel 指令結束目前的會談，才能開啟新的客戶資料。"
        )
        return

    if not args:
        await update.message.reply_text("指令用法：\n- /ask [客戶名稱] [問題]\n- /ask [客戶名稱]")
        return

    customer_name = args[0]
    notion: NotionService = context.application.bot_data["notion"]
    gcp: GcpService = context.application.bot_data["gcp"]
    client = gcp.genai_client
    
    try:
        hits = await asyncio.to_thread(notion.find_customer_pages_by_title, customer_name)
    except Exception as e:
        await update.message.reply_text(f"從 Notion 查詢資料時發生錯誤：{e}")
        return

    if not hits:
        await update.message.reply_text(f"在 Notion 中找不到名為「{customer_name}」的客戶。")
        return

    if len(hits) > 1:
        names = [h.get("_title") for h in hits]
        await update.message.reply_text("找到多筆符合的客戶，請更精確地指定名稱：\n- " + "\n- ".join(names))
        return

    page = hits[0]
    page_id = page["id"]
    title = page.get("_title") or "未命名"
    portrait = await asyncio.to_thread(notion.get_page_portrait_section, page_id)

    # --- 修改：建立新的會談狀態並存入 Redis ---
    new_state = {
        "customer_title": title,
        "page_id": page_id,
        "portrait": portrait,
        "history": [],
    }
    set_conv_state(chat_id, new_state)

    if len(args) >= 2:
        question = " ".join(args[1:])
        await update.message.reply_text("正在為您分析客戶畫像並生成回覆，請稍候...")
        
        # --- 修改：呼叫 answer_question 時傳入 history ---
        answer = await answer_question(client, title, portrait, question, history=new_state["history"])
        
        # --- 修改：更新 Redis 中的狀態 ---
        current_state = get_conv_state(chat_id)
        current_state["history"].append({"role": "user", "content": question})
        current_state["history"].append({"role": "assistant", "content": answer})
        set_conv_state(chat_id, current_state)
        
        await update.message.reply_text(
            f"【{title}】\nQ: {question}\n\n"
            f"A:\n{answer}\n\n"
            "（可成員討論，或使用 /query, /products, /end, /cancel 等指令詢問）"
        )
    else:
        question = ""
        await update.message.reply_text("正在為您分析客戶畫像並生成回覆，請稍候...")
        
        # --- 修改：呼叫 answer_question 時傳入 history ---
        answer = await answer_question(title, portrait, question, history=new_state["history"])

        # --- 修改：更新 Redis 中的狀態 ---
        current_state = get_conv_state(chat_id)
        current_state["history"].append({"role": "user", "content": question})
        current_state["history"].append({"role": "assistant", "content": answer})
        set_conv_state(chat_id, current_state)
        
        await update.message.reply_text(
            f"A:\n{answer}\n\n"
            "（可成員討論，或使用 /query, /products, /end, /cancel 等指令詢問）"
        )

async def _summarize_task(
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    history: List[Dict[str, str]],
    title: str
):
    """在背景執行摘要生成的非同步任務"""
    try:
        gcp: GcpService = context.application.bot_data["gcp"]
        client = gcp.genai_client
        
        summary_text = await summarize_conversation(client, title, history)
        state = get_conv_state(chat_id)
        if not state:
            # 如果在此期間使用者取消了會談，state 可能會是 None
            print(f"DEBUG: Chat {chat_id} state was deleted before summary task could complete.")
            return
            
        state["pending_summary"] = summary_text
        state["awaiting_save"] = True
        set_conv_state(chat_id, state)

        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"【討論摘要初稿】\n\n{summary_text}\n\n"
                "-----------------------------\n"
                "請確認以上內容：\n"
                "若需 **修改**，請直接發送修改需求。\n"
                "若 **確認無誤**，請輸入 `/save` 將其寫入 Notion 並結束會談。"
            )
        )
    except Exception as e:
        # 4. 錯誤處理：通知使用者任務失敗
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ 生成摘要時發生錯誤：{e}\n請稍後再試一次 `/end`。"
        )

async def end_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    結束指令第一階段：啟動背景任務生成摘要
    """
    chat_id = update.message.chat_id
    state = get_conv_state(chat_id)

    if not state:
        await update.message.reply_text("目前沒有進行中的會談。")
        return

    await update.message.reply_text("好的，正在為您總結對話。內容生成需要一些時間，完成後會發送給您。")

    history = state["history"]
    title = state["customer_title"]
    asyncio.create_task(
        _summarize_task(chat_id=chat_id, context=context, history=history, title=title)
    )

async def save_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    結束指令第二階段：將確認後的摘要寫入 Notion
    """
    chat_id = update.message.chat_id
    state = get_conv_state(chat_id)

    if not state or not state.get("awaiting_save"):
        await update.message.reply_text("目前沒有等待儲存的摘要。請先使用 /end 生成摘要。")
        return

    summary_text = state.get("pending_summary", "")
    page_id = state["page_id"]
    title = state["customer_title"]

    await update.message.reply_text("正在寫入 Notion...")

    # 根據聊天室類型決定摘要標題的來源名稱
    chat = update.message.chat
    if chat.type == 'private':
        source_name = chat.full_name
    else:  # 'group' or 'supergroup'
        source_name = chat.title
    
    summary_title = f"與 {source_name} 的討論摘要"

    char_limit = 2000
    summary_chunks = [summary_text[i:i+char_limit] for i in range(0, len(summary_text), char_limit)]

    blocks_to_append = [
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": summary_title}}]
            }
        }
    ]
    
    for chunk in summary_chunks:
        blocks_to_append.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": chunk}}]
            }
        })

    blocks_to_append.append({
        "object": "block",
        "type": "divider",
        "divider": {}
    })

    try:
        notion: NotionService = context.application.bot_data["notion"]
        await asyncio.to_thread(notion.append_blocks_to_page, page_id, blocks_to_append)
        await update.message.reply_text(f"✅ 已將本次討論摘要寫入 Notion 頁面：【{title}】\n會談已正式結束。")

        delete_conv_state(chat_id)
        
    except Exception as e:
        await update.message.reply_text(f"❌ 寫入 Notion 時發生錯誤：{e}\n請稍後重新嘗試輸入 `/save`。")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """處理一般訊息：可能是討論內容，也可能是修改摘要"""
    chat_id = update.message.chat_id
    state = get_conv_state(chat_id)

    if not state:
        return

    message_text = update.message.text

    if state.get("awaiting_save"):
        await update.message.reply_text("正在根據您的指令修訂摘要，請稍候...")
        gcp: GcpService = context.application.bot_data["gcp"]
        client = gcp.genai_client
        new_summary = await refine_summary(client, state["pending_summary"], message_text)
        
        state["pending_summary"] = new_summary
        set_conv_state(chat_id, state)
        
        await update.message.reply_text(
            f"【修訂後的摘要】\n\n{new_summary}\n\n"
            "-----------------------------\n"
            "請確認以上內容：\n"
            "若需 **繼續修改**，請直接發送新的指令。\n"
            "若 **確認無誤**，請輸入 `/save` 確認儲存。"
        )
        return

    user = update.message.from_user
    state["history"].append({
        "role": "discussion",
        "content": f"{user.full_name}: {message_text}"
    })
    set_conv_state(chat_id, state)

async def query_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    state = get_conv_state(chat_id)

    if not state:
        await update.message.reply_text("沒有進行中的對話，請先用 /ask [客戶名稱] [問題] 開始。" )
        return
        
    args = context.args or []
    if not args:
        await update.message.reply_text("請輸入問題：/query [你的問題]")
        return
    
    new_question = " ".join(args)
    history = state["history"]

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

    await update.message.reply_text("正在整合討論內容並為您查詢，請稍候...")
    gcp: GcpService = context.application.bot_data["gcp"]
    client = gcp.genai_client
    answer = await answer_with_grounding(client, full_question)
    await update.message.reply_text(answer)

    segment_to_summarize = (history[last_assistant_index:] if last_assistant_index != -1 else history).copy()
    segment_to_summarize.append({"role": "user", "content": new_question})
    segment_to_summarize.append({"role": "assistant", "content": answer})

    summary_text = await summarize_segment(client, segment_to_summarize)

    history_before_segment = history[:last_assistant_index] if last_assistant_index != -1 else []
    
    state["history"] = history_before_segment + [
        {"role": "assistant", "content": summary_text}
    ]
    set_conv_state(chat_id, state)
    print(f"DEBUG: Chat {chat_id} history compressed. New length: {len(state['history'])}")

async def database_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    state = get_conv_state(chat_id)

    if not state:
        await update.message.reply_text("沒有進行中的對話，請先用 /ask [客戶名稱] [問題] 開始。")
        return

    args = context.args or []
    if not args:
        await update.message.reply_text("請輸入查詢產品資料庫的問題：/products [你的問題]")
        return
    user_query = " ".join(args)
    history = state["history"]
    gcp: GcpService = context.application.bot_data["gcp"]
    client = gcp.genai_client

    product_filter = await extract_product_from_query(client, user_query)
    keywords = await extract_keywords_from_query(client, user_query)

    matching_term = product_filter
    if not matching_term and keywords:
        matching_term = ",".join(keywords)
        print(f"DEBUG: No product name found, using keywords as filter: {matching_term}")

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

    base_query = f"{context_for_search}根據上述討論，請到公司產品資料庫中搜尋並回答：{user_query}"
    
    final_query = base_query
    query_parts = []
    if keywords:
        keyword_query_part = " OR ".join([f'"{k}"' for k in keywords])
        query_parts.append(f"({keyword_query_part})")
    if base_query:
        query_parts.append(f"({base_query})")
    if query_parts:
        final_query = " AND ".join(query_parts)
    else:
        final_query = user_query 

    if matching_term: # 改用 matching_term 判斷
        final_query = f'"{matching_term}" AND {final_query}' # 這裡其實只是增加權重，核心還是靠 gcp_service 的注入
        await update.message.reply_text(f"好的，正在為您搜尋關於「{matching_term}」的相關資料...")
    else:
        await update.message.reply_text("正在參考討論內容，到公司產品資料庫搜尋，請稍候...")
    
    gcp_service: GcpService = context.application.bot_data["gcp"]
    answer = await asyncio.to_thread(
        gcp_service.query_knowledge_base,
        user_question=user_query,
        product_filter=matching_term,
        search_query=final_query, 
        conversation_history=context_for_search
    )
    await update.message.reply_text(answer)

    segment_to_summarize = (history[last_assistant_index:] if last_assistant_index != -1 else history).copy()
    segment_to_summarize.append({"role": "user", "content": f"/products {user_query}"})
    segment_to_summarize.append({"role": "assistant", "content": answer})
    
    summary_text = await summarize_segment(client, segment_to_summarize)
    
    history_before_segment = history[:last_assistant_index] if last_assistant_index != -1 else []

    state["history"] = history_before_segment + [
        {"role": "assistant", "content": summary_text}
    ]
    set_conv_state(chat_id, state)
    print(f"DEBUG: Chat {chat_id} history compressed. New length: {len(state['history'])}")

async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """處理機器人無法識別的指令"""
    await update.message.reply_text("抱歉，我無法識別這個指令。請確認您的輸入是否正確。")

async def direct_database_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """獨立查詢公司產品資料庫，不影響對話歷史"""
    args = context.args or []
    if not args:
        await update.message.reply_text("請輸入您要查詢產品資料庫的問題：/search_db [你的問題]")
        return
    user_query = " ".join(args)
    gcp: GcpService = context.application.bot_data["gcp"]
    client = gcp.genai_client
    product_filter = await extract_product_from_query(client, user_query)

    keywords = await extract_keywords_from_query(client, user_query)
    final_query = ""

    if keywords:
        keyword_query_part = " OR ".join([f'"{k}"' for k in keywords])
        keyword_query = f"({keyword_query_part})"
        
        if product_filter:
            final_query = f'("{product_filter}") AND {keyword_query}'
            await update.message.reply_text(f"好的，正在為您在「{product_filter}」中搜尋：{user_query}...")
        else:
            final_query = keyword_query
            await update.message.reply_text(f"好的，正在為您搜尋：{user_query}...")
    else:
        if product_filter:
            final_query = f'("{product_filter}") AND ("{user_query}")'
            await update.message.reply_text(f"正在嘗試在「{product_filter}」中搜尋您的原文...")
        else:
            final_query = user_query
            await update.message.reply_text("正在嘗試直接搜尋您的原文...")

    gcp_service: GcpService = context.application.bot_data["gcp"]
    answer = await asyncio.to_thread(
        gcp_service.query_knowledge_base,
        user_question=user_query,
        product_filter=product_filter,
        search_query=final_query,
    )
    await update.message.reply_text(answer)

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """停止當前會談，不將任何內容儲存到 Notion。"""
    chat_id = update.message.chat_id
    if get_conv_state(chat_id):
        delete_conv_state(chat_id)
        await update.message.reply_text("會談已停止，未儲存任何內容到 Notion。")
    else:
        await update.message.reply_text("目前沒有進行中的會談可供停止。" )