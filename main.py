import os
import logging
from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram_handler import start, ask_command, end_command, query_command, handle_message, inject_services
from notion_service import NotionService

"""
主程式（Polling 版本）
- 從 .env 載入 TELEGRAM_BOT_TOKEN、NOTION_API_KEY、NOTION_DATABASE_ID、LLM_PROVIDER、OPENAI_API_KEY、GEMINI_API_KEY
- 建立 NotionService，注入到 telegram handler（放在 application.bot_data）
- 註冊 /start、/ask、/end、/query 與一般訊息處理
"""

# 啟用日誌
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    # 讀取 .env
    load_dotenv()
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    notion_key = os.getenv("NOTION_API_KEY")
    notion_db = os.getenv("NOTION_DATABASE_ID")

    if not tg_token:
        raise RuntimeError("請在 .env 內設定 TELEGRAM_BOT_TOKEN")
    if not notion_key or not notion_db:
        raise RuntimeError("請在 .env 內設定 NOTION_API_KEY 與 NOTION_DATABASE_ID")

    # 建立 Notion 服務
    notion = NotionService(api_key=notion_key, database_id=notion_db)

    # Telegram Application
    application = Application.builder().token(tg_token).build()

    # 把服務注入到 handler（透過 application.bot_data 共用）
    inject_services(application, notion)

    # 指令與訊息處理
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("end", end_command))
    application.add_handler(CommandHandler("query", query_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot 已啟動，開始輪詢...")
    application.run_polling()

if __name__ == "__main__":
    main()