import os
import logging
from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from telegram_handler import start, ask_command, end_command, cancel_command, query_command, database_command, handle_message, inject_services, unknown_command, direct_database_command
from notion_service import NotionService
from gcp_service import GcpService

# 啟用日誌
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.WARNING
)
logger = logging.getLogger(__name__)

def main():
    # 讀取 .env
    load_dotenv()
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    notion_key = os.getenv("NOTION_API_KEY")
    notion_db = os.getenv("NOTION_DATABASE_ID")

    gcp_project_id = os.getenv("GCP_PROJECT_ID")
    gcp_location = os.getenv("GCP_LOCATION") # In gcp_search.py, this is 'us'
    gcp_engine_id = os.getenv("GCP_ENGINE_ID")
    gcp_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not tg_token:
        raise RuntimeError("請在 .env 內設定 TELEGRAM_BOT_TOKEN")
    if not notion_key or not notion_db:
        raise RuntimeError("請在 .env 內設定 NOTION_API_KEY 與 NOTION_DATABASE_ID")
    if not gcp_project_id or not gcp_location or not gcp_engine_id or not gcp_credentials_path:
        raise RuntimeError("請在 .env 內設定 GCP_PROJECT_ID, GCP_LOCATION, GCP_ENGINE_ID, GCP_SERVICE_ACCOUNT_KEY_PATH")

    # 建立服務
    notion = NotionService(api_key=notion_key, database_id=notion_db)
    gcp_service = GcpService(
        project_id=gcp_project_id,
        location=gcp_location,
        engine_id=gcp_engine_id,
        credentials_path=gcp_credentials_path
    )

    # Telegram Application
    application = Application.builder().token(tg_token).build()

    # 把服務注入到 handler
    inject_services(application, notion, gcp_service)

    # 指令與訊息處理
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("end", end_command))
    application.add_handler(CommandHandler("cancel", cancel_command))
    application.add_handler(CommandHandler("query", query_command))
    application.add_handler(CommandHandler("products", database_command))
    application.add_handler(CommandHandler("search_db", direct_database_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # 未知指令處理，放在最後
    application.add_handler(MessageHandler(filters.COMMAND & ~filters.UpdateType.EDITED_MESSAGE, unknown_command))

    logger.info("Bot 已啟動，開始輪詢...")
    application.run_polling()

if __name__ == "__main__":
    main()