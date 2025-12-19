# Cooperation Project - 智能業務助理 Bot

這是一個結合 **Notion 客戶資料庫** 與 **Google Cloud Vertex AI (RAG)** 的 Telegram 機器人。旨在協助團隊成員快速獲取客戶背景資訊（客戶畫像）、查詢公司產品知識庫，並自動將討論摘要回寫至 Notion。

## 🚀 功能特色

* **Notion 整合**：
    * 透過 `/ask` 指令快速讀取 Notion 資料庫中的客戶畫像。
    * 支援模糊搜尋客戶名稱。
    * 自動將對話摘要寫入 Notion 客戶頁面。
* **企業級知識庫搜尋 (RAG)**：
    * 整合 Google Cloud Vertex AI Search (Discovery Engine)。
    * 支援針對特定產品文件的語意搜尋與問答。
    * `/products` 指令可根據對話上下文搜尋內部產品資料。
* **AI 智能對話**：
    * 基於 Gemini 模型，具備上下文理解能力。
    * 提供 `/query` 指令針對目前討論內容進行 AI 提問。
* **自動化摘要與紀錄**：
    * 對話結束時自動生成結構化摘要。
    * 支援使用者透過自然語言修訂摘要內容，確認後再存檔。
* **Redis 狀態管理**：
    * 使用 Redis 暫存對話狀態，支援多人同時使用不衝突。

## 🛠️ 技術棧

* **語言**: Python 3.11+
* **框架**: `python-telegram-bot` (非同步)
* **AI & 搜尋**: Google Vertex AI (Gemini), Vertex AI Search
* **資料庫/筆記**: Notion API
* **快取**: Redis
* **套件管理**: `uv` (亦支援 `pip`)

## ⚙️ 安裝與設定

### 1. 環境準備

請確保您的系統已安裝：
* Python 3.11 或以上
* Redis Server (本機或雲端)
* [uv](https://github.com/astral-sh/uv) (建議使用，或使用 pip)

### 2. 下載專案

```bash
git clone <repository-url>
cd coopeartion_project
```

### 3. 安裝相依套件
使用 uv 安裝：
```bash
uv sync
```
或使用 pip 安裝：
```bash
pip install -r requirements.txt
```

### 4. 設定環境變數
請建立一個 `.env` 檔案，並填入以下必要的環境變數：
```env
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# Notion API
NOTION_API_KEY=your_notion_integration_token
NOTION_DATABASE_ID=your_database_id
NOTION_API_VERSION=2025-09-03  # 選填，預設為 2025-09-03

# Google Cloud Platform (Vertex AI & Storage)
GCP_PROJECT_ID=your_gcp_project_id
GCP_LOCATION=us                # 例如: us (Vertex AI Search location)
GCP_ENGINE_ID=your_engine_id   # Vertex AI Search App ID
GCP_BUCKET_NAME=your_bucket_name
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json

# Redis
REDIS_URL=redis://localhost:6379
```

### 5. 啟動機器人
使用 uv 啟動：
```bash
uv run bot.py
```
或使用 python 啟動：
```bash
python bot.py
```

## 📜 使用說明
啟動機器人後，您可以在 Telegram 中使用以下指令：
### 會談管理
* `/start`: 顯示歡迎訊息與指令說明。

* `/ask [客戶名稱] [問題]`: 載入客戶資料並開始會談。若帶上問題，AI 會結合客戶畫像直接回答。
範例：/ask 台積電 最近有什麼新的需求？

* `/ask [客戶名稱]`: 僅載入客戶資料，進入會談模式。
* `/end`: 結束目前會談，AI 會自動生成討論摘要供您確認。
* `/save`: 確認摘要無誤後，將其寫入 Notion 並清除會談狀態。
* `/cancel`: 強制中斷會談，不儲存任何內容。

### 查詢與輔助
* `/query [問題]`: 根據「目前的討論內容」與「客戶資料」向 AI 提問。
* `/products [問題]`: 根據「目前的討論內容」去搜尋「公司產品資料庫」。
* `/search_db [問題]`: 獨立指令，不需開啟會談即可直接搜尋公司產品資料庫。

## 📂 專案結構
``` Plaintext
.
├── main.py              # 程式進入點，初始化各個服務與 Bot
├── telegram_handler.py  # 處理 Telegram 指令與訊息邏輯
├── notion_service.py    # 封裝 Notion API (查詢客戶、寫入摘要)
├── gcp_service.py       # 封裝 GCP Vertex AI (Search & Gemini)
├── generate.py          # AI 生成邏輯 (摘要、回答問題)
├── redis_client.py      # Redis 連線與狀態管理
├── check_auth.py        # 權限檢查工具
├── pyproject.toml       # 專案依賴設定
├── uv.lock              # 依賴版本鎖定檔
└── .env                 # 環境變數設定 (需自行建立)
```