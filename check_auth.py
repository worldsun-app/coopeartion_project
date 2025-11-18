
import os
from dotenv import load_dotenv
import google.auth

# 載入 .env 檔案中的環境變數
load_dotenv()

print("--- 開始檢查 Google Cloud 驗證身份 ---")

# 檢查 GOOGLE_APPLICATION_CREDENTIALS 環境變數是否已設定
cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if cred_path:
    print(f"環境變數 GOOGLE_APPLICATION_CREDENTIALS 已設定。")
    print(f"路徑: {cred_path}")
    if not os.path.exists(cred_path):
        print("警告: 這個路徑的檔案不存在！請檢查路徑是否正確。")
else:
    print("警告: 環境變數 GOOGLE_APPLICATION_CREDENTIALS 未設定。")

try:
    # 使用 google.auth.default() 來獲取預設憑證和專案 ID
    # 這會模擬 Google 客戶端函式庫的行為
    credentials, project_id = google.auth.default()
    print(credentials)
    print(project_id)

    print("\n驗證成功！")
    
    if hasattr(credentials, 'service_account_email'):
        print(f"腳本目前使用的身份是 (服務帳號): {credentials.service_account_email}")
    else:
        # 如果是使用者帳號 (透過 gcloud auth application-default login 設定)
        # credentials 物件可能沒有 service_account_email 屬性
        # 但我們可以嘗試從其他地方獲取資訊，或者告知使用者這是一個使用者帳號
        print(f"腳本目前使用的身份是: 使用者帳號 (可能來自 gcloud 或其他設定)")
        # credentials.token 屬性通常存在，但我們不印出 token
        print("這不是您預期的服務帳號，請檢查您的環境設定。")

    print(f"偵測到的專案 ID: {project_id}")

except Exception as e:
    print(f"\n驗證失敗: {e}")
    print("請檢查您的 GOOGLE_APPLICATION_CREDENTIALS 環境變數設定是否正確，以及 JSON 檔案是否有效。")

print("\n--- 檢查完畢 ---")
