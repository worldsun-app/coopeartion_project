import os
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from google.oauth2 import service_account

from google import genai
from google.genai import types as genai_types

PROJECT_ID = "cooperationai-477907"
LOCATION = "us"
ENGINE_ID = "tg-to-datastroe_1763004052252"

# 你的 service account 金鑰路徑
CREDENTIALS_PATH = r"C:\Users\wsjoh\safe_credentials\cooperationai-key.json"


def _init_discovery_client(credentials):
    """初始化 Vertex AI Search 的 client。"""
    if LOCATION == "global":
        api_endpoint = "discoveryengine.googleapis.com"
    else:
        api_endpoint = f"{LOCATION}-discoveryengine.googleapis.com"

    client_options = ClientOptions(api_endpoint=api_endpoint)

    client = discoveryengine.SearchServiceClient(
        credentials=credentials,
        client_options=client_options,
    )

    serving_config = (
        f"projects/{PROJECT_ID}"
        f"/locations/{LOCATION}"
        f"/collections/default_collection"
        f"/engines/{ENGINE_ID}"
        f"/servingConfigs/default_search"
    )

    return client, serving_config


def _search_segments(client, serving_config, user_question, max_results=5, top_segments=12):
    """
    用 Vertex AI Search 取得與問題最相關的段落（extractive segments），
    回傳一個 list，每個元素包含 text / source_title / page。
    """
    preamble = (
        "你是一位專業、資深的企業知識庫問答助理。"
        "請嚴格根據以下提供的「資料來源」，僅使用「資料來源」中的資訊來詳細、有條理地並且用繁體中文進行回答「使用者的問題」。"
        "禁止參考任何「資料來源」以外的資訊或發揮創意。"
        "當問題是在詢問『有哪些保險、有哪些產品、目前有售哪些項目』時，"
        "你必須完整列出資料中所有相關產品名稱，不可以省略任何一項，也不可以自行合併。"
        "如果「資料來源」中沒有答案，請直接回答「根據現有資料，無法回答此問題」。"
    )

    summary_spec = discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
        summary_result_count=1,
        include_citations=False,
        model_prompt_spec=(
            discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelPromptSpec(
                preamble=preamble
            )
        ),
    )

    content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
        summary_spec=summary_spec,
        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
            max_snippet_count=1
        ),
        extractive_content_spec=(
            discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                max_extractive_answer_count=1,
                max_extractive_segment_count=10,
                return_extractive_segment_score=True,
            )
        ),
    )

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=user_question,
        page_size=max_results,
        content_search_spec=content_search_spec,
    )

    search_pager = client.search(request)
    response = next(search_pager.pages)
    # print("DEBUG: total results =", len(response.results))

    segments_info = []

    for i, result in enumerate(response.results, start=1):
        doc = result.document

        # 有些欄位會出現在 struct_data，有些在 derived_struct_data
        derived_data = dict(doc.derived_struct_data) if doc.derived_struct_data else {}
        # print(f"=== Result {i} derived_struct_data ===")
        # print(derived_data)

        # 1) 優先用 derived_struct_data 裡的 title
        raw_title = (
            derived_data.get("title")
            or derived_data.get("document_title")
        )

        # 2) 再從 link 解析出檔名（例如 忠意啟航創富 產品手冊）
        link = derived_data.get("link")
        filename_title = None
        if link:
            # link 可能是 "gs://bucket/忠意啟航創富 產品手冊.pdf"
            # 先拿最後一段，再去掉副檔名
            path_part = link
            if link.startswith("gs://"):
                # gs://bucket_name/path/to/file.pdf → 取 path/to/file.pdf
                path_part = link.split("/", 1)[-1]
            filename = os.path.basename(path_part)
            filename_title = os.path.splitext(filename)[0]  # 去掉 .pdf

        # 3) 決定最後要用哪一個顯示名稱
        if raw_title and filename_title:
            title = f"{filename_title}（{raw_title}）"
        else:
            title = raw_title or filename_title or "未命名文件"

        # 4) 取 extractive segments（注意 derived_struct_data 也是一個 Struct，要用 dict 後再拿）
        segments = derived_data.get("extractive_segments", [])
        for seg in segments:
            text = seg.get("content", "")
            if not text:
                continue
            page = seg.get("pageNumber")
            score = seg.get("score") or seg.get("confidenceScore") or 1.0

            segments_info.append(
                {
                    "text": text,
                    "source_title": title,
                    "page": page,
                    "score": float(score),
                }
            )

    segments_info.sort(key=lambda x: x["score"], reverse=True)
    selected_segments = segments_info[:top_segments]
    print(f"\nDEBUG: collected segments = {len(segments_info)}, selected = {len(selected_segments)}")
    return selected_segments, getattr(response, "summary", None)


def _generate_with_gemini(credentials, user_question, segments, summary_obj=None):
    """
    使用『Google Gen AI SDK』呼叫 Vertex 上的 Gemini，
    取代舊的 vertexai.generative_models 寫法。
    """
    # 1) 加上 cloud-platform scope，這是官方建議的做法
    scoped_credentials = credentials.with_scopes(
        ["https://www.googleapis.com/auth/cloud-platform"]
    )

    # 2) 建立 Google Gen AI 的 client，後端指定走 Vertex AI
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="us-central1",   # 你 Gemini 用的 region，照你專案調整
        credentials=scoped_credentials,
        http_options=genai_types.HttpOptions(api_version="v1"),
    )

    # 3) 把 Search 找到的段落組成 context
    if not segments and summary_obj and getattr(summary_obj, "summary_text", None):
        # 沒 segment，退而求其次用 Search 的 summary
        context_text = summary_obj.summary_text
        sources_index = "（本次僅使用摘要內容，未提供逐頁來源。）"
    else:
        context_parts = []
        sources_index_lines = []

        for idx, seg in enumerate(segments, start=1):
            title = seg.get("source_title") or "未命名文件"
            page = seg.get("page")
            page_str = f"在{title} 的第 {page} 頁附近" if page is not None else "頁碼未知"

            # 給模型看的「來源標頭＋內容」
            context_parts.append(
                f"檔名：{title}；頁碼：{page_str}\n{seg.get('text')}"
            )

            # 再額外做一份清楚的索引表，方便它在答案中引用
            sources_index_lines.append(f"{title}，{page_str}")

        context_text = "\n\n".join(context_parts)
        sources_index = "\n".join(sources_index_lines)

    prompt = f"""
        你是壽險與家族傳承規劃顧問，熟悉分紅壽險、保障結構設計與後備持有人 / 被保人安排。
        以下是從保險公司產品手冊與相關文件中擷取出來的段落（可能來自不同商品）：
        【資料來源片段】
        {context_text}

        【來源索引表】
        以下是各來源編號對應的檔名與大約頁碼：
        {sources_index}

        下面是客戶的實際情境與需求：
        【客戶問題】
        {user_question}
        請你根據上述「資料來源片段」中的內容，並搭配一般合理的壽險與家族傳承規劃原則，回覆最適合客戶問題的建議方案，供內部討論草案使用。
        
        回答時請務必引用「來源索引表」中的檔名與頁碼，說明你的建議來自哪些文件與位置。
        禁止任何前言、問候、自我介紹、後序等贅述，直接切入重點回答客戶問題。
        回答使用一般文字格式，不要使用任何標記語言（例如 Markdown）。
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text

def query_knowledge_base(user_question: str) -> str:
    print(f"INFO: 開始使用 Vertex AI Search + Gemini 查詢，問題: {user_question}")

    try:
        # 1) 讀取憑證
        credentials = service_account.Credentials.from_service_account_file(
            CREDENTIALS_PATH
        )
        # 2) 初始化 Search client
        client, serving_config = _init_discovery_client(credentials)
        # 3) 先用 Search 找出相關段落
        segments, summary_obj = _search_segments(
            client, serving_config, user_question, max_results=5, top_segments=12
        )
        if not segments and not (summary_obj and summary_obj.summary_text):
            print("WARN: 找不到任何相關段落或摘要。")
            return "根據目前索引到的文件，找不到與此問題足夠相關的內容，請確認資料庫或換個問法。"
        # 4) 再用 Gemini 產生最終規劃建議
        answer = _generate_with_gemini(credentials, user_question, segments, summary_obj)
        print("INFO: 查詢與回答已完成。")
        return answer

    except Exception as e:
        print(f"ERROR: 呼叫 Vertex AI Search 或 Gemini 時發生錯誤: {e}")
        return "抱歉，查詢或生成回答時發生了錯誤，請檢查 API 設定與 Data Store。"


# --- 用於直接測試此腳本 ---
if __name__ == "__main__":
    test_question = (
        "吳家姊弟，姐姐46歲有兩個未成年小孩、弟弟44歲單身沒有小孩，"
        "家族的錢是一起處理的，我想要規劃姊弟各一張分紅壽險，"
        "目標姊弟管理保單，並且要互相cover(後備持有人、後備被保人)，"
        "並在未來可以永續傳承給姐姐的小孩，或是未來弟弟的小孩，"
        "妳會推薦哪個商品，為什麼?他們該怎麼投保才能達到目標?"
    )
    # test_question = "保誠販售甚麼產品？"

    print("--- 開始獨立測試 gcp_search.py ---")
    final_answer = query_knowledge_base(test_question)
    print("\n--- 最終結果 ---")
    print(final_answer)
    print("--- 測試結束 ---")
