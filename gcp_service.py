import os
from rapidfuzz import fuzz
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from google.oauth2 import service_account
from google.cloud import storage

from google import genai
from google.genai import types as genai_types

class GcpService:
    def __init__(self, project_id: str, location: str, engine_id: str, credentials_path: str, bucket_name: str):
        self.project_id = project_id
        self.location = location
        self.engine_id = engine_id
        self.credentials_path = credentials_path
        self.bucket_name = bucket_name
        self.credentials = service_account.Credentials.from_service_account_file(self.credentials_path)
        self.client, self.serving_config = self._init_discovery_client()
        self.storage_client = storage.Client(credentials=self.credentials, project=project_id)
        self.product_files = self._fetch_all_filenames()

    def _init_discovery_client(self):
        """初始化 Vertex AI Search 的 client。"""
        if self.location == "global":
            api_endpoint = "discoveryengine.googleapis.com"
        else:
            api_endpoint = f"{self.location}-discoveryengine.googleapis.com"

        client_options = ClientOptions(api_endpoint=api_endpoint)

        client = discoveryengine.SearchServiceClient(
            credentials=self.credentials,
            client_options=client_options,
        )

        serving_config = (
            f"projects/{self.project_id}"
            f"/locations/{self.location}"
            f"/collections/default_collection"
            f"/engines/{self.engine_id}"
            f"/servingConfigs/default_search"
        )
        return client, serving_config
    
    def _fetch_all_filenames(self):
        """從 GCS Bucket 獲取所有檔案名稱，用於模糊匹配"""
        try:
            blobs = self.storage_client.list_blobs(self.bucket_name)
            # 提取檔名，去除副檔名，例如 "重疾-友邦香港愛伴航系列 - 產品手冊.pdf" -> "重疾-友邦香港愛伴航系列 - 產品手冊"
            # 同時保留原始完整檔名以便後續對照（視 Vertex AI Search 的 Title 欄位而定，通常是檔名）
            file_list = []
            for blob in blobs:
                filename = os.path.basename(blob.name)
                name_without_ext = os.path.splitext(filename)[0]
                file_list.append({
                    "blob_name": blob.name,
                    "clean_name": name_without_ext
                })
            return file_list
        except Exception as e:
            print(f"ERROR: 無法讀取 GCS Bucket: {e}")
            return []
        
    def _match_filenames_fuzzy(self, keyword: str) -> list[dict]:
        """
        使用多種策略回傳匹配到的完整檔案資訊列表。
        策略順序：
        1. 類別標籤匹配 (針對 '分紅-xxx' 格式)
        2. Fuzzy Token Set Ratio (針對同音異字或部分匹配)
        """
        if not keyword:
            return []
        
        matches = []
        clean_keyword = keyword.strip().replace('"', '').replace("'", "")
        
        # 設定模糊比對門檻
        THRESHOLD = 60 

        for file_info in self.product_files:
            clean_name = file_info["clean_name"]
            
            # --- 策略 1: 類別標籤匹配 (針對您的檔名結構優化) ---
            if "-" in clean_name:
                # 取得減號前的部分當作類別 (例如 "分紅", "重疾")
                category_tag = clean_name.split("-")[0].strip()
                # 只有當標籤長度大於等於2 (避免匹配到無意義單字) 且 關鍵字包含此標籤時
                if len(category_tag) >= 2 and category_tag in clean_keyword:
                    print(f"DEBUG: Category Match '{category_tag}' found in keyword '{clean_keyword}' -> Match '{clean_name}'")
                    matches.append(file_info)
                    continue # 匹配成功，跳過後續檢查，處理下一個檔案
            # --- 策略 2: RapidFuzz Token Set Ratio (原本的模糊比對) ---
            score = fuzz.token_set_ratio(clean_keyword, clean_name)
            if score >= THRESHOLD:
                print(f"DEBUG: Fuzzy Match (token_set) '{clean_keyword}' vs '{clean_name}' = {score}")
                matches.append(file_info)
                continue
        # 去重邏輯
        unique_matches = {m["blob_name"]: m for m in matches}.values()
        return list(unique_matches)

    def _search_segments(self, user_question, filter_str: str = None, product_filter: str = None, search_query: str = None, max_results=5, top_segments=12):
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

            "禁止任何前言、問候、自我介紹、後序等贅述，直接切入重點回答客戶問題。"
            "回答使用一般文字格式，不要使用任何標記語言（例如 Markdown）、也不使用任何標記符號 (例如 *、#)。"
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
        if not search_query:
            search_query = user_question
            if product_filter:
                search_query = f"{user_question} {product_filter}"

        request = discoveryengine.SearchRequest(
            serving_config=self.serving_config,
            query=search_query,
            page_size=max_results,
            content_search_spec=content_search_spec,
            filter=filter_str,
        )
        search_pager = self.client.search(request)
        response = next(search_pager.pages)
        # print("DEBUG: total results =", len(response.results))

        segments_info = []
        for i, result in enumerate(response.results, start=1):
            doc = result.document
            derived_data = dict(doc.derived_struct_data) if doc.derived_struct_data else {}
            # print(f"=== Result {i} derived_struct_data ===")
            # print(derived_data)
            raw_title = (derived_data.get("title")or derived_data.get("document_title"))
            link = derived_data.get("link")
            filename_title = None
            if link:
                path_part = link
                if link.startswith("gs://"):
                    path_part = link.split("/", 1)[-1]
                filename = os.path.basename(path_part)
                filename_title = os.path.splitext(filename)[0]
            if raw_title and filename_title:
                title = f"{filename_title}（{raw_title}）"
            else:
                title = raw_title or filename_title or "未命名文件"

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
        return segments_info, getattr(response, "summary", None)

    def _generate_with_gemini(self, user_question, segments, summary_obj=None, product_filter: str = None, extra_context: str = None):
        """
        使用『Google Gen AI SDK』呼叫 Vertex 上的 Gemini。
        Args:
            user_question: 使用者的原始問題
            segments: 搜尋到的文件段落
            summary_obj: 搜尋結果的摘要物件 (可選)
            product_filter: 產品過濾器 (可選)
            extra_context: 額外的上下文資訊 (例如客戶資料、對話歷史)，通常來自 search_query 的前半部
        """
        scoped_credentials = self.credentials.with_scopes(
            ["https://www.googleapis.com/auth/cloud-platform"]
        )
        client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location="us-central1",
            credentials=scoped_credentials,
            http_options=genai_types.HttpOptions(api_version="v1"),
        )
        if not segments and summary_obj and getattr(summary_obj, "summary_text", None):
            context_text = summary_obj.summary_text
        else:
            context_parts = [
                f"來源文件：{seg.get('source_title', '未命名')}\n頁碼：{seg.get('page', '未知')}\n內容：\n{seg.get('text')}"
                for seg in segments
            ]
            context_text = "\n\n---\n\n".join(context_parts)
        base_instruction = """
            你是一位專業的保險顧問。
            你的任務是根據提供的「產品資料片段」來回答客戶問題。
        """
        if extra_context:
            base_instruction += f"""
            【參考資訊】
            以下是客戶的背景資料或團隊討論摘要，請參考這些資訊來判斷適合的產品：
            ---
            {extra_context}
            ---
            請根據【參考資訊】中的客戶需求，從下方的【產品資料片段】中挑選最合適的產品或條款進行推薦與說明。
            """
        base_instruction += """
            【回答規則】
            1. 當你引用【產品資料片段】中的具體條款、數據或產品特色時，必須在句末標註出處，格式為 (產品名稱, 第XX頁)。
            2. 如果是根據【參考資訊】進行的邏輯推演或建議（例如：「因為客戶有XX需求，所以建議...」），則不需要標註產品出處，但必須說明理由。
            3. 如果【產品資料片段】中完全沒有相關資訊可以支持你的回答或建議，請誠實告知：「根據目前的產品資料庫，無法找到合適的產品資訊。」
            4. 嚴禁捏造產品內容。
            5. "禁止任何前言、問候、自我介紹、後序等贅述，直接切入重點回答客戶問題。"
            6. "回答使用一般文字格式，不要使用任何標記語言（例如 Markdown）、也不使用任何標記符號 (例如 *、#）。"
        """
        
        if product_filter:
            base_instruction += f"\n注意：你目前被限制只能討論「{product_filter}」相關的內容。"
        prompt = f"""
            {base_instruction}
            --- 【產品資料片段】 ---
            {context_text}
            ---
            【客戶問題】
            {user_question}
            請開始回答：
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text

    def query_knowledge_base(self, user_question: str, product_filter: str | None = None, search_query: str | None = None) -> str:
        print(f"INFO: 開始查詢，Filter Term: {product_filter}, User Question: {user_question}")
        
        matched_files = []
        display_names = []
        final_search_query = search_query or user_question # 預設查詢

        if product_filter:
            keywords = [k.strip() for k in product_filter.replace('"', '').replace("'", "").split(',')]
            
            all_matched_files = []
            for kw in keywords:
                if kw:
                    # 對每個分割後的關鍵字進行模糊匹配
                    found = self._match_filenames_fuzzy(kw)
                    all_matched_files.extend(found)
            
            # 去重：避免重複加入相同的檔案 (根據 blob_name 去重)
            if all_matched_files:
                unique_matches = {m["blob_name"]: m for m in all_matched_files}.values()
                matched_files = list(unique_matches)

                matched_names = [f['clean_name'] for f in matched_files]
                display_names = matched_names
                print(f"INFO: 關鍵字 '{product_filter}' 本地匹配到檔案: {matched_names}")
                
                files_context_str = " ".join(matched_names)
                
                final_search_query = f"{files_context_str} {user_question}"
                print(f"INFO: Search Query: {final_search_query}")
            else:
                print(f"WARN: 關鍵字 '{product_filter}' 沒有匹配到任何本地檔案，將使用原始查詢。")

        try:
            # 注意：這裡 filter_str 永遠傳 None，避免 400 錯誤
            segments, summary_obj = self._search_segments(
                user_question=user_question,
                search_query=final_search_query, # 傳入組裝好的 Query
                filter_str=None, # 這裡設為 None
                max_results=5, 
                top_segments=12
            )

            if not segments and not (summary_obj and summary_obj.summary_text):
                msg = "根據目前索引到的文件，找不到相關內容。"
                if display_names:
                    msg += f" (已嘗試搜尋：{', '.join(display_names)}，但未發現具體內容)"
                return msg

            # 雖然沒有用 Filter，但我們還是要告訴 Gemini 我們主要鎖定了哪些檔案
            context_info = user_question
            if display_names:
                prefix = "系統已根據檔名優先檢索以下文件：\n"
                context_info = prefix + "\n".join(display_names) + f"\n\n原始問題：{user_question}"

            answer = self._generate_with_gemini(
                user_question=user_question, 
                segments=segments, 
                summary_obj=summary_obj, 
                product_filter=product_filter,
                extra_context=context_info
            )
            return answer

        except Exception as e:
            print(f"ERROR: 查詢過程發生錯誤: {e}")
            return "抱歉，系統發生錯誤，請檢查後台日誌。"