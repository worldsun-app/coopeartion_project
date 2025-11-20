import os
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from google.oauth2 import service_account

from google import genai
from google.genai import types as genai_types

class GcpService:
    def __init__(self, project_id: str, location: str, engine_id: str, credentials_path: str):
        self.project_id = project_id
        self.location = location
        self.engine_id = engine_id
        self.credentials_path = credentials_path
        self.credentials = service_account.Credentials.from_service_account_file(self.credentials_path)
        self.client, self.serving_config = self._init_discovery_client()

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

    def _search_segments(self, user_question, product_filter: str = None, max_results=5, top_segments=12):
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
        search_query = user_question
        if product_filter:
            search_query = f"{user_question} {product_filter}"

        request = discoveryengine.SearchRequest(
            serving_config=self.serving_config,
            query=search_query,
            page_size=max_results,
            content_search_spec=content_search_spec,
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

        # 構建 Prompt
        # 如果有 extra_context (通常包含客戶資料或討論摘要)，我們要允許模型進行「推薦」
        
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
        print(f"INFO: 開始使用 Vertex AI Search + Gemini 查詢，問題: {user_question}")
        search_text = search_query or user_question

        try:
            # 3) 先用 Search 找出相關段落
            # 注意：這裡我們用 search_text (包含 context) 去做搜尋，這樣比較容易搜到跟 context 相關的關鍵字
            segments, summary_obj = self._search_segments(
                search_text, product_filter=product_filter, max_results=5, top_segments=12
            )
            
            # 如果搜尋不到，但我們有 context，有時候還是可以嘗試回答 (例如只是閒聊)，
            # 但這裡是產品資料庫，如果沒搜到產品資料，通常就無法推薦。
            # 不過為了保險起見，如果完全沒 segments，還是回傳找不到。
            if not segments and not (summary_obj and summary_obj.summary_text):
                print("WARN: 找不到任何相關段落或摘要。")
                return "根據目前索引到的文件，找不到與此問題足夠相關的內容，請確認資料庫或換個問法。"

            # 4) 再用 Gemini 產生最終規劃建議
            # 這裡我們把 search_query 當作 extra_context 傳進去，
            # 因為 search_query 通常長這樣： "請參考...討論... \n 搜尋並回答: {user_question}"
            # 雖然有點重複，但讓 LLM 清楚知道這是背景資訊是有幫助的。
            # 為了更乾淨，我們可以嘗試只把 user_question 以外的部分當 context，
            # 但簡單起見，直接傳 search_query 讓 prompt 裡的 extra_context 處理 (雖然 prompt 裡我是分開欄位的)
            # 修正：search_query 包含了 context + question。
            # 為了避免 prompt 重複太多，我們可以把 search_query 當作 context 傳入，
            # 或者在 telegram_handler 裡就分開傳。
            # 鑑於介面限制，這裡我們直接把 search_query 視為 "包含 Context 的完整查詢字串"，
            # 但為了 Prompt 結構漂亮，我們把它傳給 extra_context 參數。
            
            answer = self._generate_with_gemini(
                user_question=user_question, 
                segments=segments, 
                summary_obj=summary_obj, 
                product_filter=product_filter,
                extra_context=search_query # 將完整的搜尋字串(含context)作為背景資訊給 LLM 參考
            )
            return answer

        except Exception as e:
            print(f"ERROR: 呼叫 Vertex AI Search 或 Gemini 時發生錯誤: {e}")
            return "抱歉，查詢或生成回答時發生了錯誤，請檢查 API 設定與 Data Store。"
