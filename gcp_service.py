import os
import re
import numpy as np
from generate import generate_with_gemini
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from google.oauth2 import service_account
from google.cloud import storage

from google import genai
from google.genai import types as genai_types
from google.genai.types import EmbedContentConfig

class GcpService:
    def __init__(self, project_id: str, location: str, engine_id: str, credentials_path: str, bucket_name: str):
        self.project_id = project_id
        self.location = location
        self.engine_id = engine_id
        self.credentials_path = credentials_path
        self.bucket_name = bucket_name
        self.credentials = service_account.Credentials.from_service_account_file(self.credentials_path)
        self.genai_client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location="us-central1",
            credentials=self.credentials.with_scopes(["https://www.googleapis.com/auth/cloud-platform"]),
            http_options=genai_types.HttpOptions(api_version="v1"),
        )
        self.client, self.serving_config = self._init_discovery_client()
        self.storage_client = storage.Client(credentials=self.credentials, project=project_id)
        self.product_files = self._fetch_all_filenames()
        self.file_index = self._build_filename_index()

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
        
    def _get_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        """呼叫 Vertex AI 取得向量"""
        try:
            response = self.genai_client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=EmbedContentConfig(task_type=task_type)
            )
            return response.embeddings[0].values
        except Exception as e:
            print(f"ERROR: 計算 Embedding 失敗 ({text}): {e}")
            return []
        
    def _build_filename_index(self):
        """
        預先計算所有檔名的 Embedding 並存入記憶體。
        這會在程式啟動時執行一次。
        """
        print("INFO: 正在建立檔案名稱的語意向量索引 (Embedding Index)...")
        index = []
        for file_info in self.product_files:
            vector = self._get_embedding(file_info["clean_name"], task_type="RETRIEVAL_DOCUMENT")
            if vector:
                index.append({
                    "info": file_info,
                    "vector": np.array(vector)
                })
        
        print(f"INFO: 向量索引建立完成，共 {len(index)} 筆。")
        return index

    def _match_filenames_by_vector(self, keyword: str, top_k: int = 3, threshold1: float = 0.55) -> list[tuple[float, dict]]:
        """
        使用向量相似度 (Cosine Similarity) 找出最相關的檔案。
        """
        if not keyword or not self.file_index:
            return []
        query_vector = np.array(self._get_embedding(keyword, task_type="RETRIEVAL_QUERY"))
        if query_vector.size == 0:
            return []
        matches = []
        norm_q = np.linalg.norm(query_vector)
        for item in self.file_index:
            file_vector = item["vector"]
            norm_f = np.linalg.norm(file_vector)
            if norm_q == 0 or norm_f == 0:
                score = 0.0
            else:
                score = np.dot(query_vector, file_vector) / (norm_q * norm_f)
            if score >= threshold1:
                matches.append((score, item["info"]))

        high_confidence_matches = [m for m in matches if m[0] >= 0.7]
        if high_confidence_matches:
            final_matches = high_confidence_matches
        else:
            final_matches = matches
        final_matches.sort(key=lambda x: x[0], reverse=True)
        
        print(f"DEBUG: '{keyword}' 的向量匹配結果:")
        for score, info in final_matches[:top_k]:
            print(f"  - [{score:.4f}] {info['clean_name']}")

        return final_matches[:top_k]

    def _search_segments(self, user_question, filter_str: str = None, product_filter: str = None, search_query: str = None, max_results=5, top_segments=12, allowed_filenames: list[str] = None):
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
        final_query = search_query or user_question
        request = discoveryengine.SearchRequest(
            serving_config=self.serving_config,
            query=final_query,
            page_size=max_results,
            content_search_spec=content_search_spec,
            filter=filter_str,
        )
        try:
            search_pager = self.client.search(request)
            response = next(search_pager.pages)
            print(f"DEBUG: Vertex Search found {len(response.results)} documents.")
        except Exception as e:
            print(f"Vertex Search Error: {e}")
            return [], None

        segments_info = []
        for i, result in enumerate(response.results, start=1):
            doc = result.document
            derived_data = dict(doc.derived_struct_data) if doc.derived_struct_data else {}
            
            raw_title = (derived_data.get("title") or derived_data.get("document_title"))
            link = derived_data.get("link")
            filename_title = None
            if link:
                path_part = link
                if link.startswith("gs://"):
                    path_part = link.split("/", 1)[-1]
                filename = os.path.basename(path_part)
                filename_title = os.path.splitext(filename)[0]
            
            title = f"{filename_title}（{raw_title}）" if (raw_title and filename_title) else (raw_title or filename_title or "未命名文件")

            if allowed_filenames:
                if filename_title not in allowed_filenames:
                    print(f"DEBUG: Skipping result '{title}' as it is not in allowed filenames.")
                    continue

            segments = derived_data.get("extractive_segments", [])
            for seg in segments:
                text = seg.get("content", "")
                if not text: continue
                page = seg.get("pageNumber")
                score = seg.get("score") or seg.get("confidenceScore") or 1.0
                segments_info.append({"text": text, "source_title": title, "page": page, "score": float(score)})

        segments_info.sort(key=lambda x: x["score"], reverse=True)
        print(f"DEBUG: _search_segments extracted {len(segments_info)} segments.")
        return segments_info[:top_segments], getattr(response, "summary", None)

    def query_knowledge_base(self, user_question: str, product_filter: str | None = None, search_query: str | None = None, conversation_history: str | None = None) -> str:
        print(f"INFO: 開始查詢，Filter Term: {product_filter}, User Question: {user_question}")

        final_segments = []
        display_names_all = []

        TOTAL_SEGMENT_TARGET = 12
        if product_filter:
            clean_filter = product_filter.replace('"', '').replace("'", "")
            keywords = [k.strip() for k in re.split(r'[,\n]+', clean_filter) if k.strip()]
            
            if keywords:
                quota_per_product = max(1, TOTAL_SEGMENT_TARGET // len(keywords))
                print(f"INFO: 偵測到 {len(keywords)} 個產品，每個產品分配 {quota_per_product} 個段落配額。")

                for kw in keywords:
                    found = self._match_filenames_by_vector(kw, top_k=3) 
                    if not found: continue

                    valid_infos = []
                    kw_chars = set(kw.lower())
                    for score, info in found:
                        clean_name = info['clean_name'].lower()
                        overlap_count = sum(1 for char in kw_chars if char in clean_name)
                        coverage = overlap_count / len(kw_chars) if kw_chars else 0.0
                        
                        # 修改邏輯：如果向量分數很高 (>= 0.7)，則放寬字元匹配門檻 (允許簡繁體差異)
                        if score >= 0.7:
                            threshold = 0.5
                        else:
                            threshold = 1.0 if len(kw_chars) < 3 else 0.6 
                        
                        if coverage >= threshold:
                            valid_infos.append(info)
                    
                    if not valid_infos: continue
                    
                    valid_names = [info['clean_name'] for info in valid_infos]
                    display_names_all.extend(valid_names)

                    files_context_str = " ".join([f'{name}' for name in valid_names])
                    mining_question = f"請詳細說明 {kw} 的產品特色、保障範圍與條款細節。"
                    
                    safe_user_query = user_question.replace("比較", "說明").replace("區別", "介紹").replace("差別", "內容")
                    single_product_query = f"{files_context_str} {kw} {safe_user_query} 產品手冊 條款"
                    
                    print(f"INFO: 執行子查詢 (資料探勘模式):")
                    print(f"  - Mining Question: {mining_question}")
                    print(f"  - Search Query: {single_product_query}")
                    # print(f"  - Filter: {filter_str}")

                    segs, _ = self._search_segments(
                        user_question=mining_question,
                        search_query=single_product_query,
                        filter_str=None,
                        max_results=5, 
                        top_segments=quota_per_product,
                        allowed_filenames=valid_names
                    )
                    final_segments.extend(segs)

        if not final_segments:
            print("INFO: 無法執行多產品分別搜尋，切換回單次搜尋模式。")
            final_search_query = search_query or user_question
            allowed_filenames = None

            vector_matches = self._match_filenames_by_vector(user_question, top_k=5)
            if vector_matches:
                print(f"INFO: 單次搜尋模式下，透過向量搜尋找到 {len(vector_matches)} 個相關檔案。")
                allowed_filenames = [info['clean_name'] for _, info in vector_matches]
                print(allowed_filenames)
                display_names_all.extend(allowed_filenames)

            final_segments, summary_obj = self._search_segments(
                user_question,
                search_query=final_search_query,
                filter_str=None,
                max_results=5, 
                top_segments=TOTAL_SEGMENT_TARGET,
                allowed_filenames=allowed_filenames 
            )
            if not final_segments and not (summary_obj and summary_obj.summary_text):
                return "根據目前索引到的文件，找不到相關內容。"

        print(f"DEBUG: Final segments count: {len(final_segments)}")
        source_titles = set(s.get('source_title') for s in final_segments)
        print(f"DEBUG: Sources in final_segments: {source_titles}")

        context_info = user_question
        if display_names_all:
            unique_names = list(set(display_names_all))
            prefix = "系統已分別檢索以下產品文件：\n"
            context_info = prefix + "\n".join(unique_names) + f"\n\n原始問題：{user_question}"
        
        if conversation_history:
            context_info = f"{context_info}\n\n【對話歷史與上下文】\n{conversation_history}"

        answer = generate_with_gemini(
            client=self.genai_client,
            user_question=user_question, 
            segments=final_segments, 
            summary_obj=None,
            product_filter=product_filter,
            extra_context=context_info
        )
        return answer