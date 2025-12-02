import os
from typing import Dict, List
from google import genai
from google.genai import types
from dotenv import load_dotenv

# load_dotenv() # 移除全域載入，由 main.py 或其他地方控制，或者保留也無妨，但 client 不再這裡 init

def _build_user_prompt(title: str, portrait: str, question: str, history: List[Dict[str, str]] | None = None) -> str:
    """建立用於問答的提示"""
    history_context = ""
    if history:
        role_map = {
            "user": "提問者",
            "assistant": "機器人",
            "discussion": "團隊討論"
        }
        transcript_lines = []
        for msg in history:
            role = msg.get("role", "unknown")
            if role == 'discussion':
                transcript_lines.append(f"- {msg.get('content', '')}")
            else:
                speaker = role_map.get(role, "發言者")
                transcript_lines.append(f"{speaker}: {msg.get('content', '')}")
        
        transcript = "\n".join(transcript_lines)
        history_context = (
            f"以下是過去的對話歷史與摘要，請作為你回答的參考：\n"
            f"---\n{transcript}\n---\n\n"
        )

    return (
        f"您好，您是公司內部成員的助理，正在協助團隊針對客戶進行客觀分析與討論。\n"
        f"請根據以下客戶畫像（特別是客戶的人格特質、著重事項及目前資金配置），並參考對話歷史，以客觀公正的立場，簡潔地回答以下問題。\n"
        f"無須前言以及後述。\n\n"
        f"{history_context}"
        f"客戶名稱：{title}\n"
        f"客戶畫像（節錄）：\n{portrait}\n\n"
        f"問題：{question}\n"
        f"輸出格式："
        f"1) 人格特質、個性：[簡述客戶的人格特質]"
        f"2) 客戶著重事項：[簡述客戶目前最著重的事項]"
        f"3) 目前資金配置：[簡述客戶目前的資金配置狀況]"
        f"4) 需特別注意事項：[列出與客戶互動時需特別注意的事項]"
        f"5) {question} 回覆：[針對問題的簡短客觀回答]"
    )

async def answer_question(client: genai.Client, title: str, portrait: str, question: str, history: List[Dict[str, str]] | None = None) -> str:
    """
    呼叫 Gemini API 回答問題。
    """
    prompt = _build_user_prompt(title, portrait, question, history)
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"呼叫 Gemini 發生錯誤：{e}"

async def summarize_conversation(client: genai.Client, title: str, history: List[Dict[str, str]]) -> str:
    """呼叫 Gemini API 產生對話摘要"""
    role_map = {
        "user": "提問者",
        "assistant": "機器人",
        "discussion": "團隊討論"
    }

    transcript_lines = []
    for msg in history:
        role = msg.get("role", "unknown")
        if role == 'discussion':
            transcript_lines.append(f"- {msg.get('content', '')}")
        else:
            speaker = role_map.get(role, "發言者")
            transcript_lines.append(f"{speaker}: {msg.get('content', '')}")

    transcript = "\n".join(transcript_lines)

    prompt = (
        f"客戶名稱：{title}\n\n"
        f"這是一段關於此客戶的對話歷史紀錄，其中包含了團隊成員的討論和與機器人的問答。\n"
        f"---\n{transcript}\n---\n"
        f"任務：請將以上整段對話紀錄（包含問答和討論）整理成一份重點摘要，總結團隊的發現、關鍵問題點和最終結論。\n"
        f"禁止任何前言、問候、自我介紹、後序等贅述，直接切入重點回答客戶問題。"
        f"回答使用一般文字格式，輸出僅使用純文字，不要使用任何 Markdown 語法或是 HTML 語法、也不使用任何標記符號 (例如 *、#、**)。"
        f"格式請用項目符號（bullet points）。"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"產生摘要時發生錯誤：{e}"

async def refine_summary(client: genai.Client, current_summary: str, instruction: str) -> str:
    """
    根據使用者的指令修改現有的摘要。
    """
    prompt = (
        f"任務：請根據使用者的「修改指令」，對目前的「摘要草稿」進行修訂。\n"
        f"規則：\n"
        f"1. 請保持原本的摘要格式（項目符號 bullet points）。\n"
        f"2. 只進行指令要求的修改（例如刪除某點、補充某點、修正語氣等），盡量保留其他未被要求修改的內容。\n"
        f"3. 直接輸出修訂後的完整摘要，不要包含任何前言或解釋。\n\n"
        f"--- 摘要草稿 ---\n"
        f"{current_summary}\n"
        f"--- 修改指令 ---\n"
        f"{instruction}\n"
        f"--- 結束 ---\n\n"
        f"修訂後的摘要："
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"修訂摘要時發生錯誤：{e}"

async def summarize_segment(client: genai.Client, segment: List[Dict[str, str]]) -> str:
    """將一個對話片段總結成滾動摘要"""
    role_map = {
        "user": "提問者",
        "assistant": "機器人",
        "discussion": "團隊討論"
    }
    transcript_lines = []
    for msg in segment:
        role = msg.get("role", "unknown")
        if role == 'discussion':
            transcript_lines.append(f"- {msg.get('content', '')}")
        else:
            speaker = role_map.get(role, "發言者")
            transcript_lines.append(f"{speaker}: {msg.get('content', '')}")

    transcript = "\n".join(transcript_lines)

    prompt = (
        f"你是一個專業的會議記錄員，你的任務是將以下的對話片段整理成一份簡潔、客觀的「中繼摘要」。\n"
        f"這份摘要將取代原始對話，用於後續的討論，所以請務必保留所有關鍵問題、決策和發現。\n"
        f"請使用條列式（bullet points）來呈現重點。\n\n"
        f"--- 對話片段開始 ---\n"
        f"{transcript}\n"
        f"--- 對話片段結束 ---\n\n"
        f"請開始生成中繼摘要："
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return f"【階段性摘要】\n{response.text}"
    except Exception as e:
        return f"生成階段性摘要時發生錯誤：{e}"

async def answer_with_grounding(client: genai.Client, question: str):
    """
    使用 Google 搜尋作為 grounding tool 來回答問題。
    """
    print(question)
    try:
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=question,
            config=config 
        )
        return response.text
    except Exception as e:
        return f"使用 grounding tool 查詢時發生錯誤：{e}"

async def extract_product_from_query(client: genai.Client, query: str) -> str:
    """從使用者問題中提取產品名稱"""
    prompt = (
        f"任務：從以下「使用者問題」中，僅提取出公司名或公司產品名的完整名稱。\n"
        f"規則：\n"
        f"1. 只回傳名稱本身，不要包含任何多餘的文字、引號或解釋。\n"
        f"2. 如果有兩個以上的公司名或公司產品名，請用半形逗號分開 (例如：公司A, 公司產品名B)。\n"
        f"3. 如果沒有提到任何具體的公司名稱或是公司產品名，請回傳一個空字串。\n\n"
        f"--- 使用者問題 ---\n"
        f"{query}\n"
        f"--- 結束 ---\n\n"
        f"公司名稱或公司產品名："
    )
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        # 清理輸出，移除多餘的引號或換行
        product_name = response.text.strip().strip('"').strip("'")
        print(f"DEBUG: Extracted product name: '{product_name}'")
        return product_name
    except Exception as e:
                print(f"從查詢中提取產品名稱時發生錯誤：{e}")
                return ""
        
async def extract_keywords_from_query(client: genai.Client, query: str) -> List[str]:
    """從使用者問題中提取關鍵字列表"""
    prompt = (
        f"任務：從以下「使用者問題」中，提取出與保險相關的核心關鍵字。\n"
        f"規則：\n"
        f"1. 關鍵字應包含：角色（如：要保人、被保人、受益人、後備持有人）、事件（如：過世、繼承、變更、理賠）、概念（如：直系親屬、豁免保費）等。\n"
        f"2. 只回傳以逗號分隔的關鍵字列表，不要有任何多餘的文字、引號或解釋。\n"
        f"3. 如果問題很模糊或沒有可識別的關鍵字，請回傳一個空字串。\n\n"
        f"--- 使用者問題 ---\n"
        f"{query}\n"
        f"--- 結束 ---\n\n"
        f"關鍵字列表："
    )
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        keywords_str = response.text.strip()
        if not keywords_str:
            return []

        keywords = [k.strip() for k in keywords_str.split(',')]
        print(f"DEBUG: Extracted keywords: {keywords}")
        return keywords
    except Exception as e:
        print(f"從查詢中提取關鍵字時發生錯誤：{e}")
        return []

def generate_with_gemini(client: genai.Client, user_question, segments, summary_obj=None, product_filter: str = None, extra_context: str = None):
    """
    使用『Google Gen AI SDK』呼叫 Vertex 上的 Gemini。
    Args:
        client: genai.Client 實例
        user_question: 使用者的原始問題
        segments: 搜尋到的文件段落
        summary_obj: 搜尋結果的摘要物件 (可選)
        product_filter: 產品過濾器 (可選)
        extra_context: 額外的上下文資訊 (例如客戶資料、對話歷史)，通常來自 search_query 的前半部
    """
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
        5. 禁止任何前言、問候、自我介紹、後序等贅述，直接切入重點回答客戶問題。
        6. 回答使用一般文字格式，使用純文字輸出，不要使用任何標記語言（例如 Markdown）、也不使用任何標記符號 (例如 *、#）。
        7. 回答必須使用繁體中文。
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
        