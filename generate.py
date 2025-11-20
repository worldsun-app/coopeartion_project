import os
from typing import Dict, List
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GEMINI_LOCATION = os.getenv("GEMINI_LOCATION", "us-central1")

try:
    client = genai.Client(vertexai=True,
                        project=PROJECT_ID,
                        location=GEMINI_LOCATION,
                    )
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])
except Exception as e:
    print(f"無法初始化 Gemini Client: {e}")
    client = None

def _build_user_prompt(title: str, portrait: str, question: str) -> str:
    """建立用於問答的提示"""
    return (
        f"您好，您是公司內部成員的助理，正在協助團隊針對客戶進行客觀分析與討論。"
        f"請根據以下客戶畫像（特別是客戶的人格特質、著重事項及目前資金配置），以客觀公正的立場，簡潔地回答以下問題。"
        f"無須前言以及後述。\n\n"
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

async def answer_question(title: str, portrait: str, question: str) -> str:
    """
    呼叫 Gemini API 回答問題。
    """
    prompt = _build_user_prompt(title, portrait, question)
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"呼叫 Gemini 發生錯誤：{e}"

async def summarize_conversation(title: str, history: List[Dict[str, str]]) -> str:
    """呼叫 Gemini API 產生對話摘要"""
    role_map = {
        "user": "提問者",
        "assistant": "機器人",
        "discussion": "團隊討論"
    }

    transcript_lines = []
    for msg in history:
        role = msg.get("role", "unknown")
        # discussion 內容已經包含發言者名稱，不用再加標籤
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
        f"回答使用一般文字格式，不要使用任何標記語言（例如 Markdown）、也不使用任何標記符號 (例如 *、#)。"
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

async def summarize_segment(segment: List[Dict[str, str]]) -> str:
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
        # 在摘要前加上一個標記，方便識別
        return f"【階段性摘要】\n{response.text}"
    except Exception as e:
        return f"生成階段性摘要時發生錯誤：{e}"

async def answer_with_grounding(question: str):
    """
    使用 Google 搜尋作為 grounding tool 來回答問題。
    """
    print(question)
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=question,
            config=config 
        )
        return response.text
    except Exception as e:
        return f"使用 grounding tool 查詢時發生錯誤：{e}"

async def extract_product_from_query(query: str) -> str:
    """從使用者問題中提取產品名稱"""
    prompt = (
        f"任務：從以下「使用者問題」中，僅提取出保險產品或公司的完整名稱。\n"
        f"規則：\n"
        f"1. 產品或公司名稱通常會出現在「在...當中」、「關於...」這類詞語的後面。\n"
        f"2. 只回傳名稱本身，不要包含任何多餘的文字、引號或解釋。\n"
        f"3. 如果沒有提到任何具體的產品或公司名稱，請回傳一個空字串。\n\n"
        f"--- 使用者問題 ---\n"
        f"{query}\n"
        f"--- 結束 ---\n\n"
        f"產品或公司名稱："
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
        
async def extract_keywords_from_query(query: str) -> List[str]:
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
        