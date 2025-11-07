import os
from typing import Dict, List
from google import genai
from google.genai import types

# 初始化 Gemini 客戶端
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
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
        f"1) 人格特質：[簡述客戶的人格特質]"
        f"2) 客戶著重事項：[簡述客戶目前最著重的事項]"
        f"3) 目前資金配置：[簡述客戶目前的資金配置狀況]"
        f"4) {question} 回覆：[針對問題的簡短客觀回答]"
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
    # 為不同角色設定更清晰的標籤
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

async def answer_with_grounding(question: str) -> str:
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
