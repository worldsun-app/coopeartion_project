from __future__ import annotations
import os
import requests
from typing import Dict, Any, List, Tuple, Optional

NOTION_VERSION = os.getenv("NOTION_API_VERSION", "2025-09-03")
BASE = "https://api.notion.com/v1"

class NotionService:
    """
    輕量 Notion REST 封裝（使用新版 Data Source 查詢）
    - get_database()
    - iter_all_pages()
    - find_customer_pages_by_title(name)
    - get_page_portrait_section(page_id, keyword="客戶畫像")
    """
    def __init__(self, api_key: str, database_id: str):
        self.api_key = api_key
        self.database_id = database_id

    # ---------- low-level ----------
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

    def get_database(self) -> Dict[str, Any]:
        r = requests.get(f"{BASE}/databases/{self.database_id}", headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    def _query_data_source(self, data_source_id: str) -> List[Dict[str, Any]]:
        url = f"{BASE}/data_sources/{data_source_id}/query"
        payload: Dict[str, Any] = {"page_size": 100}
        out: List[Dict[str, Any]] = []
        while True:
            r = requests.post(url, headers=self._headers(), json=payload, timeout=30)
            r.raise_for_status()
            res = r.json()
            out.extend(res.get("results", []))
            if res.get("has_more"):
                payload["start_cursor"] = res.get("next_cursor")
            else:
                break
        return out

    def _list_block_children(self, block_id: str) -> list[dict]:
        url = f"{BASE}/blocks/{block_id}/children"
        params = {"page_size": 100}
        out: list[dict] = []
        while True:
            r = requests.get(url, headers=self._headers(), params=params, timeout=30)
            r.raise_for_status()
            res = r.json()
            out.extend(res.get("results", []))
            if res.get("has_more"):
                params["start_cursor"] = res.get("next_cursor")
            else:
                break
        return out
    
    def _list_children_recursive(self, block_id: str, depth: int = 0) -> list[tuple[int, dict]]:
        """
        以 (depth, block) 方式回傳，方便判斷『同層』的下一個標題。
        只遞迴在 has_children=True 的節點（如 toggle/callout/list item 等）
        """
        acc: list[tuple[int, dict]] = []
        for b in self._list_block_children(block_id):
            acc.append((depth, b))
            if b.get("has_children"):
                acc.extend(self._list_children_recursive(b["id"], depth + 1))
        return acc

    def _norm(self, s: str) -> str:
        # 寬鬆比對：「客戶畫像」「客戶畫像：」「一、客戶畫像」都能命中
        return "".join(c for c in (s or "") if c not in " ：:　\t\r\n").lower()

    # ---------- high-level ----------
    def iter_all_pages(self) -> List[Dict[str, Any]]:
        """抓取此 Database 所有 Data Source 的 pages"""
        db = self.get_database()
        data_sources = db.get("data_sources", [])
        pages: List[Dict[str, Any]] = []
        for ds in data_sources:
            pages.extend(self._query_data_source(ds["id"]))
        return pages

    @staticmethod
    def _get_title_from_properties(props: Dict[str, Any]) -> str:
        for k, v in props.items():
            if v.get("type") == "title":
                # join rich_text plain_text
                texts = []
                for t in v.get("title", []):
                    if t.get("plain_text"):
                        texts.append(t["plain_text"])
                return "".join(texts) or k
        return "未命名"

    def find_customer_pages_by_title(self, name: str) -> List[Dict[str, Any]]:
        """以 Title 模糊搜尋（contains）"""
        name = (name or "").strip()
        if not name:
            return []
        pages = self.iter_all_pages()
        hits: List[Dict[str, Any]] = []
        for p in pages:
            title = self._get_title_from_properties(p.get("properties", {}))
            if name in title:
                p["_title"] = title
                hits.append(p)
        return hits

    # ---- portrait extraction ----
    @staticmethod
    def _rich_text_to_plain(rts: list[dict]) -> str:
        out = []
        for rt in rts or []:
            if rt.get("plain_text"):
                out.append(rt["plain_text"])
            elif rt.get("text", {}).get("content"):
                out.append(rt["text"]["content"])
        return "".join(out)
    
    def _text_of_block(self, b: dict) -> str:
        """回傳可閱讀的單行文字（標題/段落/清單/Callout/To-do 等）"""
        t = b.get("type")
        obj = b.get(t, {})
        if t in ("heading_1", "heading_2", "heading_3", "paragraph",
                "bulleted_list_item", "numbered_list_item", "to_do",
                "callout", "quote", "toggle"):
            return self._rich_text_to_plain(obj.get("rich_text", []))
        return ""

    def _extract_section(self, blocks: List[Dict[str, Any]], keyword: str) -> str:
        capturing = False
        lines: List[str] = []
        for b in blocks:
            t = b.get("type")
            obj = b.get(t, {})
            text = ""
            if t in ("heading_1", "heading_2", "heading_3", "paragraph"):
                text = self._rich_text_to_plain(obj.get("rich_text", []))

            if not capturing:
                if keyword in text:
                    capturing = True
                    continue
            else:
                if t in ("heading_1", "heading_2", "heading_3", "divider"):
                    break
                # paragraph & list items
                if t == "paragraph":
                    if text.strip():
                        lines.append(text.strip())
                    else:
                        lines.append("")
                elif t in ("bulleted_list_item", "numbered_list_item"):
                    prefix = "-" if t == "bulleted_list_item" else "1."
                    lines.append(f"{prefix} {self._rich_text_to_plain(obj.get('rich_text', []))}".strip())
        return "\n".join(lines).strip()
    
    def _rich_text_items_to_text(self, items: list[dict]) -> str:
        out = []
        for rt in items or []:
            if rt.get("plain_text"):
                out.append(rt["plain_text"])
            else:
                out.append(rt.get("text", {}).get("content", "") or "")
        return "".join(out)

    def _block_text(self, b: dict) -> str:
        t = b.get("type")
        obj = b.get(t, {})
        if t in ("paragraph","heading_1","heading_2","heading_3",
                "bulleted_list_item","numbered_list_item",
                "to_do","toggle","callout","quote"):
            return self._rich_text_items_to_text(obj.get("rich_text", []))
        return ""

    
    def extract_all_text_from_flat(self, flat: list[tuple[int, dict]]) -> list[str]:
        texts: list[str] = []
        for _, b in flat:
            txt = self._block_text(b)
            if txt:
                texts.append(txt)
        return texts

    def get_page_portrait_section(self, page_id: str) -> str:
        """
        抽取『客戶畫像』段落：
        - 若命中的 block 是「標題/段落」：取其『同層』後續內容直到下一個標題/分隔線
        - 若命中的是「toggle/callout」：取其『子節點』作為畫像內容（遞迴）
        """
        flat = self._list_children_recursive(page_id, depth=0)  # [(depth, block), ...]
        all_lines = self.extract_all_text_from_flat(flat)
        return "\n".join(all_lines).strip() or "（此頁未找到「客戶畫像」內容）"