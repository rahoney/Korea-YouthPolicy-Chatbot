"""
벡터 + 메타데이터 하이브리드 검색 모듈
-------------------------------------------------
1. 사용자 질문 → (지역·나이·키워드·대분류) 구조화
2. 메타 필터로 후보 정책 축소
3. Chroma + MMR 로 의미 유사 문서 검색
4. 반환: LangChain Document 리스트
"""

from __future__ import annotations

import json, os, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

import torch
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# ────────────────────────────────────────────────
# 1) 상수: 유의어·카테고리·지역 매핑
# ----------------------------------------------
synonym_dict: Dict[str, List[str]] = {
    "주거": ["부동산", "주택", "거주", "임대", "전세", "전월세", "안심주택", "쉐어하우스"],
    "취업": ["일자리", "채용", "구직", "고용", "재직자", "현장면접", "재취업", "구인구직"],
    "창업": ["스타트업", "벤처", "자영업", "창직", "사업", "브랜딩", "푸드트럭"],
    "복지문화": ["복지", "문화", "생활지원", "심리지원", "문화바우처", "청년복지"],
    "보조금": ["지원금", "활동비", "인건비", "참여수당", "창업비", "대출", "생활비", "교통비"],
    "상담": ["맞춤형상담서비스", "심리상담", "멘토링", "코칭", "집단상담", "자기개발"],
    "공공임대주택": ["LH", "SH", "임대주택", "역세권청년주택", "공공분양", "안심주택"],
    "중소기업": ["청년기업", "소기업", "중견기업", "강소기업", "지역기업"],
    "장기미취업청년": ["장기실업", "경단녀", "니트", "사회진입지원"],
    "교육": ["교육", "훈련", "역량개발", "강의", "이러닝", "자격증", "코딩교육", "교육지원"],
    "출산": ["임신", "출산", "출산지원", "육아휴직", "산후", "양육"],
    "결혼": ["혼인", "부부", "신혼", "예비부부", "결혼생활", "신혼부부", "혼례"],
    "육아": ["육아", "양육", "아이돌봄", "보육", "돌봄서비스", "어린이집", "놀이시설"],
    "바우처": ["지원카드", "포인트", "문화바우처", "교육바우처", "복지카드"],
}
user_input_to_main_keyword: Dict[str, str] = {
    syn: main for main, syns in synonym_dict.items() for syn in syns
}

# 주키워드 → 대분류 확장
category_dict: defaultdict[str, List[str]] = defaultdict(list)
category_dict["취업"].extend(["일자리", "교육", "복지문화", "참여권리", "주거"])
category_dict["건강"].extend(["복지문화", "일자리", "교육", "주거", "참여권리"])
category_dict["취약계층 및 금융지원"].extend(["복지문화", "일자리", "교육", "주거", "참여권리"])
category_dict["청년참여"].extend(["참여권리", "복지문화", "일자리", "교육", "주거"])
category_dict["정책인프라구축"].extend(["참여권리", "복지문화", "일자리", "교육"])
category_dict["문화활동"].extend(["복지문화", "참여권리", "교육", "일자리"])
category_dict["미래역량강화"].extend(["교육", "일자리", "복지문화", "참여권리", "주거"])
category_dict["창업"].extend(["일자리", "교육", "복지문화", "참여권리", "주거"])
category_dict["예술인지원"].extend(["복지문화", "일자리", "참여권리", "주거", "교육"])
category_dict["재직자"].extend(["일자리", "복지문화", "교육", "주거", "참여권리"])

REGION_KEYWORDS: Dict[str, str] = {
    k: v for k, v in [
        ("서울","서울"),("서울특별시","서울"),("서울시","서울"),
        ("경기","경기"),("경기도","경기"),
        ("부산","부산"),("부산광역시","부산"),("부산시","부산"),
        ("광주","광주"),("광주광역시","광주"),("광주시","광주"),
        ("대구","대구"),("대구광역시","대구"),("대구시","대구"),
        ("인천","인천"),("인천광역시","인천"),("인천시","인천"),
        ("대전","대전"),("대전광역시","대전"),("대전시","대전"),
        ("울산","울산"),("울산광역시","울산"),("울산시","울산"),
        ("세종","세종"),("세종특별자치시","세종"),("세종시","세종"),
        ("충북","충북"),("충청북도","충북"),
        ("충남","충남"),("충청남도","충남"),
        ("전북","전북"),("전라북도","전북"),("전북특별자치도","전북"),
        ("전남","전남"),("전라남도","전남"),
        ("경북","경북"),("경상북도","경북"),
        ("경남","경남"),("경상남도","경남"),
        ("강원","강원"),("강원도","강원"),("강원특별자치도","강원"),
        ("제주","제주"),("제주도","제주"),("제주특별자치도","제주"),
    ]
}

# ────────────────────────────────────────────────
# 2) NLU & 도우미
# ----------------------------------------------
def extract_region_from_institution(name: str | None) -> str | None:
    """기관명·주소 문자열에서 지역 명칭 추출"""
    if not name:
        return None
    for k, std in REGION_KEYWORDS.items():
        if k in name:
            return std
    return None


def parse_user_query(user_input: str) -> Dict[str, Any]:
    """
    자연어 질문 → region · age_range · matched_keywords · matched_main_categories
    """
    # 주키워드 추출
    tokens = re.findall(r"\w+", user_input)
    matched_kw: Set[str] = {
        user_input_to_main_keyword[t] for t in tokens if t in user_input_to_main_keyword
    }
    matched_cat: Set[str] = {c for kw in matched_kw for c in category_dict.get(kw, [])}

    # 지역
    region = next(
        (std for k, std in REGION_KEYWORDS.items() if k in user_input), None
    )

    # 나이 / 연령대
    age_range: Tuple[int, int] | None = None
    m = re.search(r"((\d{1,2})\s?세)|(\d{1,2}\s?살)|((\d{1,2})대)", user_input)
    if m:
        num = int(re.search(r"\d{1,2}", m.group()).group())
        age_range = (num, num + 9) if "대" in m.group() else (num, num)
    elif "청년" in user_input:
        age_range = (19, 45)

    return {
        "region": region,
        "age_range": age_range,
        "matched_keywords": list(matched_kw),
        "matched_main_categories": list(matched_cat),
    }

# ────────────────────────────────────────────────
# 3) 온통청년 JSON 로드 & 가벼운 전처리
# ----------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "YouthPolicy_data.json"

def _split(field: str) -> List[str]:
    return [
        user_input_to_main_keyword.get(t.strip(), t.strip())
        for t in field.split(",") if t.strip()
    ]

with DATA_PATH.open(encoding="utf-8") as f:
    raw_policies: List[Dict[str, Any]] = json.load(f)

cleaned_documents: List[Dict[str, Any]] = []
for p in raw_policies:
    p = p.copy()
    p["main_category_list"] = _split(p.get("lclsfNm", ""))
    p["sub_category_list"]  = _split(p.get("mclsfNm", ""))
    p["keyword_list"]       = _split(p.get("plcyKywdNm", ""))
    cleaned_documents.append(p)

# ────────────────────────────────────────────────
# 4) 1차 메타 필터
# ----------------------------------------------
def filter_policies_by_query(
    parsed: Dict[str, Any], docs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    reg, age_r = parsed["region"], parsed["age_range"]
    kw = set(parsed["matched_keywords"])
    cat = set(parsed["matched_main_categories"])

    hits = []
    for d in docs:
        # 키워드 / 대분류 매칭
        if kw or cat:
            if not (kw & set(d["keyword_list"]) or cat & set(d["main_category_list"])):
                continue

        # 지역 필터
        if reg:
            inst_region = extract_region_from_institution(
                d.get("rgtrInstCdNm") or d.get("sprvsnInstCdNm") or ""
            )
            if inst_region != reg:
                continue

        # 연령 필터
        if age_r:
            try:
                mn, mx = int(d.get("sprtTrgtMinAge") or 0), int(d.get("sprtTrgtMaxAge") or 99)
            except ValueError:
                mn, mx = 0, 99
            if age_r[1] < mn or age_r[0] > mx:
                continue

        hits.append(d)
    return hits

# ────────────────────────────────────────────────
# 5) Chroma 벡터스토어 로드
# ----------------------------------------------
embedder = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
)

VECTOR_DIR = PROJECT_ROOT / "chroma_db" / "Graph_all"
vectorstore = Chroma(persist_directory=str(VECTOR_DIR), embedding_function=embedder)

# ────────────────────────────────────────────────
# 6) 하이브리드 리트리버
# ----------------------------------------------
def hybrid_retrieve(query: str, k: int = 5) -> List[Document]:
    """
    1) 사용자 입력 파싱 → 메타 필터
    2) Chroma MMR 검색 (where 필터)
    3) 결과가 없으면 fallback 재검색
    """
    parsed = parse_user_query(query)

    # 메타 1차 필터
    allowed_ids = {
        d["plcyNo"] for d in filter_policies_by_query(parsed, cleaned_documents)
    }

    # where 필터 dict
    where_filter = {"doc_id": {"$in": list(allowed_ids)}} if allowed_ids else {}

    # MMR 리트리버
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 4, "where": where_filter},
    )
    docs = retriever.get_relevant_documents(query)

    # fallback (필터 결과 있는데 MMR -> 0개)
    if not docs and allowed_ids:
        raw = vectorstore.similarity_search(query, k=50)
        docs = [d for d in raw if d.metadata.get("doc_id") in allowed_ids][:k]

    return docs

# ────────────────────────────────────────────────
# 모듈 단독 실행 테스트
# ----------------------------------------------
if __name__ == "__main__":
    while True:
        q = input("질문(q to exit) > ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break
        for doc in hybrid_retrieve(q, k=3):
            print("—", doc.metadata.get("policy_name"), "/", doc.metadata.get("region"))
        print("-" * 50)
