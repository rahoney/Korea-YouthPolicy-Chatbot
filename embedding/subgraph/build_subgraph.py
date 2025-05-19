import os, json, re, unicodedata, itertools, time
from collections import Counter
import networkx as nx
import torch
from tqdm import tqdm
from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from embedding.json_to_docs import load_json

synonym_dict = {
    "주거": ["부동산", "주택", "거주", "임대", "전세", "전월세", "안심주택", "쉐어하우스"],
    "취업": ["일자리", "채용", "구직", "고용", "재직자", "현장면접", "재취업", "구인구직"],
    "창업": ["스타트업", "벤처", "자영업", "창직", "사업", "브랜딩", "푸드트럭"],
    "복지문화": ["복지", "문화", "생활지원", "심리지원", "문화바우처", "청년복지"],
    "보조금": ["지원금", "활동비", "인건비", "참여수당", "창업비", "대출", "생활비", "교통비"],
    "상담": ["맞춤형상담서비스", "심리상담", "멘토링", "코칭", "집단상담", "자기개발"],
    "공공임대주택": ["LH", "SH", "임대주택", "역세권청년주택", "공공분양", "안심주택"],
    "중소기업": ["청년기업", "소기업", "중견기업", "강소기업", "지역기업"],
    "장기미취업청년": ["장기실업", "경단녀", "니트", "사회진입지원"],
    "교육": ["훈련", "역량개발", "강의", "이러닝", "자격증", "코딩교육"],
    "출산": ["임신", "출산", "출산지원", "육아휴직", "산후", "양육"],
    "결혼": ["혼인", "신혼", "예비부부", "결혼생활", "신혼부부", "혼례"],
    "육아": ["육아", "양육", "아이돌봄", "보육", "돌봄서비스", "어린이집", "놀이시설"],
    "바우처": ["지원카드", "포인트", "문화바우처", "교육바우처", "복지카드"]
}
user_input_to_main_keyword = { syn: rep
    for rep, syns in synonym_dict.items() for syn in syns }

REGION_KEYWORDS = {
    "서울": "서울", "서울특별시": "서울", "서울시": "서울",
    "경기": "경기", "경기도": "경기",
    "충북": "충북", "충청북도": "충북",
    "충남": "충남", "충청남도": "충남",
    "전북": "전북", "전라북도": "전북", "전북특별자치도": "전북",
    "전남": "전남", "전라남도": "전남",
    "경북": "경북", "경상북도": "경북",
    "경남": "경남", "경상남도": "경남",
    "강원": "강원", "강원도": "강원", "강원특별자치도": "강원",
    "제주": "제주", "제주도": "제주", "제주특별자치도": "제주",
    "부산": "부산", "부산광역시": "부산", "부산시": "부산",
    "대구": "대구", "대구광역시": "대구", "대구시": "대구",
    "광주": "광주", "광주광역시": "광주", "광주시": "광주",
    "인천": "인천", "인천광역시": "인천", "인천시": "인천",
    "대전": "대전", "대전광역시": "대전", "대전시": "대전",
    "울산": "울산", "울산광역시": "울산", "울산시": "울산",
    "세종": "세종", "세종특별자치시": "세종","세종시": "세종"
}

def extract_region_from_institution(name: str|None) -> str:
    if not name: return "전국"
    for k, std in REGION_KEYWORDS.items():
        if k in name: return std
    return "전국"

def sanitize(text: str) -> str:   # 파일명 안전화
    return re.sub(r"[^\w가-힣\\-]", "_",
                  unicodedata.normalize("NFC", text or "None"))
#전처리
def preprocess_policy_fields(item: dict) -> dict:
    def split(field, mapper=None):
        out = []
        for t in (field or "").split(","):
            t = t.strip()
            if not t: continue
            out.append(mapper.get(t, t) if mapper else t)
        return list(dict.fromkeys(out))   # 중복 제거

    cleaned = item.copy()
    cleaned["main_category_list"] = split(item.get("lclsfNm",""))
    cleaned["sub_category_list"]  = split(item.get("mclsfNm",""))
    cleaned["keyword_list"]       = split(item.get("plcyKywdNm",""),
                                          mapper=user_input_to_main_keyword)
    cleaned["region"] = (
        extract_region_from_institution(item.get("rgtrInstCdNm","")) or
        extract_region_from_institution(item.get("sprvsnInstCdNm","")) or
        extract_region_from_institution(item.get("operInstCdNm","")) or
        "전국"
    )
    cleaned["policy_category"]   = cleaned["main_category_list"][0] if cleaned["main_category_list"] else "기타"
    cleaned["policy_subcategory"]= cleaned["sub_category_list"][0]  if cleaned["sub_category_list"] else "기타"
    return cleaned

# 필터, 조건
def filter_policies_by_query(
    policies:list[dict], matched_main_keywords:dict|None=None,
    target_region:str|None=None, target_main_categories:list[str]|None=None,
    target_keywords:list[str]|None=None) -> list[dict]:

    matched=[]
    for p in policies:
        if target_region and p.get("region")!=target_region: continue
        if target_main_categories and not (set(p["main_category_list"]) & set(target_main_categories)): continue
        if target_keywords and not (set(p["keyword_list"]) & set(target_keywords)): continue
        matched.append(p)
    return matched

def generate_subgraph_conditions(cleaned_docs, top_n=5):
    region_c, main_c, kw_c = Counter(), Counter(), Counter()
    for d in cleaned_docs:
        if d["region"] not in ("", "전국"): region_c[d["region"]]+=1
        for mc in d["main_category_list"]: main_c[mc]+=1
        for kw in d["keyword_list"]:       kw_c[kw]+=1
    cond=[]
    from itertools import product
    for r, mc, kw in product(
        [r for r,_ in region_c.most_common(top_n)],
        [c for c,_ in main_c.most_common(top_n)],
        [k for k,_ in kw_c.most_common(top_n)]
    ):
        if filter_policies_by_query(cleaned_docs, {}, r,[mc],[kw]):
            cond.append({"region":r,"main_category":mc,"keyword":kw})
    return cond

# 서브그래프 생성
def create_subgraph_and_embed(condition, cleaned_docs, graph_base_dir, chroma_base_dir, embedding_model):
    # 정책 필터링
    filtered = filter_policies_by_query(
        cleaned_docs,
        target_region=condition["region"],
        target_main_categories=[condition["main_category"]],
        target_keywords=[condition["keyword"]]
    )
    filtered = [d for d in filtered if d["region"] != "전국"]
    if len(filtered) < 5:
        print(f"[SKIP] {condition}: 정책수 부족")
        return None

    # 그래프(GML) 생성
    G = nx.Graph()
    for d in filtered:
        nid = d["plcyNo"]
        G.add_node(nid,
            doc_id=nid,
            policy_number=nid,
            policy_name=d.get("plcyNm",""),
            region=d["region"],
            policy_category=d["policy_category"],
            policy_subcategory=d["policy_subcategory"],
            policy_keywords=",".join(d["keyword_list"]),
            page_content=d.get("plcyExplnCn","")
        )
    for i,j in itertools.combinations(filtered,2):
        shared=(
            bool(set(i["main_category_list"]) & set(j["main_category_list"])) +
            bool(set(i["sub_category_list"])  & set(j["sub_category_list"])) +
            bool(set(i["keyword_list"])       & set(j["keyword_list"]))
        )
        if shared:
            G.add_edge(i["plcyNo"], j["plcyNo"])

    if G.number_of_edges() < 5:
        print(f"[SKIP] {condition}: 엣지수 부족")
        return None

    fname = f"policy_graph_{sanitize(condition['region'])}_{sanitize(condition['main_category'])}_{sanitize(condition['keyword'])}"
    gml_path = os.path.join(graph_base_dir, fname + ".gml")
    chroma_dir = os.path.join(chroma_base_dir, fname)

    os.makedirs(os.path.dirname(gml_path), exist_ok=True)
    nx.write_gml(G, gml_path)
    chunk_and_embed(filtered, chroma_dir, embedding_model)
    print(f"[OK] {fname}: 그래프/임베딩 저장 완료! ({len(filtered)}건)")
    return fname

def chunk_and_embed(filtered_docs, vectorstore_dir, embedding_model):
    """
    - filtered_docs: 정책 dict 리스트 (서브그래프나 전체 필터 결과)
    - vectorstore_dir: 저장할 Chroma 폴더
    - embedding_model: 임베딩 모델 인스턴스
    """
    # LangChain Document 생성
    documents = []
    for d in filtered_docs:
        content = d.get("plcyExplnCn", "").strip()
        if not content:
            continue
        meta = {
            "doc_id": d.get("plcyNo", ""),
            "policy_name": d.get("plcyNm", ""),
            "region": d.get("region", ""),
            "policy_category": d.get("policy_category", ""),
            "policy_subcategory": d.get("policy_subcategory", ""),
            "policy_keywords": ",".join(d.get("keyword_list", []))
        }
        documents.append(Document(page_content=content, metadata=meta))
    if not documents:
        print(f" 임베딩할 문서 없음({vectorstore_dir})")
        return

    # 1차 분할
    headers = [("#", "정책대분류"), ("##", "정책중분류"), ("###", "정책명")]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    split_docs = []
    for doc in documents:
        result = header_splitter.split_text(doc.page_content)
        for chunk in result:
            if isinstance(chunk, Document):
                # Document 타입이면 메타 유지(혹시 없으면 추가)
                if not getattr(chunk, "metadata", None):
                    chunk.metadata = doc.metadata.copy()
                split_docs.append(chunk)
            else:
                # str 타입이면 meta 붙여서 Document 생성
                split_docs.append(Document(page_content=chunk, metadata=doc.metadata.copy()))
           
    # 2차 분할
    semantic_splitter = SemanticChunker(embedding_model)
    final_chunks = []
    for doc in split_docs:
        final_chunks.extend(semantic_splitter.split_documents([doc]))

    if not final_chunks:
        print(f"임베딩할 청크 없음({vectorstore_dir})")
        return

    # Chroma 임베딩 저장
    os.makedirs(vectorstore_dir, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=final_chunks,
        embedding=embedding_model,
        persist_directory=vectorstore_dir
    )
    vectorstore.persist()
    print(f" {vectorstore_dir}: 임베딩/저장 완료! (청크 {len(final_chunks)}개)")


# main
def main():
    t0 = time.time()
    script = os.path.abspath(__file__)
    root   = os.path.dirname(os.path.dirname(script))   # embedding/
    project_root = os.path.dirname(root)                # Korea-YouthPolicy-Chatbot/
    data_path    = os.path.join(project_root, "data", "YouthPolicy_data.json")
    save_graph_dir = os.path.join(project_root, "graph", "subgraphs")
    persist_dir    = os.path.join(project_root, "chroma_db", "Graph_sub_separate")

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"데이터 파일이 존재하지 않습니다: {data_path}")
    graph_base_dir = os.path.join(project_root, "graph", "subgraphs")
    chroma_base_dir = os.path.join(project_root, "chroma_db", "Graph_sub_separate")

    # 임베딩 모델 준비
    embedder = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device":"cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"batch_size":32,"normalize_embeddings":True}
    )

    # 데이터 로딩 및 전처리
    data = load_json(data_path)
    cleaned = [preprocess_policy_fields(p) for p in data]
    conditions = generate_subgraph_conditions(cleaned, top_n=5)  # 상위 빈도 기준

    # 실제 서브그래프/임베딩 반복
    for cond in tqdm(conditions, desc="서브그래프 생성+임베딩"):
        create_subgraph_and_embed(cond, cleaned, graph_base_dir, chroma_base_dir, embedder)
        
        
if __name__=="__main__":
    main()
