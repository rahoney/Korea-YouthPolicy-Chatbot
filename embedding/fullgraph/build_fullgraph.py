import networkx as nx
import torch
import json
import os
from tqdm import tqdm
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from embedding.json_to_docs import load_json, convert_json_to_documents
from collections import defaultdict
from pathlib import Path
import itertools
      
"""지역 노드 생성을 위한 사전처리"""
def extract_region_from_institution(name: str) -> str:
    REGION_KEYWORDS = {
    "서울": "서울",
    "경기": "경기",
    "충북": "충북", "충청북도": "충북",
    "충남": "충남", "충청남도": "충남",
    "전북": "전북", "전라북도": "전북",
    "전남": "전남", "전라남도": "전남",
    "경북": "경북", "경상북도": "경북",
    "경남": "경남", "경상남도": "경남",
    "강원": "강원",
    "제주": "제주",
    "부산": "부산",
    "대구": "대구",
    "광주": "광주",
    "인천": "인천",
    "대전": "대전",
    "울산": "울산",
    "세종": "세종"
    }
    
    if not name:
        return "전국"
    for keyword, std_region in REGION_KEYWORDS.items(): 
        if keyword in name:
            return std_region
    return "전국"  # 지역명이 없다면 전국으로 처리  

"""정책 데이터를 기반으로 네트워크 그래프를 생성"""
def build_policy_graph(documents):
    All_G = nx.Graph()

    # 1) 문서마다 노드 등록
    for i, doc in enumerate(tqdm(documents, desc="노드 등록")):
        node_id = f"Policy_{i}"

        # 키워드를 미리 set으로 처리하여 중복 제거
        keywords_set = set([kw.strip() for kw in doc.metadata.get("policy_keywords", "").split(",") if kw.strip()])
        keywords_str = ", ".join(sorted(keywords_set))
        # 지역 추출 - 등록자기관명에서 지역 추출
        region_name = extract_region_from_institution(doc.metadata.get("registering_institution", ""))
        actual_doc_id = doc.metadata.get("doc_id", "")
        policy_number = doc.metadata.get("policy_number", "")

        All_G.add_node(
            node_id,
            doc_id=doc.metadata.get("doc_id", ""),
            policy_number=doc.metadata.get("policy_number", ""),
            policy_name=doc.metadata.get("policy_name", ""), #정책명
            min_age=doc.metadata.get("min_age", ""), #최소나이
            max_age=doc.metadata.get("max_age", ""), #최대나이
            registering_institution=doc.metadata.get("registering_institution", ""),
            supervising_institution=doc.metadata.get("supervising_institution", ""), #주관기관
            operating_institution=doc.metadata.get("operating_institution", ""), #운영기관
            policy_category=doc.metadata.get("policy_category", ""),  # 정책 대분류
            policy_subcategory=doc.metadata.get("policy_subcategory", ""),  # 정책 중분류
            start_date=doc.metadata.get("start_date", ""), #사업기간 시작일
            end_date=doc.metadata.get("end_date", ""), #사업기간 종료일
            policy_keywords=keywords_str,  # Set 형태로 저장
            region=region_name, #추출한 지역
            page_content=doc.page_content  # 정책 설명, 지원 내용 등이 담긴 전체 텍스트
        )

    # 2) 문서들 간의 관계를 찾고, 해당 관계가 확인되면 엣지를 추가
    # 등록기관, 주관기관, 정책 분류 등이 같으면 엣지를 추가
    buckets = defaultdict(list)
    for idx, doc in enumerate(documents):
        reg = extract_region_from_institution(
            doc.metadata.get("registering_institution", "")
        )
        cat = doc.metadata.get("policy_category", "")
        buckets[(reg, cat)].append(idx)
    
    for bucket in tqdm(buckets.values(), desc="엣지 연결"):
        for i, j in itertools.combinations(bucket, 2):
            node_i, node_j = f"Policy_{i}", f"Policy_{j}"
            relation_list  = []
    
            # 중분류 동일
            if All_G.nodes[node_i]["policy_subcategory"] == \
               All_G.nodes[node_j]["policy_subcategory"]:
               relation_list.append("same_policy_subcategory")
    
            # 키워드 교집합
            kws_i = set(All_G.nodes[node_i]["policy_keywords"].split(", "))
            kws_j = set(All_G.nodes[node_j]["policy_keywords"].split(", "))
            common = kws_i & kws_j
            if common:
                relation_list.append(f"shared_keywords ({len(common)})")
    
            if relation_list:
                All_G.add_edge(
                    node_i, node_j,
                    relation=", ".join(relation_list),
                    weight=max(1, len(common))
                )

    return All_G

"""그래프의 노드를 LangChain Document로 변환"""
def convert_graph_to_documents(graph: nx.Graph) -> list[Document]:
    graph_documents = []
    
    for node_id, attrs in graph.nodes(data=True):
        # page_content는 정책 설명 중심 텍스트
        content = attrs.get("page_content", "")
    
        # 필수 메타데이터 구성 (안정성 확보 위해 get 사용)
        # plcyNo로 실제 정책번호를 꺼내서 doc_id로 쓰기
        doc_id = attrs.get("doc_id", "")
        if not doc_id:
            continue  # plcyNo가 없는 노드는 건너뛰기
    
        metadata = {
            "doc_id": doc_id,    # Policy_0, Policy_1 등 고유 ID
            "policy_number": attrs.get("policy_number", ""),
            "policy_name": attrs.get("policy_name", ""),
            "region": attrs.get("region", ""),
            "policy_category": attrs.get("policy_category", ""),
            "policy_subcategory": attrs.get("policy_subcategory", ""),
            "min_age": attrs.get("min_age", ""),
            "max_age": attrs.get("max_age", ""),
            "start_date": attrs.get("start_date", ""),
            "end_date": attrs.get("end_date", ""),
            "registering_institution": attrs.get("registering_institution", ""),
            "supervising_institution": attrs.get("supervising_institution", ""),
            "operating_institution": attrs.get("operating_institution", ""),
            "policy_keywords": attrs.get("policy_keywords", "")
        }
    
        doc = Document(page_content=content, metadata=metadata)
        graph_documents.append(doc)
    
    print(f"총 {len(graph_documents)}개의 Document 객체로 변환 완료")
    return graph_documents 

"""청크 분할"""
def chunk_documents(docs: list[Document], embedder) -> list[Document]:
    # SemanticChunker 생성
    splitter = SemanticChunker(embedder)
    chunks = []
    for doc in tqdm(docs, desc="문서 청크 분할"):
        chunks.extend(splitter.split_documents([doc]))   
    print(f"SemanticChunker 완료 → {len(chunks)}개 청크")
    return chunks

"""청크 벡터화 후 ChromaDB에 저장"""
def embed_and_save_chunks(chunks: list[Document], embedder, project_root: str): 
    # Chroma DB 저장
    persist_directory = os.path.join(project_root, "chroma_db", "Graph_all")
    os.makedirs(persist_directory, exist_ok=True)
    
    Gdb_full = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=persist_directory
    )
    
    Gdb_full.persist()
    
    print("Chroma DB에 그래프 기반 벡터 저장 완료!")  
    return persist_directory


def main():
    start = time.time()
      
    #디렉토리 설정
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
      
    #임베딩 모델 설정
    embedder = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device":"cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"batch_size":32, "normalize_embeddings":True}
    )

    # JSON 로드 -> Doc 
    json_path    = os.path.join(project_root, "data", "YouthPolicy_data.json")
    data = load_json(json_path)
    documents = convert_json_to_documents(data)

    # Doc -> 그래프 빌드
    graph = build_policy_graph(documents)
    print("그래프 노드 수:", graph.number_of_nodes())
    print("그래프 엣지 수:", graph.number_of_edges())

    # 그래프 저장
    save_path = os.path.join(project_root, "graph")
    os.makedirs(save_path, exist_ok=True) #폴더 없으면 생성
    gml_path = os.path.join(save_path, "policy_graph_all.gml")
    nx.write_gml(graph, gml_path)
    print("GML 그래프 저장 완료:", gml_path)

    # 그래프 -> Doc -> Chunk 분할 -> 벡터DB 임베딩
    node_docs = convert_graph_to_documents(graph)
    chunks    = chunk_documents(node_docs, embedder)
    embed_and_save_chunks(chunks, embedder, project_root) 
    print(f"\n 전체 수행 시간: {time.time() - start:,.1f}s")
      
if __name__ == "__main__":
    main()
