import json
from langchain.schema import Document

def load_json(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def convert_json_to_documents(data: list) -> list[Document]:
    documents = []

    for idx, item in enumerate(data):
        doc_id = item.get("plcyNo", f"policy_{idx}")

        content = f"""
        [문서 ID: {doc_id}]
        정책명: {item.get("plcyNm", "정보 없음")}
        정책번호: {item.get("plcyNo", "정보 없음")}
        정책 설명: {item.get("plcyExplnCn", "정보 없음")}
        지원 내용: {item.get("plcySprtCn", "정보 없음")}
        참여제안 대상 내용: {item.get("ptcpPrpTrgtCn", "정보 없음")}
        신청 방법: {item.get("plcyAplyMthdCn", "정보 없음")}
        제출 서류: {item.get("sbmsnDcmntCn", "정보 없음")}
        기타 사항: {item.get("etcMttrCn", "정보 없음")}
        추가 신청 자격: {item.get("addAplyQlfcCndCn", "정보 없음")}
        사업기간 기타내용: {item.get("bizPrdEtcCn", "정보 없음")}
        심사 방법: {item.get("srngMthdCn", "정보 없음")}
        지원대상 최소나이: {item.get("sprtTrgtMinAge", "정보 없음")}
        지원대상 최대나이: {item.get("sprtTrgtMaxAge", "정보 없음")}
        사업기간 시작일: {item.get("bizPrdBgngYmd", "정보 없음")}
        사업기간 종료일: {item.get("bizPrdEndYmd", "정보 없음")}
        최초 등록일: {item.get("frstRegDt", "정보 없음")}
        최종 수정일: {item.get("lastMdfcnDt", "정보 없음")}
        신청기간: {item.get("aplyYmd", "정보 없음")}
        등록기관: {item.get("rgtrInstCdNm", "정보 없음")}
        주관기관: {item.get("sprvsnInstCdNm", "정보 없음")}
        운영기관: {item.get("operInstCdNm", "정보 없음")}
        신청URL: {item.get("aplyUrlAddr", "정보 없음")}
        참고URL1: {item.get("refUrlAddr1", "정보 없음")}
        참고URL2: {item.get("refUrlAddr2", "정보 없음")}
        정책 대분류: {item.get("lclsfNm", "정보 없음")}
        정책 중분류: {item.get("mclsfNm", "정보 없음")}
        정책 키워드: {item.get("plcyKywdNm", "정보 없음")}
        """

        metadata = {
            "doc_id": doc_id,
            "policy_name": item.get("plcyNm", ""),
            "policy_number": item.get("plcyNo", ""),
            "min_age": item.get("sprtTrgtMinAge", ""),
            "max_age": item.get("sprtTrgtMaxAge", ""),
            "start_date": item.get("bizPrdBgngYmd", ""),
            "end_date": item.get("bizPrdEndYmd", ""),
            "first_registration": item.get("frstRegDt", ""),
            "last_modification": item.get("lastMdfcnDt", ""),
            "apply_period": item.get("aplyYmd", ""),
            "registering_institution": item.get("rgtrInstCdNm", ""),
            "supervising_institution": item.get("sprvsnInstCdNm", ""),
            "operating_institution": item.get("operInstCdNm", ""),
            "apply_url": item.get("aplyUrlAddr", ""),
            "ref_url_1": item.get("refUrlAddr1", ""),
            "ref_url_2": item.get("refUrlAddr2", ""),
            "policy_category": item.get("lclsfNm", ""),
            "policy_subcategory": item.get("mclsfNm", ""),
            "policy_keywords": item.get("plcyKywdNm", "")
        }

        documents.append(Document(page_content=content, metadata=metadata))
    return documents
