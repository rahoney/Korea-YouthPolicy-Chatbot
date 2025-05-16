from __future__ import annotations
from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain

from chatbot.retrieval import hybrid_retrieve   # 리트리버 함수

# ────────────────────────────────────────────────
# 1) LLM 초기화 (Gemini-2.0-flash)
# ----------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    top_p=0.9
)

# ────────────────────────────────────────────────
# 2) 프롬프트 템플릿
# ----------------------------------------------
# 시스템 지시사항
system_instruction = """
- 질문을 단계별로 분석해서 논리적으로 답하세요.
- 질문속 모든 조건에 부합하는 정책이 없다면 "해당하는 정책이 없습니다."라고 간단하게 답해요.
- 최소나이와 최대나이 조건을 확인해 질문의 나이가 그 사이에 해당하는지 확인하세요.
- 데이터의 최대나이가 공란 또는 0이라면 해당 항목의 나이 제한이 없는 거에요.
- 질문에서 지역, 나이, 정책 등 조건에 모두 부합하는 것을 데이터에서 찾으세요.
- 가상의 정책을 만들지 마세요. 실제 정책 데이터만 사용하세요.
- 질문에 끼워맞춰 기존 정책을 가공하지마세요.
- 최종 수정일이 가장 최근에 된 것으로 답변하세요.
- URL이 신청url, 침고url 모두 없다면 신청 방법이나 다른 항목 데이터에 있는지 찾아주세요. 없다면 운영기관 공식 홈페이지 url을 넣어주세요.
- 답변할 항목의 데이터가 없다면 "등록기관에서 데이터를 미등록하였습니다."라고 답해주세요.(url항목 제외)
- 만약 문의할 연락처가 데이터 어디에도 없다면 운영기관의 공식 연락처를 인터넷에서 찾아서 넣어주세요.
- 부동산 질문이 있다면 주거에 대한 질문이에요.
"""
# 시스템 메시지 프롬프트
system_prompt = SystemMessagePromptTemplate.from_template(system_instruction)

# 사용자 입력 프롬프트
user_prompt = HumanMessagePromptTemplate.from_template(
    """
    사용자 질문: {question}\n\n
    아래는 질문과 관련하여 검색된 정책 내용입니다.
    질문에서 제시한 지역 조건에 정확히 맞는 정책을 가장 먼저 우선적으로 답변에 작성해주세요:\n
    이 정책들은 앞선 단계에서 LLM에 의해 정확한 평가 기준(지역, 나이, 정책 내용)에 따라 점수가 매겨졌습니다.
    점수가 가장 높은 정책부터 가장 낮은 정책 순으로 나열되어 있습니다.

    {context}
    위 정책 중 점수가 가장 높은 첫 번째 정책부터 우선적으로 아래 형식에 맞춰 답변을 구성해주세요.
    점수가 높아도 질문속 모든 조건에 부합하지 않으면 "해당하는 정책이 없습니다."라고 간단하게 답해요.

    📌 정책명: [정책 이름]\n
    📍 정책 설명: [정책에 대한 상세한 설명]\n\n
    📌 지원 내용:\n
    ✔️ [지원 항목 1]\n
    ✔️ [지원 항목 2]\n
    ✔️ [지원 항목 3]\n\n
    📌 신청 조건:\n
    📍 [신청 조건 1]\n
    📍 [신청 조건 2]\n
    📍 연령 제한:\n\n
    📌 사업 기간:\n
    📌 신청 방법:\n
    📌 기타 사항:\n
    📌 제출 서류:\n\n
    📌 주관 기관:\n
    📌 운영 기관:\n
    📌 문의 & 추가 정보:\n
    📞 관련 기관 문의: [문의처]\n
    🔗 참고 URL: [없다면 '-' 표기]\n
    ----------------------------------------------------------------\n
    """
    )

# ChatPromptTemplate 생성
prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
qa_chain = LLMChain(llm=llm, prompt=prompt)

# ────────────────────────────────────────────────
# 3) 체인 빌더
# ----------------------------------------------
def build_chain(k: int = 5) -> LLMChain:
    """
    k: hybrid_retrieve top-k
    """
    def _make_inputs(question: str) -> Dict[str, str]:
        docs = hybrid_retrieve(question, k=k)
        context = "\n\n".join(
            f"[{d.metadata.get('doc_id')}] {d.page_content[:800]}"
            for d in docs
        ) or "NONE"
        return {"question": question, "context": context}

    class _Wrapper(LLMChain):
        def run(self, inputs: Dict[str, str] | str) -> str:
            if isinstance(inputs, str):
                inputs = {"question": inputs}
            full_inputs = _make_inputs(inputs["question"])
            return super().run(full_inputs)

    return _Wrapper(llm=llm, prompt=prompt)
# ────────────────────────────────────────────────
# 4) 사용 예시 (단독 실행)
# ----------------------------------------------
if __name__ == "__main__":
    qa = build_chain(k=5)
    while True:
        q = input("질문(q to quit) > ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break
        print(qa.run({"question": q}))
        print("-" * 60)
