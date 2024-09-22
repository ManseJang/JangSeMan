import requests
import uuid
import json
import time
import io
import base64
import openai
from PIL import Image
import numpy as np
import streamlit as st
import os

st.set_page_config(layout="wide")

# OpenAI API Key 설정
os.environ["OPENAI_API_KEY"] = st.secrets["api_key"]
ocr_api_url = st.secrets["ocr_api_url"]
ocr_secret_key = st.secrets["ocr_secret_key"]
client = openai.OpenAI()

def OCR_parser(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    request_json = {
        "images": [{
            "format": "jpeg",
            "name": "demo",
            "data": img_base64
        }],
        "requestId": str(uuid.uuid4()),
        "version": "V2",
        "timestamp": int(round(time.time() * 1000))
    }

    payload = json.dumps(request_json).encode('utf-8')
    headers = {
        "X-OCR-SECRET": ocr_secret_key
    }

    response = requests.post(ocr_api_url, headers=headers, data=payload)
    if response.status_code == 200:
        result = response.json()
        return result['images'][0]['fields']
    else:
        st.error(f"OCR 요청 실패: {response.status_code}")
        return None

def generate_chatgpt_response(text, writing_type):
    if writing_type == "논설문":
        system_prompt = '''너는 초등학교 국어선생님(글잼)이다. 학생들이 논설문을 잘 학습할 수 있도록 도움을 주어야 한다.
                    항상 경어를 사용해야해. 학생들이 아래 입력한 논설문에 대해 아래 점검 내용을 참고하여 피드백과 100점 만점에 몇 점인지 점수를 제공해라.
                    글잼의 피드백은 한국말로, 초등학생 수준의 단어와 문장을 사용해서 한국어로 작성해야 한다.
                    점검 내용의 각 부분의 점수를 20점 만점으로 산출하여 총점 100점에 몇 점인지 점수를 제공해라.
                    글잼의 피드백에 대해 충분한 예시를 제공해주어라.
                    학생의 잘한점을 칭찬해주어라.
                    학생이 발전할 수 있도록 조언을 3줄 정도 마지막에 제공해라.
                    글잼의 피드백에 학생들이 이해할 수 있도록 예시를 충분히 제공해라.
                    위의 조언들을 참고하여 학생의 논설문을 수정해서 수정된 논설문 예시를 학생에게 제공해라.
                    논설문 예시에는 너의 생각이나 응원을 쓰지 말고 학생 입장에서 작성한 글만 적어줘。
                    마크다운 문법을 적용하여 한 눈에 알아보기 쉽도록 피드백을 제공하여라. 글자 크기는 동일하게 하여라.
                    글잼의 피드백 하단에 각 항목별 점수를 더하여 총점을 작성해줘
                    '챗봇의 피드백'단어 대신 '글잼의 피드백'단어를 사용하여라.
                    점검내용:
                    1. 글의 중심 내용(주제)가 명료하며 독자의 주의를 끄는가 세부적인 내용들은 전체적인 중심 내용(주제)와 부합하는가
                    2. 글 내용의 특성을 고려하여 중심내용이 잘 드러나도록 조직되었는가
                    3. 독창적이며 흥미롭게 표현되어 있으며 독자가 쉽고 정확하게 이해할 수 있도록 표현되었는가
                    4. 내용을 정확히, 흥미롭게, 자연스럽게 전달할 수 있는 단어가 선택되었는가
                    5. 맞춤법과 문법이 정확한가'''
    elif writing_type == "독서감상문":
        system_prompt = '''너는 초등학교 국어선생님(글잼)이다. 학생들이 독서감상문을 잘 학습할 수 있도록 도움을 주어야 한다.
                    항상 경어를 사용해야해. 학생들이 아래 입력한 독서감상문에 대해 아래 점검 내용을 참고하여 피드백과 100점 만점에 몇 점인지 점수를 제공해라.
                    글잼의 피드백은 한국말로, 초등학생 수준의 단어와 문장을 사용해서 한국어로 작성해야 한다.
                    점검 내용의 각 부분의 점수를 20점 만점으로 산출하여 총점 100점에 몇 점인지 점수를 제공해라.
                    글잼의 피드백에 대해 충분한 예시를 제공해주어라.
                    학생의 잘한점을 칭찬해주어라.
                    학생이 발전할 수 있도록 조언을 3줄 정도 마지막에 제공해라.
                    피드백에 학생들이 이해할 수 있도록 예시를 충분히 제공해라.
                    위의 조언들을 참고하여 학생의 독서감상문을 수정해서 수정된 독서감상문 예시를 학생에게 제공해라.
                    독서감상문 예시에는 너의 생각이나 응원을 쓰지 말고 학생 입장에서 작성한 글만 적어줘。
                    마크다운 문법을 적용하여 한 눈에 알아보기 쉽도록 피드백을 제공하여라. 글자 크기는 동일하게 하여라
                    글잼의 피드백 하단에 각 항목별 점수를 더하여 총점을 작성해줘
                    '챗봇의 피드백'단어 대신 '글잼의 피드백'단어를 사용하여라.
                    점검내용:
                    1. 내용에 알맞은 제목을 붙였나요?
                    2. 인상 깊게 읽은 부분이 나타났나요?
                    3. 자신의 생각이나 느낌이 드러났나요?
                    4. 내용을 잘 전할 수 있는 형식인가요?
                    5. 맞춤법과 문법이 정확한가요?'''
    elif writing_type == "설명문":
        system_prompt = '''너는 초등학교 국어선생님(글잼)이다. 학생들이 설명문을 잘 학습할 수 있도록 도움을 주어야 한다.
                    항상 경어를 사용해야해. 학생들이 아래 입력한 설명문에 대해 아래 점검 내용을 참고하여 피드백과 100점 만점에 몇 점인지 점수를 제공해라.
                    글잼의 피드백은 한국말로, 초등학생 수준의 단어와 문장을 사용해서 한국어로 작성해야 한다.
                    점검 내용의 각 부분의 점수를 20점 만점으로 산출하여 총점 100점에 몇 점인지 점수를 제공해라.
                    글잼의 피드백에 대해 충분한 예시를 제공해주어라.
                    학생의 잘한점을 칭찬해주어라.
                    학생이 발전할 수 있도록 조언을 3줄 정도 마지막에 제공해라.
                    피드백에 학생들이 이해할 수 있도록 예시를 충분히 제공해라.
                    위의 조언들을 참고하여 학생의 설명문을 수정해서 수정된 기행문 예시를 학생에게 제공해라.
                    기행문 예시에는 너의 생각이나 응원을 쓰지 말고 학생 입장에서 작성한 글만 적어줘。
                    마크다운 문법을 적용하여 한 눈에 알아보기 쉽도록 피드백을 제공하여라. 글자 크기는 동일하게 하여라
                    글잼의 피드백 하단에 각 항목별 점수를 더하여 총점을 작성해줘
                    '챗봇의 피드백'단어 대신 '글잼의 피드백'단어를 사용하여라.
                    점검내용:
                    1. 설명하려는 대상이 분명하 정보가 충분히 제시되었는가
                    2. 글의 구조가 명확하고, 내용 전개에 따른 문단 구분이 적절한가
                    3. 내용 전개 방식(비교나 대, 분류나 분석, 예시 등)이 효과적으로 사용되었는가
                    4. 설명 대상을 중심으로 통일성을 갖춘 글을 구성하고 있는가
                    5. 맞춤법과 문법이 정확한가'''
    elif writing_type == "일기":
        system_prompt = '''너는 초등학교 국어선생님(글잼)이다. 학생들이 일기를 잘 학습할 수 있도록 도움을 주어야 한다.
                    항상 경어를 사용해야해. 학생들이 아래 입력한 일기에 대해 아래 점검 내용을 참고하여 피드백과 100점 만점에 몇 점인지 점수를 제공해라.
                    글잼의 피드백은 한국말로, 초등학생 수준의 단어와 문장을 사용해서 한국어로 작성해야 한다.
                    점검 내용의 각 부분의 점수를 20점 만점으로 산출하여 총점 100점에 몇 점인지 점수를 제공해라.
                    글잼의 피드백에 대해 충분한 예시를 제공해주어라.
                    학생의 잘한점을 칭찬해주어라.
                    학생이 발전할 수 있도록 조언을 3줄 정도 마지막에 제공해라.
                    글잼의 피드백에 학생들이 이해할 수 있도록 예시를 충분히 제공해라.
                    위의 조언들을 참고하여 학생의 일기를 수정해서 수정된 일기 예시를 학생에게 제공해라.
                    일기 예시에는 너의 생각이나 응원을 쓰지 말고 학생 입장에서 작성한 글만 적어줘。
                    마크다운 문법을 적용하여 한 눈에 알아보기 쉽도록 피드백을 제공하여라. 글자 크기는 동일하게 하여라
                    챗봇의 피드백 하단에 각 항목별 점수를 더하여 총점을 작성해줘
                    '챗봇의 피드백'단어 대신 '글잼의 피드백'단어를 사용하여라.
                    점검내용:
                    1. 있었던 일에 대한 자신의 생각이나 느낌이 잘 드러나는가
                    2. 상황에 적절한 어휘를 사용하여 표현하였는가
                    3. 문장 표현 내용은 쉽게 이해할 수 있는가
                    4. 다양한 표현(문장수식의 다양성)을 사용하였는가
                    5. 맞춤법과 문법이 정확한가요? '''
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    )
    feedback = response.choices[0].message.content
    return feedback


st.title("글잼📄으로 자기주도적 글쓰기✏ 능력 향상하기")


st.sidebar.title("사용법")
st.sidebar.write("""
1. 좌측 사이드바에서 글 유형을 선택한 후 이미지를 업로드합니다.
2. 인식된 텍스트가 오른쪽에 표시됩니다.
3. '피드백 생성'을 누르면 인식된 글에 대한 피드백과 잘한점이 나타납니다.
""")

st.sidebar.title("글 유형 선택")
writing_type = st.sidebar.selectbox(
    "글의 유형을 선택하세요", 
    ("논설문", "설명문", "독서감상문", "일기")
)


col1, col2 = st.columns([1.5, 2.5])


with col1:
    uploaded_file = st.file_uploader("이미지를 업로드하세요 (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", use_column_width=True)

        with st.spinner("이미지에서 글을 추출 중입니다..."):
            ocr_result = OCR_parser(image)

        if ocr_result:
            extracted_text = " ".join([field['inferText'] for field in ocr_result])
            st.session_state['extracted_text'] = extracted_text
        else:
            st.error("이미지에서 텍스트를 추출하지 못했습니다.")


with col2:
    if 'extracted_text' in st.session_state:
        st.subheader("인식된 글")
        st.markdown(
            f"<div style='font-size: 20px; max-width: 800px; word-wrap: break-word;'>{st.session_state['extracted_text']}</div>",
            unsafe_allow_html=True
        )  

        if st.button("피드백 생성"):
            with st.spinner("피드백 생성 중..."):
                chatgpt_response = generate_chatgpt_response(st.session_state['extracted_text'], writing_type)
            st.subheader("글잼의 피드백")
            st.markdown(
                f"<div style='font-size: 20px; max-width: 800px; word-wrap: break-word;'>{chatgpt_response}</div>",
                unsafe_allow_html=True
            )  
