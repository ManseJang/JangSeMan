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

def generate_chatgpt_response(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": '''너는 초등학교 국어선생님이다. 학생들이 논설문을 잘 학습할 수 있도록 도움을 주어야 한다.
                    항상 경어를 사용해야해. 학생들이 아래 입력한 논설문에 대해 아래 점검 내용을 참고하여 피드백과 100점 만점에 몇 점인지 점수를 제공해라.
                    피드백은 한국말로, 초등학생 수준의 단어와 문장을 사용해서 한국어로 작성해야 한다.
                    점검 내용의 각 부분의 점수를 20점 만점으로 산출하여 총점 100점에 몇 점인지 점수를 제공해라..
                    학생이 발전할 수 있도록 조언을 3줄 정도 마지막에 제공해라.
                    피드백에 학생들이 이해할 수 있도록 예시를 충분히 제공해라.
                    위의 조언들을 참고하여 학생의 논설문을 수정해서 학생에게 제공해라.
                    점검내용:
                    1. 주장이 가치있고 중요한가
                    2. 근거가 주장과 관련이 있는가
                    3. 근거가 주장을 뒷받침하는가
                    4. 표현이 적절한가(주관적인 표현, 모하한 표현, 단정적인 표현을 쓰지 않아야 한다.)
                    5. 문장이 자연스럽게 이어지고, 맞춤법이 틀린 부분이 없는가.'''},
            {"role": "user", "content": text}
        ]
    )
    feedback = response.choices[0].message.content
    return feedback

st.title("논설문 피드백 챗봇")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.subheader("업로드된 이미지")
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    with st.spinner("글자를 읽는 중입니다..."):
        ocr_result = OCR_parser(image)

    if ocr_result:
        extracted_text = " ".join([field['inferText'] for field in ocr_result])
        st.subheader("인식한 글자")
        st.write(extracted_text)
    else:
        extracted_text = ""

    if extracted_text:
        with st.spinner("피드백을 생성 중입니다..."):
            chatgpt_response = generate_chatgpt_response(extracted_text)
        st.subheader("챗봇의 피드백")
        st.write(chatgpt_response)
