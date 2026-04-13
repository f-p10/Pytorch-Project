import streamlit as st
from PIL import Image
from model import load_model, classify_image
from ollama import chat
from ollama import ChatResponse

#ollama run gemma3 first

def ask_llm(prediction):
    prompt = f"Which of bin out of compost, recycling, and garbage should I dispose of {prediction} Keep it to 15 words."

    response : ChatResponse = chat(model ='gemma3', messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])

    return response['message']['content']


st.set_page_config(page_title="Garbage Classifier")

st.title("Garbage Classification App")
st.write("Upload an image to classify waste and find out how to dispose of it properly.")

@st.cache_resource
def get_model():
    return load_model()

model = get_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        prediction = classify_image(model, image)

        class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        item = class_names[prediction]

        st.subheader("Model Prediction")
        st.success(item.capitalize())

        if st.button("Ask an LLM"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_llm(item)
                    st.subheader("LLM Response")
                    st.info(answer)
                except Exception as e:
                    st.error("Error, check console")
                    print("Error: " ,e)