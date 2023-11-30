import os
import streamlit as st
from PIL import Image
from utils import MODEL_PATH, MODEL_ARGS
from main import LLaVALLM
from watsonx import LangChainInterface
from langchain import PromptTemplate
from langchain.chains import LLMChain
from prompt import WATSON_PROMPT
from utils import CREDS,PROJECT_ID,IBM_MODEL_PARAMS,IBM_MODEL_ID

llm_model = LangChainInterface(model=IBM_MODEL_ID, credentials=CREDS, params=IBM_MODEL_PARAMS, project_id=PROJECT_ID)

# Create a Streamlit app
st.title("ferrovie")

if 'lava_obj' not in st.session_state:
    st.session_state.lava_obj = None

# Initialize LLaVALLM model
if st.session_state.lava_obj is None:
    lava_obj = LLaVALLM(model_path=MODEL_PATH, kwargs=MODEL_ARGS)
    st.session_state.lava_obj = lava_obj

image_object = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
st.session_state.image_object = image_object

# Check if image and question are provided
if image_object is not None :
    # Display the uploaded image
    st.image(image_object, caption="Uploaded Image", use_column_width=True)

    # Perform image captioning
    image_pil = Image.open(image_object)
    st.session_state.lava_obj.set_image(image_file=image_pil)

    caption = st.session_state.lava_obj.caption_image(prompt="Describe the image in terms of safety.")
    st.session_state.caption = caption

    # Display the Inference
    st.markdown("### Inference")
    st.write(st.session_state.caption)

    # Define the prompt templates
    prompt = PromptTemplate(
        input_variables=["output"],
        template= WATSON_PROMPT,
    )
    st.session_state.prompt = prompt

    # Chaining 
    chain = LLMChain(llm=llm_model, prompt=st.session_state.prompt)
    st.session_state.chain = chain

    generated_response = st.session_state.chain.run(st.session_state.caption)
    st.session_state.generated_response=generated_response

    # Safety : Hard hat and Safety Vest,
    # Infrastructure : Train Tracks,
    # People : One,
    # The worker/s are in compliance

    # split the generated response by comma and display line by line
    st.session_state.generated_response = st.session_state.generated_response.split(",")
    # Display the Inference
    st.markdown("### Conclusion")
    for line in st.session_state.generated_response:
        st.write(line)

    st.session_state.line = line

    # Display the Inference
    # st.markdown("### Conclusion")
    # st.write(st.session_state.generated_response)


else:
    st.warning("Please upload an image and provide a question.")

