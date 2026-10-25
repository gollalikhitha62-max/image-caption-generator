import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Page configuration
st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="wide"
)

# Website heading
st.title("🖼️ AI Image Caption Generator")
st.markdown(
    "Upload an image and generate an AI-powered caption using Florence-2."
)

st.divider()

# Load the pretrained model only once
@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "microsoft/Florence-2-base"

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model.eval()
    return processor, model


# Image upload
uploaded_file = st.file_uploader(
    "Upload your image",
    type=["jpg", "jpeg", "png", "webp"]
)

# Caption options
caption_option = st.selectbox(
    "Select caption type",
    ["Short Caption", "Detailed Caption"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

        if st.button("✨ Generate Caption", type="primary"):
            with st.spinner(
                "Loading Florence-2 and generating your caption..."
            ):
                processor, model = load_model()

                task = (
                    "<CAPTION>"
                    if caption_option == "Short Caption"
                    else "<DETAILED_CAPTION>"
                )

                inputs = processor(
                    text=task,
                    images=image,
                    return_tensors="pt"
                )

                with torch.inference_mode():
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=128,
                        num_beams=3
                    )

                generated_text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=False
                )[0]

                result = processor.post_process_generation(
                    generated_text,
                    task=task,
                    image_size=(image.width, image.height)
                )

                caption = result.get(task, generated_text)

                st.subheader("📝 Generated Caption")
                st.success(str(caption))

                st.download_button(
                    "Download Caption",
                    data=str(caption),
                    file_name="image_caption.txt",
                    mime="text/plain"
                )

    except Exception as error:
        st.error("Unable to generate the caption.")
        st.warning(
            "Check the Render logs for details. "
            "The free service may not have enough memory "
            "to load Florence-2."
        )
        st.caption(f"Error details: {error}")

else:
    st.info("Please upload an image to get started.")

st.divider()
st.caption("Powered by Microsoft Florence-2 and Streamlit")
