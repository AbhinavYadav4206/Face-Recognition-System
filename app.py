import streamlit as st
from register import register_face
from search import search_face

# Page Configuration
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🧑",
    layout="centered"
)


# Helper Functions
def get_image(prefix: str):

    camera_image = st.camera_input(
        "Take a photo",
        key=f"{prefix}_camera"
    )

    uploaded_image = st.file_uploader(
        "Or upload an image",
        type=["jpg", "jpeg", "png"],
        key=f"{prefix}_upload"
    )

    if camera_image is not None:
        return camera_image

    if uploaded_image is not None:
        return uploaded_image

    return None


def show_image(image, caption: str):

    if image is not None:
        st.image(image, caption=caption, width=300)


def show_match_result(result: dict):

    if result["match"]:

        person = result["person"]

        st.success("✅ Match Found")

        st.header(person["name"])

        st.write(f"Person Name: **{person['name']}**")

        st.metric("Similarity", f"{result['similarity']:.2%}")

    else:
        st.error(result["message"])

        if result.get("similarity") is not None:
            st.write("Best similarity:", f"{result['similarity']:.2%}")


# Register Section
def register_section():

    st.header("Register Face")

    name = st.text_input("Name", placeholder="Enter person's name")

    st.subheader("Face Image")

    selected_image = get_image("register")

    show_image(selected_image, "Selected image")

    st.divider()

    if st.button("Register Face", type="primary", use_container_width=True):

        if not name.strip():

            st.error("Please enter a name.")
            return

        if selected_image is None:

            st.error("Please capture or upload an image.")
            return

        try:
            with st.spinner("Detecting face and registering..."):

                result = register_face(
                    name=name.strip(),
                    image_file=selected_image
                )

            if result["success"]:

                st.success("Face registered successfully.")

                st.info(f"Person ID: {result['person_id']}")

            else:
                st.error(result["message"])

        except Exception as error:

            st.error("Registration failed.")

            st.exception(error)


# Search Section
def search_section():

    st.header("Search Face")

    st.write("Capture or upload a face to search the database.")

    st.subheader("Face Image")

    selected_image = get_image("search")

    show_image(selected_image, "Search image")

    st.divider()

    if st.button("Search Face", type="primary", use_container_width=True):

        if selected_image is None:

            st.error("Please capture or upload an image.")
            return

        try:
            with st.spinner("Searching face..."):

                result = search_face(image_file=selected_image)

            show_match_result(result)

        except Exception as error:

            st.error("Search failed.")

            st.exception(error)


# Header
st.title("🧑 Face Recognition System")

st.write("Register a face or search for a registered person.")

# Operation Selection
operation = st.radio(
    "Choose operation",
    ["Register", "Search"],
    horizontal=True
)

st.divider()

# Main Application
if operation == "Register":
    register_section()

else:
    search_section()