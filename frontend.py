import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Set page configuration to wide mode and add custom styling
st.set_page_config(layout="wide", page_title="Iris Species Classifier")

# Custom CSS for floral theme
st.markdown("""
    <style>
    .stApp {
        background-color: #fdf6f6;
    }
    .stButton>button {
        background-color: #e6b3cc;
        color: #4a4a4a;
        border: 2px solid #d199b6;
    }
    .stButton>button:hover {
        background-color: #d199b6;
        color: white;
    }
    .css-1d391kg {
        background-color: #fff5f5;
    }
    div.row-widget.stRadio > div {
        background-color: #fff5f5;
        border-radius: 5px;
        padding: 10px;
    }
    .stNumberInput > div > div > input {
        background-color: #fff5f5;
    }
    .title-text {
        color: #967bb6;
    }
    .description-text {
        color: #967bb6;
    }
    .input-label {
        color: #967bb6;
        font-size: 1rem;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load the saved model
with open('iris_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load the dataset for visualization
iris_data = pd.read_csv('C:\Projects\OpenCV\iris.csv')
iris_data = iris_data.drop(columns=['Id'])

# Title row
st.markdown("<h1 class='title-text'>🌸 Iris Species Classification</h1>", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([3, 2])

with col1:
    # Create a container for input fields with custom styling
    with st.container():
        st.markdown("""
            <div style='background-color: #fff5f5; padding: 10px; border-radius: 10px; border: 2px solid #e6b3cc;'>
            <h3 style='color: #967bb6;'>Enter Flower Measurements</h3>
            """, unsafe_allow_html=True)
        
        # Input fields in two columns within the container
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            st.markdown("<p class='input-label'>Sepal Length (cm)</p>", unsafe_allow_html=True)
            sepal_length = st.number_input("", min_value=0.0, max_value=10.0, value=0.0, step=0.1, label_visibility="collapsed")
            st.markdown("<p class='input-label'>Sepal Width (cm)</p>", unsafe_allow_html=True)
            sepal_width = st.number_input(" ", min_value=0.0, max_value=10.0, value=0.0, step=0.1, label_visibility="collapsed")
        
        with input_col2:
            st.markdown("<p class='input-label'>Petal Length (cm)</p>", unsafe_allow_html=True)
            petal_length = st.number_input("  ", min_value=0.0, max_value=10.0, value=0.0, step=0.1, label_visibility="collapsed")
            st.markdown("<p class='input-label'>Petal Width (cm)</p>", unsafe_allow_html=True)
            petal_width = st.number_input("   ", min_value=0.0, max_value=10.0, value=0.0, step=0.1, label_visibility="collapsed")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Create columns for prediction button and result
    pred_col1, pred_col2 = st.columns([1, 2])
    
    with pred_col1:
        predict_clicked = st.button("🔍 Predict Species")
    
    with pred_col2:
        if predict_clicked:
            user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
            prediction = loaded_model.predict(user_input)
            label_encoder = pd.factorize(iris_data['Species'])[1]
            species = label_encoder[prediction[0]]
            
            st.markdown(f"""
                <div style='background-color: #fff5f5; border-radius: 10px; border: 2px solid #e6b3cc;'>
                    <h3 style='color: #967bb6; text-align: center; margin: 0;'>Predicted Species: {species}</h3>
                </div>
            """, unsafe_allow_html=True)

    # Visualization button
    if st.button("📊 Show Visualization"):
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(10, 8))
        sns.pairplot(iris_data, hue='Species', palette='husl')
        st.pyplot(plt)

with col2:
    # Add description and image
    st.markdown("""
        <div style='background-color: #fff5f5; padding: 20px; border-radius: 10px; border: 2px solid #e6b3cc;'>
            <p class='description-text'><strong>Iris classification is a machine learning problem that involves categorizing iris flowers into one of three species: Iris setosa, Iris versicolor, or Iris virginica. The classification is based on measurements of the flower's sepals and petals, such as their length and width.</strong></p>
        </div>
    """, unsafe_allow_html=True)
    
    # Add placeholder image
    st.image("C:/Users/jayde/Downloads/iris.png", caption="Iris Flower Species", use_column_width=True)

# Add footer
st.markdown("""
    <div style='text-align: center; padding: 20px; color: #888;'>
        <p>Built with Streamlit • Made with 💐</p>
    </div>
""", unsafe_allow_html=True)