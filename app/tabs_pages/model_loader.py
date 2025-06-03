import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys
import os

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from pipeline import ToxicityClassifierTrainer, ToxicityClassifierInference
from globals import COLORS

def create_model_loader():
    """Create model loader tab content"""
    st.header("🔧 Model Training & Loading")
    st.markdown(
        """
        This tab allows you to train a new toxicity classification model or load an existing one.
        The model uses the same methodology from the research notebook with balanced datasets and logistic regression.
        """
    )
    
    # Create two columns for training and loading
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Train New Model")
        st.markdown(
            """
            Train a new model using your training data. The model will use:
            - Balanced datasets for each toxicity class
            - Individual logistic regression classifiers
            - TF-IDF text vectorization
            - Advanced text preprocessing pipeline
            """
        )
        
        # Training data file uploader
        uploaded_file = st.file_uploader(
            "Upload Training Data (CSV)",
            type=['csv'],
            help="Upload a CSV file with 'comment_text' column and toxicity labels"
        )
        
        # Default data path option
        use_default_data = st.checkbox(
            "Use default training data (data/train.csv)",
            value=True,
            help="Use the default training data located at data/train.csv"
        )
        
        # Model save path
        model_save_path = st.text_input(
            "Model Save Path",
            value="utils/models/toxicity_model.pkl",
            help="Path where the trained model will be saved"
        )
        
        # Training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            if not use_default_data and uploaded_file is None:
                st.error("Please upload a training file or use default data")
            else:
                try:
                    with st.spinner("Training model... This may take several minutes."):
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Initialize trainer
                        status_text.text("Initializing trainer...")
                        progress_bar.progress(10)
                        trainer = ToxicityClassifierTrainer()
                        
                        # Load and preprocess data
                        status_text.text("Loading and preprocessing data...")
                        progress_bar.progress(30)
                        
                        if use_default_data:
                            trainer.fit()
                        else:
                            # Load uploaded file
                            df = pd.read_csv(uploaded_file)
                            trainer.fit(data=df)
                        
                        progress_bar.progress(80)
                        
                        # Save model
                        status_text.text("Saving trained model...")
                        trainer.save_model(model_save_path)
                        progress_bar.progress(100)
                        
                        status_text.text("Training completed successfully!")
                        st.success(f"✅ Model trained and saved to {model_save_path}")
                        
                        # Store in session state
                        st.session_state['trained_model_path'] = model_save_path
                        st.session_state['model_trained'] = True
                        
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    st.exception(e)
    
    with col2:
        st.subheader("📂 Load Existing Model")
        st.markdown(
            """
            Load a pre-trained toxicity classification model for inference.
            The model should be saved in pickle format from this pipeline.
            """
        )
        
        # Model file uploader
        model_file = st.file_uploader(
            "Upload Model File (PKL)",
            type=['pkl'],
            help="Upload a trained model file in pickle format"
        )
        
        # Model path input
        model_path_input = st.text_input(
            "Or enter model path",
            value="utils/models/toxicity_model.pkl",
            help="Path to an existing model file"
        )
        
        # Load button
        if st.button("📥 Load Model", type="secondary", use_container_width=True):
            try:
                with st.spinner("Loading model..."):
                    if model_file is not None:
                        # Save uploaded file temporarily
                        temp_path = f"temp_model_{model_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(model_file.getvalue())
                        
                        # Load from temporary file
                        classifier = ToxicityClassifierInference(temp_path)
                        
                        # Clean up temporary file
                        os.remove(temp_path)
                        
                        # Store in session state
                        st.session_state['classifier'] = classifier
                        st.session_state['model_loaded'] = True
                        st.session_state['model_source'] = "uploaded"
                        
                    else:
                        # Load from path
                        if not Path(model_path_input).exists():
                            st.error(f"Model file not found at {model_path_input}")
                            return
                        
                        classifier = ToxicityClassifierInference(model_path_input)
                        
                        # Store in session state
                        st.session_state['classifier'] = classifier
                        st.session_state['model_loaded'] = True
                        st.session_state['model_source'] = model_path_input
                    
                    st.success("✅ Model loaded successfully!")
                    
            except Exception as e:
                st.error(f"❌ Failed to load model: {str(e)}")
                st.exception(e)
    
    # Model status display
    st.divider()
    st.subheader("📊 Model Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.get('model_loaded', False):
            st.success("🟢 Model Loaded")
            if 'classifier' in st.session_state:
                st.info(f"Classes: {len(st.session_state['classifier'].classes)}")
        else:
            st.warning("🟡 No Model Loaded")
    
    with col2:
        if st.session_state.get('model_trained', False):
            st.success("🟢 Model Trained")
            if 'trained_model_path' in st.session_state:
                st.info(f"Saved to: {st.session_state['trained_model_path']}")
        else:
            st.info("🔵 No Training Session")
    
    with col3:
        if st.session_state.get('model_loaded', False):
            source = st.session_state.get('model_source', 'Unknown')
            if source == "uploaded":
                st.info("📁 Source: Uploaded")
            else:
                st.info(f"📁 Source: {source}")
        else:
            st.info("📁 Source: None")
    
    # Model information display
    if st.session_state.get('model_loaded', False) and 'classifier' in st.session_state:
        st.divider()
        st.subheader("🔍 Model Information")
        
        classifier = st.session_state['classifier']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Toxicity Classes:**")
            for i, class_name in enumerate(classifier.classes):
                st.write(f"{i+1}. {class_name.title()}")
        
        with col2:
            st.markdown("**Model Details:**")
            st.write(f"- **Models count:** {len(classifier.models)}")
            st.write(f"- **Vectorizer:** TF-IDF")
            st.write(f"- **Algorithm:** Logistic Regression Ensemble")
            st.write(f"- **Text Processor:** Advanced Pipeline")
        
        # Quick test section
        st.divider()
        st.subheader("🧪 Quick Model Test")
        
        test_text = st.text_input(
            "Enter test text:",
            value="This is a test message",
            help="Enter any text to quickly test the model"
        )
        
        if st.button("🔍 Test Model", use_container_width=True):
            if test_text.strip():
                try:
                    with st.spinner("Running prediction..."):
                        result = classifier.predict_single(test_text, return_probabilities=True)
                        
                        st.success("✅ Prediction completed!")
                        
                        # Display results
                        result_df = pd.DataFrame([
                            {"Class": class_name.title(), "Probability": f"{prob:.1%}"}
                            for class_name, prob in result.items()
                        ])
                        
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Show highest probability
                        max_class = max(result, key=result.get)
                        max_prob = result[max_class]
                        
                        if max_prob > 0.5:
                            st.warning(f"⚠️ Highest toxicity: **{max_class.title()}** ({max_prob:.1%})")
                        else:
                            st.info(f"ℹ️ Highest score: **{max_class.title()}** ({max_prob:.1%})")
                            
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
            else:
                st.warning("Please enter some text to test")
    
    # Instructions
    if not st.session_state.get('model_loaded', False):
        st.divider()
        st.info(
            """
            💡 **Getting Started:**
            1. Either train a new model using your data or load an existing one
            2. For training, you can use the default data or upload your own CSV file
            3. Once loaded, you can test the model here or use it for inference in the next tab
            4. The training data should have columns: 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
            """
        )