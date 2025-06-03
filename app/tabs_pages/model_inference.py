import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from pipeline import ToxicityClassifierInference
from globals import COLORS

def create_model_inference():
    """Create model inference tab content"""
    st.header("🔮 Model Inference")
    st.markdown(
        """
        Use the trained toxicity classification model to analyze text for toxic content.
        The model can detect multiple types of toxicity including threats, insults, obscenity, and more.
        """
    )
    
    # Check if model is loaded
    if not st.session_state.get('model_loaded', False) or 'classifier' not in st.session_state:
        st.warning(
            """
            ⚠️ **No model loaded!** 
            
            Please go to the **Model Loader** tab first to:
            - Train a new model, or
            - Load an existing model
            """
        )
        return
    
    classifier = st.session_state['classifier']
    
    # Inference mode selection
    st.subheader("🎯 Select Inference Mode")
    inference_mode = st.radio(
        "Choose how you want to analyze text:",
        ["Single Text Analysis", "Batch Analysis", "Interactive Chat Mode"],
        horizontal=True
    )
    
    st.divider()
    
    if inference_mode == "Single Text Analysis":
        _single_text_analysis(classifier)
    elif inference_mode == "Batch Analysis":
        _batch_analysis(classifier)
    else:  # Interactive Chat Mode
        _interactive_chat_mode(classifier)

def _single_text_analysis(classifier):
    """Single text analysis interface"""
    st.subheader("📝 Single Text Analysis")
    
    # Text input options
    input_method = st.radio("Input method:", ["Text Area", "Text Examples"], horizontal=True)
    
    if input_method == "Text Examples":
        example_texts = {
            "Neutral": "I love this new product, it works great!",
            "Mild Negative": "This is not very good, disappointed.",
            "Potentially Toxic": "What a stupid idea, complete waste of time.",
            "Clearly Toxic": "You're such an idiot, go kill yourself.",
            "Custom": ""
        }
        
        selected_example = st.selectbox("Choose an example:", list(example_texts.keys()))
        input_text = st.text_area(
            "Text to analyze:",
            value=example_texts[selected_example],
            height=100,
            help="You can modify the example text or write your own"
        )
    else:
        input_text = st.text_area(
            "Enter text to analyze:",
            height=100,
            placeholder="Type or paste the text you want to analyze for toxicity..."
        )
    
    # Analysis settings
    col1, col2 = st.columns(2)
    with col1:
        show_preprocessing = st.checkbox("Show text preprocessing steps", value=False)
    with col2:
        threshold = st.slider("Classification threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Analyze button
    if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Please enter some text to analyze")
            return
        
        try:
            with st.spinner("Analyzing text..."):
                # Get detailed prediction
                result = classifier.predict_toxic(input_text, verbose=False)
                probabilities = classifier.predict_single(input_text, return_probabilities=True)
                
                # Display results
                st.success("✅ Analysis completed!")
                
                # Overall toxicity status
                is_toxic = result['is_toxic']
                max_score = result['max_toxicity_score']
                max_class = result['max_toxicity_class']
                
                if is_toxic:
                    st.error(f"🚨 **TOXIC CONTENT DETECTED**")
                    st.error(f"Primary concern: **{max_class.title()}** ({max_score:.1%})")
                else:
                    if max_score > threshold:
                        st.warning(f"⚠️ **POTENTIALLY CONCERNING**")
                        st.warning(f"Highest score: **{max_class.title()}** ({max_score:.1%})")
                    else:
                        st.success(f"✅ **CONTENT APPEARS SAFE**")
                        st.info(f"Highest score: **{max_class.title()}** ({max_score:.1%})")
                
                # Detailed breakdown
                st.subheader("📊 Detailed Analysis")
                
                # Create visualization
                classes = list(probabilities.keys())
                scores = list(probabilities.values())
                
                # Color code based on scores
                colors = []
                for score in scores:
                    if score > 0.7:
                        colors.append('#FF4B4B')  # High risk - red
                    elif score > threshold:
                        colors.append('#FFA500')  # Medium risk - orange
                    else:
                        colors.append('#00C851')  # Low risk - green
                
                # Create bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=[c.title() for c in classes],
                        y=scores,
                        marker_color=colors,
                        text=[f'{s:.1%}' for s in scores],
                        textposition='auto',

                    )
                ])
                
                fig.update_layout(
                    title="Toxicity Scores by Category",
                    xaxis_title="Toxicity Categories",
                    yaxis_title="Probability Score",
                    yaxis=dict(range=[0, 1]),
                    paper_bgcolor=COLORS['transparent'],
                    plot_bgcolor=COLORS['transparent'],
                    height=400
                )
                
                # Add threshold line
                fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                             annotation_text=f"Threshold: {threshold}")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                results_df = pd.DataFrame([
                    {
                        "Category": class_name.title(),
                        "Probability": f"{prob:.1%}",
                        "Risk Level": "High" if prob > 0.7 else "Medium" if prob > threshold else "Low",
                        "Status": "🚨 Flagged" if prob > threshold else "✅ Safe"
                    }
                    for class_name, prob in probabilities.items()
                ]).sort_values("Probability", ascending=False)
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Show preprocessing if requested
                if show_preprocessing:
                    st.subheader("🔧 Text Preprocessing")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original Text:**")
                        st.code(result['raw_text'])
                    
                    with col2:
                        st.markdown("**Processed Text:**")
                        st.code(result['processed_text'])
                
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.exception(e)

def _batch_analysis(classifier):
    """Batch analysis interface"""
    st.subheader("📊 Batch Analysis")
    
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload CSV file with text data",
        type=['csv'],
        help="Upload a CSV file with a column containing text to analyze"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows")
            
            # Column selection
            text_column = st.selectbox(
                "Select the text column to analyze:",
                df.columns.tolist()
            )
            
            # Analysis settings
            col1, col2, col3 = st.columns(3)
            with col1:
                threshold = st.slider("Classification threshold", 0.0, 1.0, 0.5, 0.05)
            with col2:
                max_rows = st.number_input("Max rows to analyze", 1, len(df), min(1000, len(df)))
            with col3:
                show_progress = st.checkbox("Show detailed progress", value=True)
            
            # Analyze button
            if st.button("🚀 Start Batch Analysis", type="primary", use_container_width=True):
                try:
                    # Prepare data
                    texts_to_analyze = df[text_column].fillna("").astype(str).head(max_rows).tolist()
                    
                    # Progress tracking
                    if show_progress:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                    
                    results = []
                    
                    # Analyze in batches for better performance
                    batch_size = 50
                    for i in range(0, len(texts_to_analyze), batch_size):
                        batch_texts = texts_to_analyze[i:i+batch_size]
                        
                        if show_progress:
                            progress = (i + len(batch_texts)) / len(texts_to_analyze)
                            progress_bar.progress(progress)
                            status_text.text(f"Analyzing batch {i//batch_size + 1}...")
                        
                        # Get predictions for batch
                        batch_probs = classifier.predict_probabilities(batch_texts)
                        batch_predictions = classifier.predict(batch_texts, threshold=threshold)
                        
                        # Process results
                        for j, (text, probs, preds) in enumerate(zip(batch_texts, batch_probs, batch_predictions)):
                            results.append({
                                'original_index': i + j,
                                'text': text[:100] + "..." if len(text) > 100 else text,
                                'is_toxic': any(preds),
                                'max_toxicity_score': max(probs),
                                'max_toxicity_class': classifier.classes[np.argmax(probs)],
                                **{f'{cls}_prob': prob for cls, prob in zip(classifier.classes, probs)},
                                **{f'{cls}_pred': pred for cls, pred in zip(classifier.classes, preds)}
                            })
                    
                    if show_progress:
                        progress_bar.progress(1.0)
                        status_text.text("Analysis completed!")
                    
                    # Create results dataframe
                    results_df = pd.DataFrame(results)
                    
                    # Summary statistics
                    st.subheader("📈 Summary Statistics")
                    
                    total_analyzed = len(results_df)
                    toxic_count = results_df['is_toxic'].sum()
                    toxic_rate = toxic_count / total_analyzed if total_analyzed > 0 else 0
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Analyzed", total_analyzed)
                    with col2:
                        st.metric("Toxic Content", toxic_count)
                    with col3:
                        st.metric("Toxicity Rate", f"{toxic_rate:.1%}")
                    with col4:
                        avg_max_score = results_df['max_toxicity_score'].mean()
                        st.metric("Avg Max Score", f"{avg_max_score:.1%}")
                    
                    # Distribution visualization
                    st.subheader("📊 Toxicity Distribution")
                    
                    # Class distribution
                    class_counts = {}
                    for cls in classifier.classes:
                        class_counts[cls.title()] = results_df[f'{cls}_pred'].sum()
                    
                    fig = px.bar(
                        x=list(class_counts.keys()),
                        y=list(class_counts.values()),
                        title="Detected Toxicity by Category",
                        labels={'x': 'Toxicity Category', 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    
                    # Filter options
                    col1, col2 = st.columns(2)
                    with col1:
                        show_filter = st.selectbox(
                            "Show results:",
                            ["All", "Toxic Only", "Safe Only"]
                        )
                    with col2:
                        sort_by = st.selectbox(
                            "Sort by:",
                            ["Max Toxicity Score", "Original Order"]
                        )
                    
                    # Apply filters
                    display_df = results_df.copy()
                    if show_filter == "Toxic Only":
                        display_df = display_df[display_df['is_toxic']]
                    elif show_filter == "Safe Only":
                        display_df = display_df[~display_df['is_toxic']]
                    
                    if sort_by == "Max Toxicity Score":
                        display_df = display_df.sort_values('max_toxicity_score', ascending=False)
                    
                    # Select columns to display
                    display_columns = ['text', 'is_toxic', 'max_toxicity_score', 'max_toxicity_class']
                    display_df_show = display_df[display_columns].copy()
                    display_df_show.columns = ['Text', 'Is Toxic', 'Max Score', 'Primary Concern']
                    
                    st.dataframe(display_df_show, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results",
                        data=csv,
                        file_name="toxicity_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Batch analysis failed: {str(e)}")
                    st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Failed to load file: {str(e)}")
    else:
        st.info(
            """
            📁 **Upload a CSV file to get started with batch analysis**
            
            Your CSV should contain:
            - A column with text data to analyze
            - Any other columns you want to keep for reference
            
            The analysis will add toxicity predictions and probabilities for each text.
            """
        )

def _interactive_chat_mode(classifier):
    """Interactive chat mode interface"""
    st.subheader("💬 Interactive Chat Mode")
    st.markdown("Test the model in a conversational format - type messages and see real-time toxicity analysis.")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Settings
    col1, col2, col3 = st.columns(3)
    with col1:
        threshold = st.slider("Alert threshold", 0.0, 1.0, 0.5, 0.05)
    with col2:
        auto_moderate = st.checkbox("Auto-moderate toxic content", value=False)
    with col3:
        show_scores = st.checkbox("Show detailed scores", value=True)
    
    # Chat input
    user_input = st.chat_input("Type a message to analyze...")
    
    if user_input:
        try:
            # Analyze the input
            result = classifier.predict_single(user_input, return_probabilities=True)
            max_score = max(result.values())
            max_class = max(result, key=result.get)
            is_concerning = max_score > threshold
            
            # Determine if message should be moderated
            if auto_moderate and is_concerning:
                moderated_message = "[Message flagged by toxicity filter]"
                status = "moderated"
            else:
                moderated_message = user_input
                status = "concerning" if is_concerning else "safe"
            
            # Add to chat history
            st.session_state.chat_history.append({
                "message": user_input,
                "displayed_message": moderated_message,
                "status": status,
                "max_score": max_score,
                "max_class": max_class,
                "scores": result,
                "timestamp": pd.Timestamp.now()
            })
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Conversation History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-20:])):  # Show last 20 messages
            # Message container
            if chat["status"] == "moderated":
                message_type = "error"
                icon = "🚨"
            elif chat["status"] == "concerning":
                message_type = "warning"
                icon = "⚠️"
            else:
                message_type = "success"
                icon = "✅"
            
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    if message_type == "error":
                        st.error(f"{icon} {chat['displayed_message']}")
                    elif message_type == "warning":
                        st.warning(f"{icon} {chat['displayed_message']}")
                    else:
                        st.success(f"{icon} {chat['displayed_message']}")
                    
                    if show_scores and chat["status"] != "moderated":
                        score_text = f"**{chat['max_class'].title()}**: {chat['max_score']:.1%}"
                        if chat["status"] == "concerning":
                            st.caption(f"🔍 Primary concern: {score_text}")
                        else:
                            st.caption(f"🔍 Highest score: {score_text}")
                
                with col2:
                    # Show timestamp
                    time_str = chat["timestamp"].strftime("%H:%M:%S")
                    st.caption(f"🕒 {time_str}")
        
        # Chat statistics
        st.divider()
        st.subheader("📊 Chat Statistics")
        
        total_messages = len(st.session_state.chat_history)
        concerning_messages = sum(1 for chat in st.session_state.chat_history if chat["status"] in ["concerning", "moderated"])
        moderated_messages = sum(1 for chat in st.session_state.chat_history if chat["status"] == "moderated")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Messages", total_messages)
        with col2:
            st.metric("Concerning", concerning_messages)
        with col3:
            st.metric("Moderated", moderated_messages)
        with col4:
            concern_rate = concerning_messages / total_messages if total_messages > 0 else 0
            st.metric("Concern Rate", f"{concern_rate:.1%}")
        
        # Clear history button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    else:
        st.info("💡 Start typing messages above to see real-time toxicity analysis!")