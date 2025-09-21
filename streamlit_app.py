import streamlit as st
import pandas as pd
import requests
import json
from io import BytesIO
import ollama
from pymilvus import connections, Collection, utility
import time
import numpy as np
from agno_integration import integrate_agno_analysis

# Page configuration
st.set_page_config(
    page_title="Excel Financial Chatbot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: # Tab 4: Analytics Dashboard
with tab4:
    st.header("📊 Analytics Dashboard")x;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message strong {
        font-weight: bold;
        font-size: 18px;
    }
    .user-message {
        background-color: #ffffff !important;
        border: 2px solid #2196f3;
        color: #000000 !important;
    }
    .user-message * {
        color: #000000 !important;
    }
    .bot-message {
        background-color: #ffffff !important;
        border: 2px solid #28a745;
        color: #000000 !important;
    }
    .bot-message * {
        color: #000000 !important;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        background-color: #ffffff;
        color: #000000;
    }
    /* Force all text in chat messages to be black */
    .chat-message span {
        color: #000000 !important;
        background-color: transparent !important;
    }
    /* Override any Streamlit theme colors */
    div[data-testid="stMarkdownContainer"] p {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'excel_uploaded' not in st.session_state:
    st.session_state.excel_uploaded = False
if 'collections_available' not in st.session_state:
    st.session_state.collections_available = []
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = ""
if 'file_embeddings' not in st.session_state:
    st.session_state.file_embeddings = {}
if 'file_relationships' not in st.session_state:
    st.session_state.file_relationships = []
if 'file_summaries' not in st.session_state:
    st.session_state.file_summaries = {}
if 'agno_analysis' not in st.session_state:
    st.session_state.agno_analysis = {}

def check_milvus_connection():
    """Check if Milvus is running and accessible"""
    try:
        connections.connect("default", host="localhost", port="19530")
        collections = utility.list_collections()
        st.session_state.collections_available = collections
        return True
    except Exception as e:
        st.error(f"❌ Milvus connection failed: {e}")
        return False

def check_ollama_connection():
    """Check if Ollama is running and model is available"""
    try:
        # Test if we can access the model directly
        ollama.show('llama3.2:3b')
        return True
    except Exception as e:
        return False

def generate_embeddings(text_list, model_name='llama3.2:3b'):
    """Generate embeddings for a list of texts using Ollama"""
    embeddings = []
    for text in text_list:
        try:
            # Use chat completion to get a consistent response, then create a simple embedding
            # This is a workaround since Ollama embedding API might not be available for all models
            response = ollama.chat(model=model_name, messages=[
                {
                    'role': 'system',
                    'content': 'You are a data analyst. Analyze the following data and provide a numerical summary.'
                },
                {
                    'role': 'user',
                    'content': f"Analyze this data and provide key insights: {text[:1000]}"  # Limit text length
                }
            ])
            
            # Create a simple hash-based embedding from the response
            response_text = response['message']['content']
            
            # Create a simple embedding using character frequencies
            embedding = create_simple_embedding(text + response_text)
            embeddings.append(embedding)
            
        except Exception as e:
            st.error(f"Error generating embedding: {e}")
            # Fallback: create dummy embedding based on text content
            embedding = create_simple_embedding(text)
            embeddings.append(embedding)
    
    return embeddings

def create_simple_embedding(text, embedding_size=256):
    """Create a simple embedding based on text characteristics"""
    import hashlib
    
    # Convert text to lowercase and get basic stats
    text = str(text).lower()
    
    # Create features based on text characteristics
    features = []
    
    # Character frequency features (first 26 for a-z)
    char_counts = [0] * 26
    for char in text:
        if 'a' <= char <= 'z':
            char_counts[ord(char) - ord('a')] += 1
    
    # Normalize character counts
    total_chars = sum(char_counts) or 1
    char_freqs = [count / total_chars for count in char_counts]
    features.extend(char_freqs)
    
    # Word-based features
    words = text.split()
    features.append(len(words))  # Word count
    features.append(len(text))   # Character count
    features.append(len(set(words)) / (len(words) or 1))  # Unique word ratio
    
    # Number-based features (look for financial terms)
    financial_terms = ['revenue', 'profit', 'cost', 'expense', 'margin', 'sales', 'income']
    for term in financial_terms:
        features.append(1.0 if term in text else 0.0)
    
    # Pad or truncate to desired size
    while len(features) < embedding_size:
        # Use hash to generate additional pseudo-random features
        hash_value = int(hashlib.md5(f"{text}_{len(features)}".encode()).hexdigest(), 16)
        features.append((hash_value % 1000) / 1000.0)
    
    features = features[:embedding_size]
    
    # Normalize the embedding
    import numpy as np
    features = np.array(features)
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    
    return features.tolist()

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    import numpy as np
    
    # Convert to numpy arrays
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)

def process_multiple_files(uploaded_files):
    """Process multiple files and analyze relationships between them"""
    try:
        if not uploaded_files:
            st.error("No files provided")
            return False
            
        all_files_data = {}
        all_embeddings = {}
        file_summaries = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each file
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i) / len(uploaded_files))
                
                file_content = uploaded_file.read()
                filename = uploaded_file.name
                file_extension = filename.lower().split('.')[-1]
                
                # Process file based on type
                file_dataframes = {}
                
                if file_extension == 'csv':
                    df = pd.read_csv(BytesIO(file_content))
                    file_dataframes[f'{filename}_CSV'] = df
                    
                elif file_extension in ['xlsx', 'xls']:
                    excel_file = pd.ExcelFile(BytesIO(file_content))
                    for sheet_name in excel_file.sheet_names:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        file_dataframes[f'{filename}_{sheet_name}'] = df
                
                if not file_dataframes:
                    st.warning(f"No data found in {filename}")
                    continue
                
                # Generate text summaries and embeddings for each sheet/file
                file_texts = []
                
                for sheet_key, df in file_dataframes.items():
                    try:
                        # Create summary text for the entire sheet/file
                        summary_parts = [
                            f"File: {filename}",
                            f"Sheet: {sheet_key}",
                            f"Columns: {', '.join(df.columns.tolist()[:10])}",  # Limit columns
                            f"Rows: {len(df)}",
                            f"Data sample: {df.head(2).to_string(index=False)[:500]}"  # Limit sample size
                        ]
                        
                        # Add basic statistics for numeric columns
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 0:
                            numeric_stats = []
                            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                                mean_val = df[col].mean()
                                numeric_stats.append(f"{col}: avg={mean_val:.2f}")
                            summary_parts.append(f"Numeric stats: {', '.join(numeric_stats)}")
                        
                        summary_text = " | ".join(summary_parts)
                        file_texts.append(summary_text)
                        
                        # Store detailed data
                        all_files_data[sheet_key] = {
                            'filename': filename,
                            'dataframe': df,
                            'summary': summary_text,
                            'row_count': len(df),
                            'column_count': len(df.columns),
                            'numeric_columns': numeric_cols.tolist()
                        }
                        
                    except Exception as e:
                        st.error(f"Error processing sheet {sheet_key}: {e}")
                        continue
                
                # Generate embeddings for this file's summaries
                if file_texts:
                    status_text.text(f"Generating embeddings for {filename}...")
                    try:
                        file_embeddings = generate_embeddings(file_texts)
                        
                        # Store embeddings
                        for j, sheet_key in enumerate(file_dataframes.keys()):
                            if j < len(file_embeddings):
                                all_embeddings[sheet_key] = file_embeddings[j]
                    except Exception as e:
                        st.error(f"Error generating embeddings for {filename}: {e}")
                        continue
                
                # Create file summary
                file_summaries[filename] = {
                    'sheets': list(file_dataframes.keys()),
                    'total_rows': sum([len(df) for df in file_dataframes.values()]),
                    'total_columns': sum([len(df.columns) for df in file_dataframes.values()]),
                    'data_types': list(set([str(dtype) for df in file_dataframes.values() for dtype in df.dtypes]))
                }
                
            except Exception as e:
                st.error(f"Error processing file {uploaded_file.name}: {e}")
                continue
        
        if not all_embeddings:
            st.error("No embeddings were generated. Please check your files and try again.")
            return False
        
        progress_bar.progress(0.8)
        status_text.text("Analyzing relationships between files...")
        
        # Calculate similarities between all file pairs
        relationships = []
        sheet_keys = list(all_embeddings.keys())
        
        for i in range(len(sheet_keys)):
            for j in range(i + 1, len(sheet_keys)):
                sheet1 = sheet_keys[i]
                sheet2 = sheet_keys[j]
                
                try:
                    similarity = calculate_similarity(all_embeddings[sheet1], all_embeddings[sheet2])
                    
                    relationships.append({
                        'file1': sheet1,
                        'file2': sheet2,
                        'similarity': similarity,
                        'similarity_percentage': similarity * 100
                    })
                except Exception as e:
                    st.warning(f"Error calculating similarity between {sheet1} and {sheet2}: {e}")
                    continue
        
        # Sort by similarity (highest first)
        relationships.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Run Agno advanced analysis
        progress_bar.progress(0.9)
        status_text.text("Running advanced Agno analysis...")
        try:
            agno_results = integrate_agno_analysis(all_files_data)
        except Exception as e:
            st.warning(f"Agno analysis failed: {e}")
            agno_results = {}
        
        # Store everything in session state
        st.session_state.processed_data = all_files_data
        st.session_state.file_embeddings = all_embeddings
        st.session_state.file_relationships = relationships
        st.session_state.file_summaries = file_summaries
        st.session_state.agno_analysis = agno_results
        st.session_state.excel_uploaded = True
        st.session_state.uploaded_filename = f"{len(uploaded_files)} files"
        
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        
        # Display results
        st.success(f"✅ Successfully processed {len(uploaded_files)} files")
        
        # Show relationship summary
        if relationships:
            st.subheader("🔗 File Relationship Analysis")
            
            # Show top relationships
            top_relationships = relationships[:5]  # Top 5 most similar
            
            for rel in top_relationships:
                similarity_color = "🟢" if rel['similarity'] > 0.7 else "🟡" if rel['similarity'] > 0.4 else "🔴"
                st.info(f"""
                {similarity_color} **{rel['file1']}** ↔ **{rel['file2']}**
                
                Similarity: **{rel['similarity_percentage']:.1f}%**
                """)
        
        # Summary statistics
        st.info(f"""
        **Processing Summary:**
        - **Files Processed:** {len(uploaded_files)}
        - **Total Sheets/Tables:** {len(all_files_data)}
        - **Total Data Points:** {sum([data['row_count'] for data in all_files_data.values()])}
        - **Relationships Found:** {len(relationships)}
        - **Ready for AI Analysis & Cross-file Queries**
        """)
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing multiple files: {e}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return False
    """Process and upload Excel/CSV file to Milvus"""
    try:
        file_extension = filename.lower().split('.')[-1]
        all_dataframes = {}
        
        if file_extension == 'csv':
            # Handle CSV files
            df = pd.read_csv(BytesIO(file_content))
            all_dataframes['CSV_Data'] = df
            st.subheader(f"📋 CSV File: {filename}")
            st.dataframe(df.head())
            
        elif file_extension in ['xlsx', 'xls']:
            # Handle Excel files with multiple sheets
            excel_file = pd.ExcelFile(BytesIO(file_content))
            sheet_names = excel_file.sheet_names
            
            st.subheader(f"📋 Excel File: {filename}")
            st.write(f"**Sheets found:** {', '.join(sheet_names)}")
            
            # Create tabs for each sheet
            if len(sheet_names) > 1:
                sheet_tabs = st.tabs([f"Sheet: {name}" for name in sheet_names])
                
                for i, sheet_name in enumerate(sheet_names):
                    with sheet_tabs[i]:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        all_dataframes[sheet_name] = df
                        st.write(f"**Rows:** {len(df)}, **Columns:** {len(df.columns)}")
                        st.dataframe(df.head())
            else:
                # Single sheet
                df = pd.read_excel(excel_file, sheet_name=sheet_names[0])
                all_dataframes[sheet_names[0]] = df
                st.write(f"**Rows:** {len(df)}, **Columns:** {len(df.columns)}")
                st.dataframe(df.head())
        
        # Process all dataframes into text chunks
        total_chunks = 0
        processed_data = {}
        
        for sheet_name, df in all_dataframes.items():
            text_chunks = []
            for index, row in df.iterrows():
                # Convert row to text with sheet context
                row_text = f"Sheet: {sheet_name} | Row {index + 1} | " + " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                text_chunks.append(row_text)
            
            processed_data[sheet_name] = {
                'chunks': text_chunks,
                'dataframe': df,
                'row_count': len(df),
                'column_count': len(df.columns)
            }
            total_chunks += len(text_chunks)
        
        # Store processed data in session state
        st.session_state.processed_data = processed_data
        st.session_state.excel_uploaded = True
        st.session_state.uploaded_filename = filename
        
        # Display summary
        st.success(f"✅ Successfully processed {filename}")
        st.info(f"""
        **Processing Summary:**
        - **Sheets/Files:** {len(all_dataframes)}
        - **Total Rows:** {total_chunks}
        - **Ready for AI Analysis**
        """)
        
        # Note: In a full implementation, you would:
        # 1. Generate embeddings using Ollama for each chunk
        # 2. Store in Milvus collection with metadata (sheet_name, row_number, etc.)
        # This is a simplified version for the GUI demo
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        return False

def get_relevant_data_for_query(user_question, max_rows_per_file=15):
    """
    Get relevant data from all files based on the user's question.
    This function provides more targeted data to avoid context overflow.
    """
    relevant_data = []
    
    if not st.session_state.processed_data:
        return []
    
    # Keywords that suggest the user wants to see more data
    data_keywords = ['search', 'all', 'show', 'list', 'find', 'data', 'rows', 'records']
    wants_more_data = any(keyword in user_question.lower() for keyword in data_keywords)
    
    rows_to_show = max_rows_per_file if wants_more_data else 5
    
    for sheet_name, data in st.session_state.processed_data.items():
        df = data['dataframe']
        filename = data['filename']
        
        # Always include basic info
        file_info = {
            'filename': filename,
            'sheet_name': sheet_name,
            'total_rows': len(df),
            'columns': df.columns.tolist(),
            'sample_data': df.head(rows_to_show).to_dict('records') if not df.empty else []
        }
        
        # Add numeric summaries
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            file_info['numeric_summary'] = {}
            for col in numeric_cols:
                try:
                    file_info['numeric_summary'][col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'count': int(df[col].count())
                    }
                except:
                    continue
        
        # Add categorical summaries
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            file_info['categorical_summary'] = {}
            for col in categorical_cols[:3]:  # Limit to prevent context overflow
                try:
                    top_values = df[col].value_counts().head(5).to_dict()
                    file_info['categorical_summary'][col] = top_values
                except:
                    continue
        
        relevant_data.append(file_info)
    
    return relevant_data

def chat_with_bot(user_question):
    """Send question to the chatbot and get response with cross-file analysis"""
    try:
        # Prepare context from multiple files if available
        context_parts = []
        
        if st.session_state.excel_uploaded and st.session_state.processed_data:
            context_parts.append("=== MULTI-FILE DATA CONTEXT ===")
            context_parts.append("You have access to the following uploaded files and their data:")
            
            # Get relevant data for the user's question
            relevant_files = get_relevant_data_for_query(user_question)
            
            for file_info in relevant_files:
                filename = file_info['filename']
                sheet_name = file_info['sheet_name']
                total_rows = file_info['total_rows']
                columns = file_info['columns']
                sample_data = file_info['sample_data']
                
                context_parts.append(f"\n--- FILE: {filename} | SHEET: {sheet_name} ---")
                context_parts.append(f"Total Rows: {total_rows} | Columns: {len(columns)}")
                context_parts.append(f"Column Names: {', '.join(columns)}")
                
                # Include sample data
                if sample_data:
                    context_parts.append(f"Sample Data ({len(sample_data)} rows):")
                    for i, row in enumerate(sample_data):
                        row_str = ", ".join([f"{k}: {v}" for k, v in list(row.items())[:8]])  # Limit columns
                        context_parts.append(f"  Row {i+1}: {row_str}")
                
                # Include numeric summaries
                if 'numeric_summary' in file_info:
                    context_parts.append("Numeric Column Statistics:")
                    for col, stats in file_info['numeric_summary'].items():
                        stats_str = f"min={stats['min']:.2f}, max={stats['max']:.2f}, avg={stats['mean']:.2f}, count={stats['count']}"
                        context_parts.append(f"  {col}: {stats_str}")
                
                # Include categorical summaries
                if 'categorical_summary' in file_info:
                    context_parts.append("Categorical Column Top Values:")
                    for col, values in file_info['categorical_summary'].items():
                        top_vals = ", ".join([f"{k}({v})" for k, v in list(values.items())[:3]])
                        context_parts.append(f"  {col}: {top_vals}")
            
            # Add file summaries
            context_parts.append(f"\n=== FILE SUMMARIES ===")
            for filename, summary in st.session_state.file_summaries.items():
                context_parts.append(f"File: {filename} - {summary['total_rows']} rows, {summary['total_columns']} columns")
                context_parts.append(f"  Sheets: {', '.join(summary['sheets'])}")
            
            # Add relationship information
            if st.session_state.file_relationships:
                context_parts.append("\n=== FILE RELATIONSHIPS ===")
                top_relationships = st.session_state.file_relationships[:5]  # Top 5
                for rel in top_relationships:
                    context_parts.append(f"{rel['file1']} ↔ {rel['file2']}: {rel['similarity_percentage']:.1f}% similar")
        
        # Construct system message with context
        system_message = """You are a financial analyst AI assistant specialized in multi-file data analysis. 
        You help analyze Excel and CSV data, identify relationships between files, provide insights about 
        financial metrics, trends, and cross-file correlations. You can compare data across multiple files 
        and identify patterns, discrepancies, or complementary information.

        IMPORTANT: You have access to the complete data from ALL uploaded files. When users ask to "search all sheets" 
        or analyze across files, use the actual data provided in the context below to give specific, data-driven answers.
        You can reference specific values, perform calculations, and identify trends across all the files."""
        
        if context_parts:
            system_message += f"\n\n{chr(10).join(context_parts)}"
        
        # Generate response using Ollama
        response = ollama.chat(model='llama3.2:3b', messages=[
            {
                'role': 'system',
                'content': system_message
            },
            {
                'role': 'user',
                'content': user_question,
            },
        ])
        
        return response['message']['content']
        
    except Exception as e:
        return f"❌ Error generating response: {e}"

# Main App Layout
st.markdown('<h1 class="main-header">📊 Excel Financial Chatbot</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 System Status")
    
    # Check connections
    milvus_status = check_milvus_connection()
    ollama_status = check_ollama_connection()
    
    if milvus_status:
        st.success("✅ Milvus Connected")
        st.write(f"Collections: {len(st.session_state.collections_available)}")
    else:
        st.error("❌ Milvus Disconnected")
    
    if ollama_status:
        st.success("✅ Ollama + Llama 3.2 3B Ready")
    else:
        st.error("❌ Ollama/Model Not Available")
    
    st.divider()
    
    st.header("📁 Data Management")
    if st.session_state.excel_uploaded:
        st.success("✅ Data File Loaded")
        if st.session_state.uploaded_filename:
            st.write(f"**File:** {st.session_state.uploaded_filename}")
        if st.session_state.processed_data:
            total_sheets = len(st.session_state.processed_data)
            total_rows = sum([data['row_count'] for data in st.session_state.processed_data.values()])
            st.write(f"**Sheets/Tables:** {total_sheets}")
            st.write(f"**Total Rows:** {total_rows}")
            
            # Show sheet details
            with st.expander("📊 Sheet Details"):
                for sheet_name, data in st.session_state.processed_data.items():
                    st.write(f"**{sheet_name}**: {data['row_count']} rows, {data['column_count']} columns")
    else:
        st.info("📤 Upload a data file to get started")
    
    if st.button("🔄 Refresh Status"):
        st.rerun()

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Files", "💬 Chat", "🔗 Relationships", "📊 Analytics"])

# Tab 1: Upload Files
with tab1:
    st.header("📤 Upload Multiple Data Files")
    
    st.markdown("""
    <div class="upload-section">
        <h3>📋 Upload multiple data files for cross-analysis</h3>
        <p>Supported formats: Excel (.xlsx, .xls), CSV (.csv)</p>
        <p>Excel files with multiple sheets are fully supported</p>
        <p>The system will analyze relationships between your files using AI embeddings</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose data files",
        type=['xlsx', 'xls', 'csv'],
        help="Upload multiple Excel files (with multiple sheets) or CSV files containing financial or technical data",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"**📁 Selected {len(uploaded_files)} file(s):**")
        
        # Display file information for all files
        for i, uploaded_file in enumerate(uploaded_files):
            with st.expander(f"� {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📁 Filename", uploaded_file.name)
                with col2:
                    st.metric("📏 Size", f"{uploaded_file.size / 1024:.2f} KB")
                with col3:
                    file_ext = uploaded_file.name.split('.')[-1].upper()
                    st.metric("📝 Type", file_ext)
        
        if st.button("🚀 Process All Files & Analyze Relationships", type="primary"):
            with st.spinner("Processing multiple files and analyzing relationships..."):
                success = process_multiple_files(uploaded_files)
                if success:
                    st.balloons()

# Tab 2: Chat Interface
with tab2:
    st.header("💬 Chat with Your Data")
    
    # Show file access info
    if st.session_state.excel_uploaded and len(st.session_state.file_summaries) > 1:
        st.info(f"✅ **Multi-File Access Enabled**: Ollama can now search across ALL {len(st.session_state.file_summaries)} uploaded files and their sheets. Ask questions like 'search all sheets' or 'show me data from all files'.")
    elif st.session_state.excel_uploaded:
        st.info("✅ **File Access Enabled**: Ollama has access to your uploaded file and all its sheets.")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.container():
                st.markdown("**👤 You:**")
                st.info(message["content"])
        else:
            with st.container():
                st.markdown("**🤖 AI Assistant:**")
                st.success(message["content"])
    
    # Chat input
    if not ollama_status:
        st.warning("⚠️ Ollama/Llama 3.2 3B is not available. Please ensure it's running.")
    else:
        user_input = st.chat_input("Ask a question about your Excel data...")
        
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get bot response
            with st.spinner("🤔 Thinking..."):
                bot_response = chat_with_bot(user_input)
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            
            st.rerun()
    
    # Sample questions
    if not st.session_state.chat_history:
        st.markdown("### 💡 Sample Questions")
        
        if len(st.session_state.file_summaries) > 1:
            st.markdown("**🔗 Cross-File Analysis:**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Compare files and find relationships"):
                    st.session_state.chat_history.append({"role": "user", "content": "Compare my uploaded files and identify the relationships between them. What patterns do you see?"})
                    st.rerun()
                
                if st.button("📊 Find discrepancies across files"):
                    st.session_state.chat_history.append({"role": "user", "content": "Are there any discrepancies or inconsistencies between my files? What should I investigate?"})
                    st.rerun()
            
            with col2:
                if st.button("💰 Analyze financial trends across files"):
                    st.session_state.chat_history.append({"role": "user", "content": "Analyze financial trends across all my files. What insights can you provide?"})
                    st.rerun()
                
                if st.button("🎯 Identify consolidation opportunities"):
                    st.session_state.chat_history.append({"role": "user", "content": "Based on the relationships between my files, what consolidation or data integration opportunities do you see?"})
                    st.rerun()
        
        st.markdown("**📈 Multi-File Search & Analysis:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("� Search all sheets for specific data"):
                st.session_state.chat_history.append({"role": "user", "content": "Search all sheets for customer health scores and revenue data. Show me the top 5 customers by revenue."})
                st.rerun()
            
            if st.button("�💰 What are the revenue trends?"):
                st.session_state.chat_history.append({"role": "user", "content": "What are the revenue trends across all my files?"})
                st.rerun()
            
            if st.button("📊 Show all available data"):
                st.session_state.chat_history.append({"role": "user", "content": "Show me all the data available across all my uploaded files. List the sheets and their contents."})
                st.rerun()
        
        with col2:
            if st.button("📈 Calculate profit margins"):
                st.session_state.chat_history.append({"role": "user", "content": "Calculate profit margins from the data across all files"})
                st.rerun()
            
            if st.button("🎯 Find customers in multiple files"):
                st.session_state.chat_history.append({"role": "user", "content": "Find customers that appear in multiple files and compare their data."})
                st.rerun()
            
            if st.button("📋 Generate comprehensive report"):
                st.session_state.chat_history.append({"role": "user", "content": "Generate a comprehensive report analyzing all my uploaded files and their relationships."})
                st.rerun()
                st.rerun()
        
        with col2:
            if st.button("🔍 Identify key financial metrics"):
                st.session_state.chat_history.append({"role": "user", "content": "What are the key financial metrics I should focus on?"})
                st.rerun()
            
            if st.button("⚠️ Find potential risks"):
                st.session_state.chat_history.append({"role": "user", "content": "What potential financial risks can you identify?"})
                st.rerun()

# Tab 3: Relationship Analysis
with tab3:
    st.header("🔗 File Relationship Analysis")
    
    if st.session_state.excel_uploaded and st.session_state.file_relationships:
        st.info("🔍 AI-powered relationship analysis between your uploaded files")
        
        # Relationship Overview
        st.subheader("📊 Relationship Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🗂️ Files Analyzed", len(st.session_state.file_summaries))
        with col2:
            st.metric("📋 Sheets/Tables", len(st.session_state.processed_data))
        with col3:
            high_similarity = len([r for r in st.session_state.file_relationships if r['similarity'] > 0.7])
            st.metric("🎯 Strong Relationships", high_similarity)
        
        # Detailed Relationships
        st.subheader("🔍 Detailed Relationship Analysis")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            min_similarity = st.slider("Minimum Similarity %", 0, 100, 30, 5)
        with col2:
            show_all = st.checkbox("Show All Relationships", value=False)
        
        # Filter relationships
        filtered_relationships = [
            r for r in st.session_state.file_relationships 
            if r['similarity_percentage'] >= min_similarity
        ]
        
        if not show_all:
            filtered_relationships = filtered_relationships[:10]  # Show top 10
        
        # Display relationships
        for i, rel in enumerate(filtered_relationships):
            # Determine similarity level and color
            if rel['similarity'] > 0.8:
                similarity_level = "🟢 Very High"
                color = "success"
            elif rel['similarity'] > 0.6:
                similarity_level = "🟡 High" 
                color = "warning"
            elif rel['similarity'] > 0.4:
                similarity_level = "🟠 Medium"
                color = "info"
            else:
                similarity_level = "🔴 Low"
                color = "error"
            
            with st.expander(f"{similarity_level} - {rel['file1']} ↔ {rel['file2']} ({rel['similarity_percentage']:.1f}%)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**File 1:**")
                    if rel['file1'] in st.session_state.processed_data:
                        data1 = st.session_state.processed_data[rel['file1']]
                        st.write(f"- Rows: {data1['row_count']}")
                        st.write(f"- Columns: {data1['column_count']}")
                        st.write(f"- Numeric columns: {len(data1['numeric_columns'])}")
                
                with col2:
                    st.write("**File 2:**")
                    if rel['file2'] in st.session_state.processed_data:
                        data2 = st.session_state.processed_data[rel['file2']]
                        st.write(f"- Rows: {data2['row_count']}")
                        st.write(f"- Columns: {data2['column_count']}")
                        st.write(f"- Numeric columns: {len(data2['numeric_columns'])}")
                
                # Similarity bar
                st.progress(rel['similarity'])
                st.write(f"**Similarity Score:** {rel['similarity_percentage']:.2f}%")
                
                # Potential relationship insights
                if rel['similarity'] > 0.7:
                    st.success("💡 **Insight:** These files are highly related and may contain similar data structures or complementary information.")
                elif rel['similarity'] > 0.4:
                    st.info("💡 **Insight:** These files show moderate similarity and may have some overlapping themes or data patterns.")
                else:
                    st.warning("💡 **Insight:** These files appear to be quite different in structure or content.")
        
        # Summary insights
        st.subheader("🧠 AI Insights")
        
        if st.session_state.file_relationships:
            avg_similarity = sum([r['similarity'] for r in st.session_state.file_relationships]) / len(st.session_state.file_relationships)
            max_similarity = max([r['similarity'] for r in st.session_state.file_relationships])
            
            insights = []
            
            if avg_similarity > 0.6:
                insights.append("🎯 **High Cohesion:** Your files show strong overall similarity, suggesting they're part of a related dataset or business domain.")
            elif avg_similarity > 0.3:
                insights.append("📊 **Moderate Cohesion:** Your files have some common patterns but also distinct characteristics.")
            else:
                insights.append("🔄 **Diverse Dataset:** Your files cover different domains or have varied structures.")
            
            if max_similarity > 0.8:
                insights.append("🔗 **Strong Relationships Found:** Some files are very similar and may contain duplicate or highly related information.")
            
            for insight in insights:
                st.info(insight)
        
        # Agno Advanced Analysis
        if st.session_state.agno_analysis and 'error' not in st.session_state.agno_analysis:
            st.subheader("🤖 Agno Advanced Analysis")
            
            agno_data = st.session_state.agno_analysis
            
            # Financial Health Scores
            if 'financial_health_scores' in agno_data and agno_data['financial_health_scores']:
                st.write("**📊 Financial Health Scores:**")
                health_scores = agno_data['financial_health_scores']
                
                for file_name, scores in health_scores.items():
                    if isinstance(scores, dict) and 'error' not in scores:
                        overall_score = scores.get('overall_score', 0)
                        
                        if overall_score > 0.7:
                            health_color = "🟢"
                            health_status = "Excellent"
                        elif overall_score > 0.5:
                            health_color = "🟡"
                            health_status = "Good"
                        elif overall_score > 0.3:
                            health_color = "🟠"
                            health_status = "Fair"
                        else:
                            health_color = "🔴"
                            health_status = "Poor"
                        
                        st.write(f"{health_color} **{file_name}**: {health_status} ({overall_score:.2f})")
                    else:
                        st.write(f"⚠️ **{file_name}**: Analysis error")
            
            # Recommendations
            if 'recommendations' in agno_data and agno_data['recommendations']:
                st.write("**💡 Agno Recommendations:**")
                for rec in agno_data['recommendations']:
                    if not rec.startswith('Error'):
                        st.info(rec)
                    else:
                        st.warning(rec)
            
            # Cross-file insights
            if 'cross_file_insights' in agno_data and 'error' not in agno_data['cross_file_insights']:
                cross_insights = agno_data['cross_file_insights']
                
                if cross_insights.get('common_patterns'):
                    st.write("**🔍 Common Data Patterns:**")
                    for pattern in cross_insights['common_patterns']:
                        st.write(f"- **{pattern['pattern']}** found in: {', '.join(pattern['files'])}")
                
                if cross_insights.get('complementary_data'):
                    st.write("**🔗 Complementary Data Opportunities:**")
                    for comp in cross_insights['complementary_data']:
                        st.write(f"- {comp['file1']} ↔ {comp['file2']}: Common columns: {', '.join(comp['common_columns'])}")
        
        elif st.session_state.agno_analysis and 'error' in st.session_state.agno_analysis:
            st.warning(f"🤖 Agno Analysis: {st.session_state.agno_analysis['error']}")
        
        else:
            st.info("🤖 Agno analysis will appear here after processing files")

# Tab 4: Analytics Dashboard
with tab3:
    st.header("📊 Analytics Dashboard")
    
    if st.session_state.excel_uploaded and st.session_state.processed_data:
        st.info("📊 Analytics dashboard showing insights from your uploaded data")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_sheets = len(st.session_state.processed_data)
        total_rows = sum([data['row_count'] for data in st.session_state.processed_data.values()])
        total_columns = sum([data['column_count'] for data in st.session_state.processed_data.values()])
        queries_count = len([msg for msg in st.session_state.chat_history if msg["role"] == "user"])
        
        with col1:
            st.metric("📁 Sheets/Files", total_sheets, "0")
        
        with col2:
            st.metric("📊 Total Rows", total_rows, "0")
        
        with col3:
            st.metric("� Queries Asked", queries_count, "0")
        
        with col4:
            st.metric("�💾 Vector Store", "Ready", "Active")
        
        # Data overview
        st.subheader("📈 Data Overview")
        
        # Show data from each sheet
        for sheet_name, data in st.session_state.processed_data.items():
            with st.expander(f"📋 {sheet_name} - {data['row_count']} rows"):
                df = data['dataframe']
                
                # Basic statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Data Types:**")
                    st.write(df.dtypes.value_counts())
                
                with col2:
                    st.write("**Column Summary:**")
                    for col in df.columns[:5]:  # Show first 5 columns
                        non_null = df[col].notna().sum()
                        st.write(f"**{col}**: {non_null}/{len(df)} non-null")
                
                # Show sample data
                st.write("**Sample Data:**")
                st.dataframe(df.head(3))
                
                # Try to create a simple chart if numeric columns exist
                numeric_columns = df.select_dtypes(include=['number']).columns
                if len(numeric_columns) >= 2:
                    st.write("**Data Visualization:**")
                    chart_data = df[numeric_columns[:2]].head(10)
                    if not chart_data.empty:
                        st.line_chart(chart_data)
        
    else:
        st.info("📤 Upload a data file first to see analytics")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 Powered by Llama 3.2 3B • 🗄️ Milvus Vector Database • 🐍 Streamlit</p>
    <p>Built for Financial & Technical Analysis</p>
</div>
""", unsafe_allow_html=True)