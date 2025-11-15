import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import os
import json
import re
import time
import google.genai as genai  # <-- This is the correct, new import
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIGURATION ---

# 1. Configure Google API Key (for GENERATION)
# This client automatically finds the GOOGLE_API_KEY from your environmentSyAL_5
try:
    client = genai.Client()
    print("Google AI Client created successfully.")
except Exception as e:
    st.error(f"Error creating Google AI Client: {e}")
    st.error("Make sure your GOOGLE_API_KEY is set correctly in your terminal (e.g., $env:GOOGLE_API_KEY = '...')")
    st.stop()

# 2. Configure Local Embedding Model (for RETRIEVAL)
@st.cache_resource  # Cache this so it only loads once
def load_embeddings():
    print("Loading local embedding model...")
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}
    )
    print("Local model loaded.")
    return embeddings

# 3. Configure Vector Database (Your "Rules Library")
@st.cache_resource  # Cache the database connection
def load_vector_db(_embeddings):
    print("Connecting to vector database (chroma_db)...")
    db_persist_folder = "chroma_db"
    if not os.path.exists(db_persist_folder):
        st.error("Error: 'chroma_db' folder not found. Please run 'build_knowledge_base.py' first.")
        st.stop()
   
    db = Chroma(
        persist_directory=db_persist_folder,
        embedding_function=_embeddings
    )
    print("Connected to database.")
    return db

# 4. Configure Price Database (Your "Price List")
@st.cache_data  # Cache this so it only loads once
def load_price_db():
    price_db_path = "data_rates/clean_material_prices.csv"
    print(f"Loading price database from {price_db_path}...")
    try:
        return pd.read_csv(price_db_path)
    except FileNotFoundError:
        st.error(f"Error: Price database '{price_db_path}' not found.")
        st.error("Please run 'src/build_price_database.py' first.")
        return None

# --- HELPER FUNCTIONS ---

def extract_text_pdf(uploaded_file):
    """Extracts all text from an uploaded PDF file."""
    try:
        file_bytes = uploaded_file.getvalue()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        all_text = ""
        for page in doc:
            all_text += page.get_text("text")
        doc.close()
        return all_text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def get_gemini_response(client, prompt_text, as_json=True, max_retries=3, retry_delay=2):
    """Function to call the Gemini API and get a response with automatic retry logic.
    
    Args:
        client: The Gemini API client
        prompt_text: The prompt to send to the AI
        as_json: Whether to parse the response as JSON
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (exponential backoff)
    
    Returns:
        Parsed JSON or text response, or None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            # --- This is the new, correct way to call the API ---
            response = client.models.generate_content(
                model='gemini-2.0-flash',  # Or 'gemini-1.5-flash' if you prefer
                contents=prompt_text
            )
            # ---------------------------------------------------
           
            if as_json:
                cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
                return json.loads(cleaned_response)
            else:
                return response.text
        except Exception as e:
            attempt_num = attempt + 1
            if attempt_num < max_retries:
                # Calculate exponential backoff: 2s, 4s, 8s, etc.
                wait_time = retry_delay * (2 ** attempt)
                error_msg = str(e)
                
                if "exhausted" in error_msg.lower() or "rate" in error_msg.lower() or "quota" in error_msg.lower():
                    st.warning(f"⏳ AI rate limit hit (attempt {attempt_num}/{max_retries}). Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    st.warning(f"⚠️ Error in AI call (attempt {attempt_num}/{max_retries}): {error_msg}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            else:
                st.error(f"❌ AI call failed after {max_retries} attempts: {e}")
                return None
    
    return None

# --- MAIN APPLICATION ---
st.set_page_config(layout="wide")
st.title("🛣️ Road Safety Cost Estimator")

# Load persistent models and databases
embeddings = load_embeddings()
db = load_vector_db(embeddings)
price_db = load_price_db()

if price_db is None:
    st.stop()

# 1. User Uploads Report
st.header("Step 1: Upload Your Intervention Report")
uploaded_file = st.file_uploader("Upload your .pdf report", type="pdf", help="Upload the 'Road_Safety_Intervention_Report_Final.pdf' or a similar file.")

if uploaded_file is not None:
    # 2. Extract Text from PDF
    with st.spinner("Reading PDF..."):
        report_text = extract_text_pdf(uploaded_file)
        if not report_text:
            st.error("Failed to read PDF.")
            st.stop()
    st.success("PDF read successfully.")

    # 3. AI Call #1: Extract Interventions
    st.header("Step 2: Identifying Interventions")
    with st.spinner("Asking AI to find interventions in the report..."):
        prompt_1 = f"""
        You are an expert civil engineer. Read the following road safety report.
        Identify all recommended safety interventions and their quantities.
        Respond ONLY with a JSON list in the format:
        [
          {{"intervention": "Name of intervention", "quantity_str": "Quantity as string (e.g., '150 meters')"}}
        ]
       
        REPORT TEXT:
        ---
        {report_text[:10000]}
        ---
        """
        interventions = get_gemini_response(client, prompt_1, as_json=True)
   
    if not interventions:
        st.error("AI could not find any interventions in the report.")
        st.stop()
   
    st.success(f"Found {len(interventions)} interventions.")
    st.json(interventions)

    # 4. AI Call #2 (RAG): Get Bill of Materials
    st.header("Step 3: Generating Bill of Materials (BoM) from IRC Standards")
    all_boms = []
    failed_interventions = []
   
    for item in interventions:
        intervention_name = item['intervention']
        intervention_qty_str = item['quantity_str']
       
        with st.spinner(f"Generating BoM for: {intervention_name}..."):
            st.subheader(f"BoM for: {intervention_name} ({intervention_qty_str})")
           
            # 4a. RETRIEVE relevant rules from your database
            try:
                relevant_rules = db.similarity_search(intervention_name, k=5)
                context_rules = "\n---\n".join([doc.page_content for doc in relevant_rules])
                st.info(f"Found {len(relevant_rules)} relevant rules from your IRC standards.")
            except Exception as e:
                st.error(f"Failed to query ChromaDB: {e}")
                failed_interventions.append(intervention_name)
                continue

            # 4b. GENERATE BoM with AI using the rules (with retry logic)
            prompt_2 = f"""
            You are a cost estimator. Using ONLY the 'Relevant IRC Standards' provided,
            create a material-only Bill of Materials (BoM) for:
           
            Intervention: {intervention_name}
            Quantity: {intervention_qty_str}

            First, parse the quantity string to get a number.
            Then, list all materials needed. Calculate the required quantity for each material.
            Respond ONLY with a JSON list. Ignore non-material costs (labor, taxes).
            Format:
            [
              {{"material": "Material name", "spec_clause": "Source (e.g., IRC:SP:87, Cl 4.2)", "quantity": 150.0, "unit": "meters"}}
            ]

            Relevant IRC Standards:
            ---
            {context_rules}
            ---
            """
           
            # Call with retry logic (default: 3 attempts with exponential backoff)
            bom = get_gemini_response(client, prompt_2, as_json=True, max_retries=3, retry_delay=2)
           
            if bom:
                st.success(f"✅ Successfully generated BoM for {intervention_name}.")
                st.json(bom)
                all_boms.extend(bom) # Add this item's BoM to the master list
            else:
                st.error(f"❌ AI failed to generate BoM for {intervention_name} after all retries.")
                failed_interventions.append(intervention_name)

    # 5. Price Lookup and Final Calculation (FINAL CORRECTED VERSION)
    st.header("🏁 Step 4: Final Cost Estimate")

    if not all_boms:
        st.error("No Bill of Materials was generated. Cannot calculate cost.")
        st.stop()
       
    final_estimate_table = []
    total_cost = 0.0

    st.write("Finding prices for all materials using best-match keyword search...")

    for item in all_boms:
        material_name = item.get('material', 'Unknown')
        quantity = item.get('quantity', 0.0)
       
        try:
            quantity = float(quantity)
        except (ValueError, TypeError):
            quantity = 0.0

        keywords = re.split(r'[\s\(\),-]+', material_name)
        valid_keywords = [k.lower() for k in keywords if len(k) > 2 and k.lower() not in ['and', 'for', 'with', 'the', 'unit']]
       
        # --- FIX: Create a *copy* of the price_db to work with ---
        working_price_db = price_db.copy()
       
        if not valid_keywords:
            best_match_row = None
        else:
            # 2. Create "match_count" on the copy
            working_price_db['match_count'] = 0
           
            for keyword in valid_keywords:
                matches = working_price_db['description'].str.contains(keyword, case=False, na=False)
               
                # --- FIX: Use the correct 'match_count' (singular) name ---
                working_price_db.loc[matches, 'match_count'] += 1

            # 4. Find the row with the highest score
            max_score = working_price_db['match_count'].max()
           
            if max_score > 0:
                best_match_row = working_price_db.loc[working_price_db['match_count'].idxmax()].to_dict()
            else:
                best_match_row = None
       
        # --- End of Fixes ---

        if best_match_row is not None:
            # We found a match!
            unit_price = float(best_match_row['rate'])
            unit = best_match_row['unit']
            description = best_match_row['description']
           
            line_total = unit_price * quantity
            total_cost += line_total
           
            final_estimate_table.append({
                "Material (from AI)": material_name,
                "Quantity": quantity,
                "Unit": item.get('unit'),
                "Matched DSR Item": description,
                "DSR Unit": unit,
                "Unit Price (₹)": unit_price,
                "Total (₹)": line_total
            })
        else:
            # No price found
            final_estimate_table.append({
                "Material (from AI)": material_name,
                "Quantity": quantity,
                "Unit": item.get('unit'),
                "Matched DSR Item": "--- PRICE NOT FOUND ---",
                "DSR Unit": "N/A",
                "Unit Price (₹)": 0.0,
                "Total (₹)": 0.0
            })

    # Display the final table
    st.dataframe(final_estimate_table)

    st.header(f"✨ Grand Total Material Cost: ₹ {total_cost:,.2f}")
    st.balloons()