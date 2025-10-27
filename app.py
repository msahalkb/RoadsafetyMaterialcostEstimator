import streamlit as st
import pandas as pd
import fitz # PyMuPDF
import os
import json
import re
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIGURATION ---

# 1. Configure Google API Key (for GENERATION)
# Make sure to set this in your terminal before running:
# $env:GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    print("Google API Key configured for generation.")
except KeyError:
    st.error("Error: GOOGLE_API_KEY environment variable not set. Please set it in your terminal.")
    st.stop()

# 2. Configure Local Embedding Model (for RETRIEVAL)
@st.cache_resource # Cache this so it only loads once
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
@st.cache_resource # Cache the database connection
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
@st.cache_data # Cache this so it only loads once
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

def get_gemini_response(prompt_text, as_json=True):
    """Function to call the Gemini API and get a response."""
    response = None
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt_text)

        if as_json:
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_response)
        else:
            return response.text
    except Exception as e:
        # Avoid referencing `response` if the error happened before it was assigned
        raw_text = getattr(response, "text", None)
        if raw_text:
            st.warning(f"Error parsing AI response. Raw text: {raw_text} (Error: {e})")
        else:
            st.warning(f"Error calling AI model: {e}")
        return None # Failed to get a valid JSON

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
        interventions = get_gemini_response(prompt_1, as_json=True)
    
    if not interventions:
        st.error("AI could not find any interventions in the report.")
        st.stop()
    
    st.success(f"Found {len(interventions)} interventions.")
    st.json(interventions)

    # 4. AI Call #2 (RAG): Get Bill of Materials
    st.header("Step 3: Generating Bill of Materials (BoM) from IRC Standards")
    all_boms = []
    
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
                continue

            # 4b. GENERATE BoM with AI using the rules
            # We also ask the AI to parse the quantity string into a number
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
            
            bom = get_gemini_response(prompt_2, as_json=True)
            
            if bom:
                st.success(f"Successfully generated BoM for {intervention_name}.")
                st.json(bom)
                all_boms.extend(bom) # Add this item's BoM to the master list
            else:
                st.error(f"AI failed to generate BoM for {intervention_name}.")

    # 5. Price Lookup and Final Calculation
    st.header("🏁 Step 4: Final Cost Estimate")
    
    if not all_boms:
        st.error("No Bill of Materials was generated. Cannot calculate cost.")
        st.stop()
        
    final_estimate_table = []
    total_cost = 0.0
    
    st.write("Finding prices for all materials in the 'clean_material_prices.csv'...")
    
    for item in all_boms:
        material_name = item.get('material', 'Unknown')
        quantity = item.get('quantity', 0.0)
        
        # Ensure quantity is a float
        try:
            quantity = float(quantity)
        except (ValueError, TypeError):
            quantity = 0.0
        
        # Search the price database
        try:
            # Find all rows where the description contains the material name
            match = price_db[price_db['description'].str.contains(material_name, case=False, na=False)]
            
            if not match.empty:
                # Get the first match
                found_item = match.iloc[0]
                unit_price = float(found_item['rate'])
                unit = found_item['unit']
                description = found_item['description']
                
                line_total = unit_price * quantity
                total_cost += line_total
                
                final_estimate_table.append({
                    "Material": material_name,
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
                    "Material": material_name,
                    "Quantity": quantity,
                    "Unit": item.get('unit'),
                    "Matched DSR Item": "--- PRICE NOT FOUND IN DSR ---",
                    "DSR Unit": "N/A",
                    "Unit Price (₹)": 0.0,
                    "Total (₹)": 0.0
                })
        except Exception as e:
            st.error(f"Error searching for '{material_name}': {e}")

    # Display the final table
    st.dataframe(final_estimate_table)
    
    st.header(f"✨ Grand Total Material Cost: ₹ {total_cost:,.2f}")
    st.balloons()