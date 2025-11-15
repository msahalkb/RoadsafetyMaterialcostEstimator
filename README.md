🛣️ AI-Powered Road Safety Cost Estimator

This project is an AI-powered tool that automatically generates a material-only cost estimate for road safety interventions, built for the IIT Madras Hackathon 2025.

It works by reading a PDF intervention report (provided by the user), identifying the required interventions, and then cross-referencing this with a "Knowledge Base" built from:

    IRC Standards (The Rules): A vector database built from official Indian Roads Congress (IRC) PDFs to determine the Bill of Materials (BoM).
    CPWD Rates (The Prices): A clean price list built from the official CPWD Delhi Schedule of Rates (DSR) Excel file.

This application uses a RAG (Retrieval-Augmented Generation) pipeline with Google's Gemini AI to parse, query, and generate the final estimate.

🚀 How to Run This Project

Follow these steps exactly to set up and run the application.
Step 1: Set Up the Environment

    Create a Virtual Environment:

    python -m venv appenv

    Activate the Environment:
        On Windows (PowerShell): .\appenv\Scripts\Activate.ps1
        On macOS/Linux: source appenv/bin/activate
    Install All Required Libraries:
        pip install -r requirements.txt

Step 2: Set the Google API Key

This project requires a Google Gemini API key to function.

    Get Your API Key: Go to Google AI Studio.
    Set the Environment Variable: In your terminal, set the key.
        On Windows (PowerShell):

        $env:GOOGLE_API_KEY = "YOUR_API_KEY_HERE"

        On macOS/Linux:

        export GOOGLE_API_KEY="YOUR_API_KEY_HERE"

Step 3: Build the Knowledge Base (One-Time Setup)

You must run these two scripts one time before launching the app.

    Build the "Rules Library":
        This script reads all your PDFs from the src/irc_standards folder and builds the chroma_db vector database.
        Run:

        python src/build_knowledge_base.py

        (This will take a few minutes the first time it downloads the local embedding model).
    Build the "Price List":
        This script reads your DSR2023.xlsx file and creates the clean clean_material_prices.csv file.
        Run:

        python src/build_price_database.py

Step 4: Launch the Application

You are now ready to run the app.

   streamlit run src/app.py

Your web browser will automatically open to the application, and you can upload your PDF report.
