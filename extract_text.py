import fitz
from pathlib import Path
def extract_text_pdf(file_path):
    doc=fitz.open(file_path)
    all_text=""
    for page in doc:
        text=page.get_text("text")
        all_text+=text
    doc.close()
    return all_text
if __name__=="__main__":
    pdf_path=Path("C:/Users/LENOVO/Desktop/RoadSafetyEstimator/data/Road_Safety_Intervention_Report_Final.pdf")
    text=extract_text_pdf(pdf_path)
    print(f"Total characters extracted: {len(text)}")
    print("Preview of Extracted Text:",text[:10000])