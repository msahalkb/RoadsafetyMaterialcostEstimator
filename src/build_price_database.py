import pandas as pd
import re
import os
SOURCE_EXCEL_PATH="data_rates/DSR2023.xlsx"
SHEET_NAME_TO_READ='Schedule'
CLEAN_CSV_PATH='data_rates/clean_material_prices.csv'
DESCRIPTION_COLUMN=7  # Description is in column H (index 7)
RATE_COLUMN=6        # Rate is in column G (index 6)
UNIT_COLUMN=5        # Unit is in column F (index 5)
def clean_excel_data(file_path):
    print(f"Loading Excel file from '{file_path}'...")
    if not os.path.exists(file_path):
        print(f"--- ERROR ---")
        print(f"File not found at: {file_path}")
        print(f"Please make sure your Excel file is in the 'data_rates' folder")
        print(f"and the name matches 'SOURCE_EXCEL_PATH' in this script.")
        return
    try:
        print(f"Attempting to read sheet '{SHEET_NAME_TO_READ}'...")
        # Skip the first 3 rows which appear to be headers
        df = pd.read_excel(file_path, sheet_name=SHEET_NAME_TO_READ, header=None, skiprows=3)
        print("Sheet loaded successfully.")
        
        # Print first few rows to verify structure
        print("\nFirst 5 rows of data:")
        print(df.head())
    except ValueError as e:
        print(f"--- ERROR ---")
        print(f"Could not find sheet named '{SHEET_NAME_TO_READ}' in the Excel file.")
        print(f"Check your Excel file.Is the tab name exactly'Schedule'?")
        print(f"Error details: {e}")
        return
    all_items=[]
    print("Processing rows...")
    for index,row in df.iterrows():
        try:
            description=str(row.get(DESCRIPTION_COLUMN,"")).strip()
            unit=str(row.get(UNIT_COLUMN,"")).strip()
            rate_val=row.get(RATE_COLUMN)
            
            # Debug print for problematic rows
            if index < 5:  # Print first 5 rows for debugging
                print(f"\nRow {index}:")
                print(f"Description: '{description}'")
                print(f"Unit: '{unit}'")
                print(f"Rate: {rate_val}")
            
            # Convert rate to float only if it's not empty
            if pd.notna(rate_val):
                rate_val = float(rate_val)
            else:
                continue
                
            if rate_val>0 and description and description != "nan" and len(description)>5:
                all_items.append({"description":description.replace('\n', ' '),"unit":unit,"rate":rate_val})
        except (ValueError, TypeError,Exception) as e:
            if index < 5:  # Print errors for first 5 rows
                print(f"Error in row {index}: {str(e)}")
    if not all_items:
        print("--- ERROR ---")
        print("No items were extracted.")
        print("Did you set the correct column numbers at the top of the script?")
    print(f"Successfully extracted {len(all_items)} items with prices")
    clean_df=pd.DataFrame(all_items)
    clean_df.to_csv(CLEAN_CSV_PATH, index=False)
    print("--- SUCCESS ---")
    print(f"Cleaned data saved to '{CLEAN_CSV_PATH}'")
if __name__=="__main__":
    clean_excel_data(SOURCE_EXCEL_PATH)  