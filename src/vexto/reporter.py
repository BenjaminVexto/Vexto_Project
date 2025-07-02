# src/vexto/reporter.py

import pandas as pd
from pathlib import Path
from datetime import datetime
from openpyxl.utils import get_column_letter
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule

def create_excel_report(df_nested: pd.DataFrame):
    """
    Genererer en komplet og poleret Excel-rapport (v2.1) med alle nye KPI'er,
    flere faner, avanceret formatering og timestamp.
    """
    # --- TRIN 0: FORBEREDELSE & FILNAVN ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    path = f"output/Vexto_Rapport_{timestamp}.xlsx"
    Path("output").mkdir(exist_ok=True)

    # --- TRIN 1: DATA FORBEREDELSE (Udpakning af nestede kolonner) ---
    print("Forbereder data og pakker nestede kolonner ud...")
    
    # Liste over alle de datakategorier, din analyzer producerer
    nested_columns = [
        'basic_seo', 'technical_seo', 'performance', 'authority', 
        'security', 'privacy', 'conversion', 'social_and_reputation'
    ]
    
    # Start med URL som base
    df_flat = pd.DataFrame(df_nested['url'])

    for col_name in nested_columns:
        if col_name in df_nested.columns and df_nested[col_name].notna().any():
            # Håndter at nogle rækker kan have None for en hel kategori
            valid_rows = df_nested[df_nested[col_name].apply(isinstance, args=(dict,))]
            if not valid_rows.empty:
                # Normaliser den nestede ordbog til flade kolonner med prefix
                normalized_data = pd.json_normalize(valid_rows[col_name]).add_prefix(f'{col_name}_')
                # Sæt indekset til at matche for en korrekt join
                normalized_data.index = valid_rows.index
                df_flat = df_flat.join(normalized_data)
    
    print("Data er færdigbehandlet.")

    # --- TRIN 2: OPRET FANER ---
    with pd.ExcelWriter(path, engine="openpyxl") as xls:
        # Redesignet definition af faner, der bruger de nye, flade kolonnenavne
        sheet_definitions = {
            "Overview": {
                "URL": "url",
                "Titel": "basic_seo_title_text",
                "Performance Score": "performance_performance_score",
                "Authority Rank": "authority_global_rank",
                "Døde Links": "technical_seo_broken_links_count",
                "Sikkerheds-tjek": "security_hsts_enabled"
            },
            "Technical SEO": {
                "URL": "url",
                "HTTPS": "technical_seo_is_https",
                "Robots.txt Fundet": "technical_seo_robots_txt_found",
                "Sitemap Fundet": "technical_seo_sitemap_xml_found",
                "Døde Links Antal": "technical_seo_broken_links_count",
                "Døde Links Liste": "technical_seo_broken_links_list"
            },
            "Performance": {
                "URL": "url",
                "PSI Score": "performance_performance_score",
                "LCP (ms)": "performance_lcp_ms",
                "Total JS (KB)": "performance_total_js_size_kb",
                "Gns. Billedstr. (KB)": "basic_seo_avg_image_size_kb"
            },
            "Conversion & Privacy": {
                "URL": "url",
                "GA4 Fundet": "conversion_has_ga4",
                "Meta Pixel Fundet": "conversion_has_meta_pixel",
                "Cookie Banner": "privacy_cookie_banner_detected",
                "Fundne E-mails": "conversion_emails_found",
                "Fundne Tlf.nr.": "conversion_phone_numbers_found",
                "Trust Signals": "conversion_trust_signals_found"
            }
        }

        # Opret hver fane
        for sheet_name, cols_map in sheet_definitions.items():
            existing_cols = {display_name: actual_name for display_name, actual_name in cols_map.items() if actual_name in df_flat.columns}
            if existing_cols:
                sheet_df = df_flat[list(existing_cols.values())].copy()
                sheet_df.columns = list(existing_cols.keys())
                sheet_df.to_excel(xls, sheet_name=sheet_name, index=False)

        # Gem en fane med alle udpakkede data for fuld gennemsigtighed
        df_flat.to_excel(xls, "Alle Data", index=False)

        # --- TRIN 3: VISUEL POLISH ---
        wb = xls.book
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill("solid", fgColor="003366")
        red_fill = PatternFill("solid", fgColor="FFC7CE")
        green_fill = PatternFill("solid", fgColor="C6EFCE")

        for ws in wb.worksheets:
            if ws.max_row == 0: continue
            
            # Style header
            for cell in ws[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(vertical="center", horizontal="center")

            # Autofit kolonnebredder
            for col_idx, column in enumerate(ws.columns, 1):
                max_length = max(len(str(cell.value or "")) for cell in column)
                max_length = max(max_length, len(ws.cell(row=1, column=col_idx).value or ""))
                ws.column_dimensions[get_column_letter(col_idx)].width = max_length + 4

        # Specifik conditional formatting
        if "Overview" in wb.sheetnames:
            ws = wb["Overview"]
            ws.conditional_formatting.add(f"C2:C{ws.max_row}", ColorScaleRule(start_type='num', start_value=0, start_color='FFC7CE', mid_type='num', mid_value=70, mid_color='FFEB9C', end_type='num', end_value=100, end_color='C6EFCE'))
            ws.conditional_formatting.add(f"E2:E{ws.max_row}", CellIsRule(operator="greaterThan", formula=["0"], fill=red_fill))
            ws.conditional_formatting.add(f"F2:F{ws.max_row}", CellIsRule(operator="equal", formula=["TRUE"], fill=green_fill))

    print(f"FÆRDIG: Rapport 2.1 er genereret og gemt som '{path}'")