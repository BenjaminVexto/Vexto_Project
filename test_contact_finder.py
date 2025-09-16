import sys
import os
import logging
import re
from datetime import datetime
from pathlib import Path
import traceback
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor


# --- Setup Logging with File Output -----------------------------------
def setup_logging(log_dir="logs"):
    """Setup comprehensive logging with both console and file output (robust)."""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"contact_finder_test_{timestamp}.log"

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Fjern eksisterende handlers for at undgå duplikering/inkonsistens
    for h in list(root.handlers):
        root.removeHandler(h)

    # File handler (DEBUG+)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Console handler (INFO+)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger, log_file


# --- Import Checker ---------------------------------------------------
def check_and_import_module(logger):
    """Importér contact_finder robust – og lav fallback-runner hvis shims mangler."""
    current_dir = Path.cwd()
    # Sørg for at både projektroden og ./src er på sys.path
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        logger.debug(f"Added {current_dir} to Python path")
    src_dir = current_dir / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
        logger.debug(f"Added {src_dir} to Python path")

    module_candidates = [
        "src.vexto.enrichment.contact_finder",
        "vexto.enrichment.contact_finder",
        "contact_finder",
    ]

    errors = {}
    for modname in module_candidates:
        try:
            mod = __import__(modname, fromlist=["*"])
            logger.info(f"✓ Imported module: {modname}")

            run_enrichment = getattr(mod, "run_enrichment_on_dataframe", None)
            check_status = getattr(mod, "check_api_status", None)

            # Hvis shims mangler: lav en simpel fallback-runner baseret på ContactFinder
            if run_enrichment is None:
                cf_cls = getattr(mod, "ContactFinder", None)
                if cf_cls is None:
                    raise ImportError(f"{modname} mangler både run_enrichment_on_dataframe og ContactFinder")
                logger.info("ℹ run_enrichment_on_dataframe mangler – opretter fallback-runner fra ContactFinder")

                def _fallback_run_enrichment_on_dataframe(
                    df,
                    url_col=None,
                    *, limit_pages=4, top_n=1, gdpr_minimize=False, use_browser="auto"
                ):
                    import json
                    import pandas as _pd
                    # Gæt URL-kolonne
                    candidates = ["url", "website", "web", "domain", "hjemmeside", "site", "company_url"]
                    if url_col is None:
                        url_col = next((c for c in candidates if c in df.columns), None)
                    if not url_col:
                        raise ValueError(f"Kunne ikke finde URL-kolonne. Prøv en af: {candidates}")

                    cf = cf_cls(gdpr_minimize=gdpr_minimize, use_browser=use_browser)
                    best_names, best_titles, best_emails, best_phones, best_scores, best_urls, all_json = \
                        [], [], [], [], [], [], []

                    for _, row in df.iterrows():
                        url = (row.get(url_col) or "").strip()
                        if not url:
                            best_names.append(None); best_titles.append(None); best_emails.append([])
                            best_phones.append([]); best_scores.append(None); best_urls.append(None); all_json.append("[]")
                            continue
                        if top_n and top_n > 1:
                            results = cf.find_all(url, limit_pages=limit_pages)[:top_n]
                        else:
                            b = cf.find(url, limit_pages=limit_pages)
                            results = [b] if b else []
                        if results:
                            r0 = results[0]
                            best_names.append(r0.get("name"))
                            best_titles.append(r0.get("title"))
                            best_emails.append(r0.get("emails") or [])
                            best_phones.append(r0.get("phones") or [])
                            best_scores.append(r0.get("score"))
                            best_urls.append(r0.get("url"))
                            all_json.append(json.dumps(results, ensure_ascii=False))
                        else:
                            best_names.append(None); best_titles.append(None); best_emails.append([])
                            best_phones.append([]); best_scores.append(None); best_urls.append(url); all_json.append("[]")

                    out = df.copy()
                    out["cf_best_name"] = best_names
                    out["cf_best_title"] = best_titles
                    out["cf_best_emails"] = best_emails
                    out["cf_best_phones"] = best_phones
                    out["cf_best_score"] = best_scores
                    out["cf_best_url"] = best_urls
                    out["cf_all"] = all_json
                    return out

                run_enrichment = _fallback_run_enrichment_on_dataframe

            # Hvis check_status ikke findes, lav en simpel placeholder
            if check_status is None:
                def check_status():
                    logger.info("check_api_status ikke defineret i modulet – bruger no-op.")
                    return {"module_loaded": True}

            logger.info(f"✓ Ready to run enrichment via: {('native' if getattr(mod, 'run_enrichment_on_dataframe', None) else 'fallback')}")
            return run_enrichment, check_status

        except Exception as e:
            errors[modname] = e
            logger.debug(f"Import failed for {modname}: {e}")

    # Hvis vi rammer her: alt fejlede
    logger.error(f"✗ Failed to import contact_finder from candidates: {list(errors.keys())}")
    for k, v in errors.items():
        logger.error(f"{k}: {v}")
    logger.error(traceback.format_exc())
    return None, None

# --- Data Validation --------------------------------------------------
def validate_csv_file(csv_path, logger):
    """Validate and load CSV file with error handling."""
    csv_path = Path(csv_path)
    
    # Check if file exists
    if not csv_path.exists():
        logger.error(f"✗ CSV file not found: {csv_path}")
        logger.info("Please check the file path and try again")
        return None
    
    logger.info(f"✓ Found CSV file: {csv_path}")
    logger.info(f" File size: {csv_path.stat().st_size / 1024:.1f} KB")
    
    # Try to read CSV with different encodings
    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            logger.info(f"✓ Successfully loaded CSV with {encoding} encoding")
            logger.info(f" Rows: {len(df)}, Columns: {len(df.columns)}")
            
            # Show column names
            logger.debug(f"Columns found: {list(df.columns)}")
            
            # Add missing columns
            if 'scraped_contact_text' not in df.columns:    
                df['scraped_contact_text'] = ''
                logger.info("Added missing 'scraped_contact_text' column")
            
            # Show sample data
            logger.debug("Sample data (first 3 rows):")
            logger.debug(f"\n{df.head(3).to_string()}")
            return df
        except UnicodeDecodeError:
            logger.debug(f"Failed with {encoding} encoding, trying next...")
            continue
        except Exception as e:
            logger.error(f"Error reading CSV with {encoding}: {e}")
            continue
    
    logger.error("✗ Could not read CSV file with any encoding")
    return None

# --- Progress Tracker -------------------------------------------------
class ProgressTracker:
    """Track and display enrichment progress."""
    def __init__(self, total_items, logger):
        self.total = total_items
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.start_time = datetime.now()
        self.logger = logger
    
    def update(self, success=True):
        self.processed += 1
        if success:
            self.successful += 1
        else:
            self.failed += 1
        
        # Calculate progress
        progress_pct = (self.processed / self.total) * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # Estimate time remaining
        if self.processed > 0:
            rate = elapsed / self.processed
            remaining = rate * (self.total - self.processed)
            eta = f"{remaining:.0f}s"
        else:
            eta = "calculating..."
        
        # Log progress every 10 items or 10%
        step = max(1, self.total // 10)
        if self.processed % step == 0 or self.processed == self.total:
            self.logger.info(
                f"Progress: {self.processed}/{self.total} ({progress_pct:.1f}%) "
                f"| Success: {self.successful} | Failed: {self.failed} "
                f"| ETA: {eta}"
            )
    
    def summary(self):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        success_rate = (self.successful / self.total * 100) if self.total > 0 else 0
        self.logger.info("=" * 60)
        self.logger.info("ENRICHMENT SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total processed: {self.processed}")
        self.logger.info(f"Successful: {self.successful} ({success_rate:.1f}%)")
        self.logger.info(f"Failed: {self.failed}")
        self.logger.info(f"Time taken: {elapsed:.1f} seconds")
        avg_str = f"{(elapsed/self.total):.2f} seconds" if self.total > 0 else "N/A"
        self.logger.info(f"Average time per item: {avg_str}")
        self.logger.info("=" * 60)

# --- Main Test Function -----------------------------------------------
def run_test(csv_path=None, output_dir=None, sample_size=None):
    """Main test function with comprehensive error handling.
    
    Args:
        csv_path: Path to input CSV file
        output_dir: Directory for output files
        sample_size: Number of rows to process (None for all)
    """
    # Setup logging
    logger, log_file = setup_logging()
    logger.info("=" * 60)
    logger.info("VEXTO CONTACT FINDER - TEST RUN")
    logger.info("=" * 60)
    
    # Default paths
    if csv_path is None:
        csv_path = r'C:\Users\benja\Desktop\Vexto_Project\output\cvr_data.csv'
    if output_dir is None:
        output_dir = Path(r'C:\Users\benja\Desktop\Vexto_Project\output')
    else:
        output_dir = Path(output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Import module
        logger.info("\n--- Step 1: Importing Contact Finder Module ---")
        run_enrichment, check_status = check_and_import_module(logger)
        if not run_enrichment:
            logger.error("Cannot proceed without contact_finder module")
            return False
        
        # Step 2: Check API status (hvis function eksisterer i module)
        if check_status:
            logger.info("\n--- Step 2: Checking API Status ---")
            check_status()  # Kald fra module
        
        # Step 3: Load and validate CSV
        logger.info("\n--- Step 3: Loading and Validating CSV ---")
        df = validate_csv_file(csv_path, logger)
        if df is None:
            logger.error("Cannot proceed without valid CSV data")
            return False

        # >>> ANKER START: PREFILTER_URLS
        # Pre-filter ugyldige/blanke URL’er og normalisér til https://
        def _norm_url_value(u):
            if not isinstance(u, str):
                return None
            u = u.strip()
            if not u or u.lower() in ("nan", "none", "null"):
                return None
            if not u.startswith(("http://", "https://")):
                # Accepter domæne/host (evt. med www.) og tilføj https://
                if re.match(r"^(www\.)?([a-z0-9\-]+\.)+[a-z]{2,}$", u, re.I):
                    return "https://" + u
                return None
            return u

        # Find sandsynlig URL-kolonne (matcher senere brug)
        _candidates = {
            "url","website","web","domain","hjemmeside","site","company_url","homepage","website_url","www"
        }
        _colmap = {c.lower(): c for c in df.columns}
        _url_col = next(( _colmap[c] for c in _candidates if c in _colmap ), None)

        if _url_col is not None:
            _before = len(df)
            df[_url_col] = df[_url_col].map(_norm_url_value)
            df = df.dropna(subset=[_url_col]).reset_index(drop=True)
            _removed = _before - len(df)
            logger.info(f"Pre-filtered rows on '{_url_col}': {_before} -> {len(df)} (removed {_removed})")
        else:
            logger.warning("No URL-like column found for pre-filtering (looked for: "
                           "url, website, web, domain, hjemmeside, site, company_url, homepage, website_url, www)")
        # <<< ANKER SLUT: PREFILTER_URLS
        
        # Apply sample size if specified
        if sample_size is not None:
            df = df.head(sample_size)
            logger.info(f"Limited to sample size: {sample_size} rows")
        
        # Apply sample size if specified
        if sample_size is not None:
            df = df.head(sample_size)
            logger.info(f"Limited to sample size: {sample_size} rows")
        
        # Step 4: Running Enrichment
        logger.info("\n--- Step 4: Running Enrichment ---")
        # >>> ANKER START: PARALLEL_AND_METRICS
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        import json
        from src.vexto.enrichment.contact_finder import ContactFinder  # Direkte import for at få adgang til cf

        # Gæt URL-kolonne
        url_hint = next((c for c in df.columns if c.lower() in {
            "url","website","web","domain","hjemmeside","site","company_url","homepage","website_url","www"
        }), None)
        if not url_hint:
            logger.error("✗ Ingen URL-kolonne fundet. Prøv en af: url, website, hjemmeside, osv.")
            return False

        start_run = time.time()
        results = []
        errors = 0
        tracker = ProgressTracker(len(df), logger)

        def _one(idx, row):
            try:
                url = str(row[url_hint]).strip()
                if not url:
                    return (idx, url, [], "No URL provided")
                cf = ContactFinder(timeout=10.0, use_browser="auto")  # Opret ny instans pr. række for trådsikkerhed
                out = cf.find_all(url, limit_pages=4, directors=row.get("directors", []))
                return (idx, url, out, None)
            except Exception as e:
                return (idx, row.get(url_hint, ""), [], str(e))

        max_workers = 6
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_one, i, r) for i, r in df.iterrows()]
            for fut in as_completed(futs):
                idx, url, out, err = fut.result()
                if err:
                    errors += 1
                    logger.warning(f"[PAR] Fejl på {url}: {err}")
                    tracker.update(success=False)
                else:
                    tracker.update(success=bool(out))
                results.append((idx, url, out))

        results.sort(key=lambda x: x[0])
        
        # Map resultater til DataFrame
        best_names, best_titles, best_emails, best_phones, best_scores, best_urls, all_json = [], [], [], [], [], [], []
        for _, url, out in results:
            if out:
                r0 = out[0]
                best_names.append(r0.get("name"))
                best_titles.append(r0.get("title"))
                best_emails.append(r0.get("emails") or [])
                best_phones.append(r0.get("phones") or [])
                best_scores.append(r0.get("score"))
                best_urls.append(r0.get("url") or url)
                all_json.append(json.dumps(out, ensure_ascii=False))
            else:
                best_names.append(None)
                best_titles.append(None)
                best_emails.append([])
                best_phones.append([])
                best_scores.append(None)
                best_urls.append(url)
                all_json.append("[]")

        enriched_df = df.copy()
        enriched_df["cf_best_name"] = best_names
        enriched_df["cf_best_title"] = best_titles
        enriched_df["cf_best_emails"] = best_emails
        enriched_df["cf_best_phones"] = best_phones
        enriched_df["cf_best_score"] = best_scores
        enriched_df["cf_best_url"] = best_urls
        enriched_df["cf_all"] = all_json

        # Log metrics
        duration = time.time() - start_run
        hits = sum(1 for _, _, out in results if out)
        hit_rate = (hits / max(1, len(results))) * 100.0
        logger.info(f"Run summary: hits={hits}/{len(results)} ({hit_rate:.1f}%), duration={duration:.1f}s, errors={errors}")
        # <<< ANKER SLUT: PARALLEL_AND_METRICS
        
        # Step 5: Save output
        output_path = output_dir / 'enriched_cvr_data.csv'
        enriched_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"✓ Enriched data saved to: {output_path}")
        
        # Step 6: Show summary
        # >>> ANKER START: EXTRA_SUMMARY
        try:
            tracker.summary()
            logger.info("\n=== EXTENDED METRICS SUMMARY ===")
            total = len(results)
            with_contacts = sum(1 for _, _, out in results if out)
            hit_rate = (with_contacts / max(1, total)) * 100.0
            logger.info(f"Total rows processed: {total}")
            logger.info(f"Rows with contacts: {with_contacts} ({hit_rate:.1f}%)")
            logger.info(f"Errors: {errors}")
            logger.info(f"Total duration: {duration:.1f}s")
            avg_time = duration / max(1, total)
            logger.info(f"Average time per row: {avg_time:.2f}s")
            logger.info("==============================")
        except Exception as e:
            logger.error(f"Fejl ved generering af summary: {e}")
        # <<< ANKER SLUT: EXTRA_SUMMARY
        
        return True
                    
    except Exception as e:
        logger.error(f"Unexpected error in test run: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    run_test()
    print("Test completed - check logs for details")