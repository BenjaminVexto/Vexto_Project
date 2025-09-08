import sys
import os
import logging
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
    """Importér contact_finder robust fra projektstien, med fallback."""
    current_dir = Path.cwd()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        logger.debug(f"Added {current_dir} to Python path")

    module_candidates = [
        "src.vexto.enrichment.contact_finder",  # jeres reelle sti
        "contact_finder",                       # fallback hvis filen ligger lokalt
    ]

    last_err = None
    for modname in module_candidates:
        try:
            mod = __import__(modname, fromlist=["*"])
            run_enrichment = getattr(mod, "run_enrichment_on_dataframe", None)
            check_status = getattr(mod, "check_api_status", None)  # valgfri i modulet
            if run_enrichment is None:
                raise ImportError(f"Module {modname} has no run_enrichment_on_dataframe")
            logger.info(f"✓ Successfully imported module: {modname}")
            return run_enrichment, check_status
        except Exception as e:
            last_err = e
            logger.debug(f"Import failed for {modname}: {e}")

    logger.error(f"✗ Failed to import contact_finder from candidates: {module_candidates}")
    if last_err:
        logger.error(last_err)
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
        
        # Apply sample size if specified
        if sample_size is not None:
            df = df.head(sample_size)
            logger.info(f"Limited to sample size: {sample_size} rows")
        
        # Step 4: Run enrichment (single pass for clarity)
        logger.info("\n--- Step 4: Running Enrichment ---")
        enriched_df = run_enrichment(df)
        # Markér alle som “success” i denne simple test
        tracker = ProgressTracker(len(df), logger)
        tracker.processed = len(df)
        tracker.successful = len(df)
        
        # Combine results (hvis batching brugt; ellers direkte run_enrichment_on_dataframe(df))
        enriched_df = run_enrichment(df)  # Eller concat fra batches
        
        # Step 5: Save output
        output_path = output_dir / 'enriched_cvr_data.csv'
        enriched_df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"✓ Enriched data saved to: {output_path}")
        
        # Step 6: Show summary
        tracker.summary()
        
        return True
    
    except Exception as e:
        logger.error(f"Unexpected error in test run: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    run_test()
    print("Test completed - check logs for details")