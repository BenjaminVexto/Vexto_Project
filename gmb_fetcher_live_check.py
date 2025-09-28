# gmb_fetcher_live_check.py
import os
import sys
import json
import asyncio
import argparse
import logging
import importlib
from contextlib import asynccontextmanager

# --- UTF-8 stdout på Windows ---
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# -------- .env support --------
def load_env(env_file: str | None):
    try:
        from dotenv import load_dotenv, find_dotenv
    except Exception:
        return None
    path = env_file if env_file and os.path.isfile(env_file) else None
    if not path:
        path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(dotenv_path=path, override=False)
        return path
    return None

def get_api_key():
    for key in ("GOOGLE_API_KEY", "GOOGLE_MAPS_API_KEY", "MAPS_API_KEY"):
        v = os.getenv(key)
        if v:
            return v
    return None

# -------- argumenter & logging --------
def setup_logging(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    p = argparse.ArgumentParser(description="Live-test af vexto/scoring gmb_fetcher mod Google Places API")
    p.add_argument("--url", help="Fx https://www.vexto.dk")
    p.add_argument("--name", help="Valgfrit firmanavn (fx 'Vexto')")
    p.add_argument("--phone", help="Valgfrit telefonnummer (fx +4560123456)")
    p.add_argument("--file", help="Fil med linjer: url[,name[,phone]]")
    p.add_argument("--env-file", help="Sti til .env (valgfri) – ellers auto-findes")
    p.add_argument("--http-path", default=r"C:\Users\benja\Desktop\Vexto_Project\src\vexto\scoring\http_client.py",
                   help="Sti til http_client.py")
    p.add_argument("--gmb-path", default=r"C:\Users\benja\Desktop\Vexto_Project\src\vexto\scoring\gmb_fetcher.py",
                   help="Sti til gmb_fetcher.py")
    p.add_argument("--log-level", default="INFO", help="DEBUG|INFO|WARNING|ERROR")
    p.add_argument("--quiet", action="store_true", help="Print kun status (ok/zero_results/...)")
    return p.parse_args()

# -------- hjælpere til pakkestien --------
def add_src_to_syspath_from_module_path(module_path: str, top_pkg_name: str = "vexto"):
    """
    Finder folderen som indeholder top-pakken (fx '...\\src') ud fra filstien
    '...\\src\\vexto\\scoring\\gmb_fetcher.py' og prepender den til sys.path.
    """
    module_path = os.path.abspath(module_path)
    parts = module_path.split(os.sep)
    try:
        idx = len(parts) - 1 - parts[::-1].index(top_pkg_name)  # index for 'vexto'
    except ValueError:
        raise RuntimeError(f"Kunne ikke finde pakken '{top_pkg_name}' i stien: {module_path}")
    pkg_parent = os.sep.join(parts[:idx])  # mappen som indeholder 'vexto'
    if pkg_parent and pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)

@asynccontextmanager
async def client_cm(AsyncHtmlClient):
    c = AsyncHtmlClient()
    if hasattr(c, "__aenter__"):
        async with c as client:
            yield client
    else:
        try:
            yield c
        finally:
            aclose = getattr(c, "aclose", None)
            if aclose:
                try:
                    await aclose()
                except Exception:
                    pass

async def run_one(fetch_gmb_data, AsyncHtmlClient, url: str, name: str | None, phone: str | None, quiet: bool):
    api_key = get_api_key()
    if not api_key:
        print("ERROR: Google API-nøgle mangler – læg den i .env som GOOGLE_API_KEY=...", file=sys.stderr)
        return 3

    async with client_cm(AsyncHtmlClient) as client:
        metrics = await fetch_gmb_data(client=client, url=url, company_name=name, phone=phone)

    if quiet:
        print(metrics.get("gmb_status", "unknown"))
    else:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    status = metrics.get("gmb_status")
    if status == "ok":
        return 0
    elif status in ("zero_results", "invalid_input"):
        return 1
    else:
        return 3

async def main():
    args = parse_args()
    setup_logging(args.log_level)

    dotenv_used = load_env(args.env_file)
    if dotenv_used:
        logging.info("Indlæste .env fra: %s", dotenv_used)

    # Sørg for at 'src' (mappen der indeholder 'vexto') er på sys.path
    add_src_to_syspath_from_module_path(args.gmb_path, top_pkg_name="vexto")

    # Importer via pakkesti, så relative imports inde i modul virker
    try:
        http_mod = importlib.import_module("vexto.scoring.http_client")
        gmb_mod = importlib.import_module("vexto.scoring.gmb_fetcher")
    except Exception as e:
        print(f"ERROR: Kunne ikke importere pakken via 'vexto.scoring.*' – {e}", file=sys.stderr)
        sys.exit(3)

    try:
        AsyncHtmlClient = getattr(http_mod, "AsyncHtmlClient")
    except AttributeError:
        print("ERROR: Kunne ikke finde AsyncHtmlClient i vexto.scoring.http_client", file=sys.stderr)
        sys.exit(3)
    try:
        fetch_gmb_data = getattr(gmb_mod, "fetch_gmb_data")
    except AttributeError:
        print("ERROR: Kunne ikke finde fetch_gmb_data i vexto.scoring.gmb_fetcher", file=sys.stderr)
        sys.exit(3)

    if not args.url and not args.file:
        print("Brug enten --url eller --file", file=sys.stderr)
        sys.exit(2)

    if args.file:
        rc_all = 0
        with open(args.file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [x.strip() for x in line.split(",")]
                url = parts[0]
                name = parts[1] if len(parts) > 1 and parts[1] else None
                phone = parts[2] if len(parts) > 2 and parts[2] else None
                print(f"\n=== TEST: {url} (name={name or '-'}, phone={phone or '-'}) ===")
                rc = await run_one(fetch_gmb_data, AsyncHtmlClient, url, name, phone, args.quiet)
                rc_all = rc_all or rc
        sys.exit(rc_all)
    else:
        rc = await run_one(fetch_gmb_data, AsyncHtmlClient, args.url, args.name, args.phone, args.quiet)
        sys.exit(rc)

if __name__ == "__main__":
    asyncio.run(main())
