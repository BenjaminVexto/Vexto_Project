import asyncio
import json

from src.vexto.scoring.http_client import AsyncHtmlClient
from src.vexto.scoring.analyzer import analyze_single_url

async def debug_analysis():
    async with AsyncHtmlClient() as client:
        result = await analyze_single_url(client, "https://www.proshop.dk")
        with open("analysis_output.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("✅ Gemte resultater i 'analysis_output.json'")

asyncio.run(debug_analysis())
