# debug_fetchers.py
import asyncio
import json
from src.vexto.scoring.analyzer import analyze_single_url
from src.vexto.scoring.http_client import AsyncHtmlClient

async def main():
    async with AsyncHtmlClient() as client:
        result = await analyze_single_url(client, "https://www.proshop.dk")
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
