import asyncio
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://127.0.0.1:8081/v1/chat/completions",
            json={
                "model": "coding-router",
                "messages": [{"role": "user", "content": "write a python function to add two numbers"}],
                "stream": True
            },
            headers={"Authorization": "Bearer sk-router-pws-u5NtTPqvp1axgof6DYPWzfTaKdVzAdQzqB6Z70I"},
            timeout=10.0
        )
        print(f"Status: {resp.status_code}")
        print(f"Headers: {resp.headers}")
        text = await resp.aread()
        print(f"Body: {text.decode('utf-8')}")

asyncio.run(main())
