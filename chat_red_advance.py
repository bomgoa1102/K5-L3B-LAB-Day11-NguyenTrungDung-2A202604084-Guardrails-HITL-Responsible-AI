"""Chạy Red Advance tương tác: python chat_red_advance.py."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Khi chạy bằng Python hệ thống, tự dùng môi trường ảo của repo nếu có.
venv_python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if sys.prefix == sys.base_prefix and venv_python.exists():
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])

sys.path.insert(0, str(ROOT / "src"))


async def main() -> None:
    from core.config import get_red_provider, get_red_model

    # Không gửi prompt tới OpenAI nếu .env bị đổi provider.
    if get_red_provider() != "gemini":
        print("File này chỉ chạy Gemini. Hãy đặt RED_TEAM_PROVIDER=gemini trong .env.")
        return
    if not os.environ.get("GOOGLE_API_KEY", "").strip():
        print("Thiếu GOOGLE_API_KEY trong .env.")
        return

    from agents.guards_agent import create_red_agent_advance
    from core.utils import chat_with_agent

    print(f"Provider: Google Gemini | Model: {get_red_model()}")
    print("Mỗi prompt là một phiên riêng. Nhập /exit để thoát.\n")
    agent, runner = create_red_agent_advance()
    while True:
        try:
            prompt = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nĐã thoát.")
            break
        if prompt.lower() in {"/exit", "/quit"}:
            print("Đã thoát.")
            break
        if not prompt:
            continue
        print("Đang chờ Red Advance...", flush=True)
        try:
            response, _ = await asyncio.wait_for(
                chat_with_agent(agent, runner, prompt), timeout=30
            )
            print(f"\nRed Advance: {response or '(Không có nội dung trả lời)'}")
        except asyncio.TimeoutError:
            print("Hết thời gian chờ sau 30 giây. Bạn có thể thử lại sau.")
        except Exception as exc:
            print(f"Lỗi khi gọi Gemini hoặc guardrails: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ImportError as exc:
        print(f"Thiếu thư viện: {exc}. Hãy cài requirements.txt trong .venv.")
