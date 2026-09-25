from app.llm import chat_complete

SYSTEM = ("你是答案质检员。判断【答案】是否真正回答了【问题】、是否出现资料之外的编造内容。"
          "只回答 pass 或 fail，不要解释。")


def pass_check(question: str, answer: str) -> bool:
    """Critic Agent：返回 True 表示通过。"""
    resp = chat_complete([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"【问题】{question}\n【答案】{answer}"},
    ])
    return (resp.choices[0].message.content or "").strip().lower().startswith("pass")

