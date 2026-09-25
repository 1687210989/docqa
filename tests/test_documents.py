"""文档 CRUD + 事务一致性（数据库与向量库是否同步）。"""

PAYLOAD = {
    "title": "报销制度",
    "content": "员工出差需在返回后 30 天内提交报销单，超过期限不予受理。"
    "报销单需附发票原件，并由部门主管签字确认。",
}


def create(client, **over):
    body = {**PAYLOAD, **over}
    return client.post("/api/documents/", json=body)


def test_create_document(client, vector_count):
    r = create(client)
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["id"] > 0
    assert data["title"] == PAYLOAD["title"]
    assert data["content"] == PAYLOAD["content"]
    assert data["created_at"] and data["updated_at"]       # 时间字段非空 → utc_now 生效
    assert vector_count(data["id"]) >= 1, "入库成功但向量库没写入"


def test_list_documents(client):
    create(client, title="文档一")
    create(client, title="文档二")
    r = client.get("/api/documents/")
    assert r.status_code == 200
    assert [d["title"] for d in r.json()] == ["文档一", "文档二"]


def test_get_document(client):
    doc_id = create(client).json()["id"]
    r = client.get(f"/api/documents/{doc_id}")
    assert r.status_code == 200
    assert r.json()["id"] == doc_id


def test_get_missing_document_404(client):
    r = client.get("/api/documents/999999")
    assert r.status_code == 404
    assert r.json()["detail"] == "文档不存在"


def test_create_missing_field_422(client):
    r = client.post("/api/documents/", json={"title": "只有标题"})
    assert r.status_code == 422


def test_create_wrong_type_422(client):
    """pydantic v2 默认不做 int→str 隐式转换，应被挡在校验层。"""
    r = client.post("/api/documents/", json={"title": 123, "content": 456})
    assert r.status_code == 422


def test_pagination(client):
    for i in range(5):
        create(client, title=f"d{i}")
    assert len(client.get("/api/documents/?limit=2").json()) == 2
    assert len(client.get("/api/documents/?skip=3").json()) == 2
    assert client.get("/api/documents/?limit=100").status_code == 200


def test_delete_document(client, vector_count):
    doc_id = create(client).json()["id"]
    assert vector_count(doc_id) >= 1

    r = client.delete(f"/api/documents/{doc_id}")
    assert r.status_code == 204, r.text

    assert client.get(f"/api/documents/{doc_id}").status_code == 404
    assert vector_count(doc_id) == 0, "库记录删了，向量还在 → 会搜到已删内容"


def test_delete_missing_document_404(client):
    r = client.delete("/api/documents/999999")
    assert r.status_code == 404
    assert r.json()["detail"] == "文档不存在"


def test_delete_keeps_other_documents(client):
    a = create(client, title="保留的").json()["id"]
    b = create(client, title="要删的").json()["id"]
    assert client.delete(f"/api/documents/{b}").status_code == 204
    left = client.get("/api/documents/").json()
    assert [d["id"] for d in left] == [a]
