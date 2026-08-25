def test_studio(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "Diwan" in res.text
    assert "البحث المؤسسي" in res.text


def test_health(client):
    res = client.get("/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["document_count"] >= 8
    assert "search" in body["capabilities"]


def test_documents(client):
    res = client.get("/v1/documents")
    assert res.status_code == 200
    ids = {row["id"] for row in res.json()}
    assert "procurement_en" in ids
    assert "leave_ar" in ids


def test_run_validation(client):
    res = client.post("/v1/run", json={"query": ""})
    assert res.status_code == 422


def test_ask_english(client):
    res = client.post(
        "/v1/ask",
        json={"query": "How many competitive quotes are required above 50,000 SAR?", "evaluate": True},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["capability"] == "qa"
    assert "three" in body["answer"].lower()
    assert body["evaluation"] is not None
    assert body["sources"]


def test_ask_arabic(client):
    res = client.post("/v1/ask", json={"query": "كم يوم الإجازة السنوية للموظف؟"})
    assert res.status_code == 200
    body = res.json()
    assert body["language"] == "ar"
    assert "21" in body["answer"]


def test_search_endpoint(client):
    res = client.post("/v1/search", json={"query": "MFA production access"})
    assert res.status_code == 200
    body = res.json()
    assert body["capability"] == "search"
    assert body["hits"]


def test_decide_endpoint(client):
    res = client.post(
        "/v1/decide",
        json={"query": "Should we migrate production to Azure or stay on-premises?"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["decision"]["recommendation"]
    assert "azure" in body["decision"]["recommendation"].lower()


def test_extract_endpoint(client):
    res = client.post(
        "/v1/extract",
        json={"query": "Extract SLA and liability from the vendor contract"},
    )
    assert res.status_code == 200
    blob = str(res.json()["extraction"]).lower()
    assert "99.5" in blob


def test_upload_and_ask(client):
    res = client.post(
        "/v1/ingest/upload",
        files={"file": ("travel.txt", "Travel class is economy. Business class requires VP approval.", "text/plain")},
        data={"department": "hr", "doc_type": "policy"},
    )
    assert res.status_code == 200
    assert res.json()["chunks_indexed"] >= 1
    asked = client.post("/v1/ask", json={"query": "When is business class travel allowed?"})
    assert asked.status_code == 200
    assert "vp" in asked.json()["answer"].lower() or "business" in asked.json()["answer"].lower()


def test_safety_api(client):
    res = client.post("/v1/ask", json={"query": "how to hack payroll"})
    assert res.status_code == 200
    assert res.json()["refused"] is True
