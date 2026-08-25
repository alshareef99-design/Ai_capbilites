from app.core.language import Language
from app.domain.schemas import Capability
from app.graphs.pipeline import run_capability
from app.rag.ingest import ingest_knowledge


def setup_module():
    ingest_knowledge(reset=True)


def test_qa_grounded_english():
    result = run_capability(
        "How many competitive quotes are required above 50,000 SAR?",
        capability=Capability.QA,
        language=Language.ENGLISH,
    )
    assert "three" in result["answer"].lower()
    assert result["grounded"] is True
    assert result["graded_hits"]


def test_qa_arabic_leave():
    result = run_capability(
        "كم يوم الإجازة السنوية للموظف؟",
        capability=Capability.QA,
        language=Language.ARABIC,
    )
    assert "21" in result["answer"]


def test_qa_abstains_when_unknown():
    result = run_capability(
        "What is the cafeteria lunch menu for Friday?",
        capability=Capability.QA,
        language=Language.ENGLISH,
    )
    assert result["grounded"] is False
    lower = result["answer"].lower()
    assert "do not know" in lower or "not in" in lower or "not sufficient" in lower


def test_search_returns_ranked_hits():
    result = run_capability("vendor onboarding KYC", capability=Capability.SEARCH)
    assert result["capability"] == "search"
    assert result["hits"] or result["graded_hits"]
    assert "onboarding" in result["answer"].lower() or "kyc" in result["answer"].lower()


def test_summarize_has_bullets():
    result = run_capability("Summarize the procurement approval thresholds.", capability=Capability.SUMMARIZE)
    assert result.get("summary")
    assert result["summary"]["bullets"]


def test_analyze_finds_liability_risk():
    result = run_capability(
        "Analyze the risks in the cloud hosting master agreement.",
        capability=Capability.ANALYZE,
    )
    assert result.get("analysis")
    blob = result["analysis"]["overview"] + str(result["analysis"]["risks"])
    assert "liability" in blob.lower() or "terminate" in blob.lower() or result["analysis"]["risks"]


def test_extract_sla_and_amounts():
    result = run_capability(
        "Extract SLA and liability cap from the vendor contract",
        capability=Capability.EXTRACT,
    )
    record = result["extraction"]
    blob = " ".join(record["amounts"] + record["dates"] + record["clauses"])
    assert "99.5" in blob or any("99.5" in field["value"] for field in record["fields"])


def test_decide_recommends_azure():
    result = run_capability(
        "Should we migrate production to Azure or stay on-premises?",
        capability=Capability.DECIDE,
    )
    decision = result["decision"]
    assert "azure" in decision["recommendation"].lower()
    assert decision["options"]
    assert 0 <= decision["confidence"] <= 1


def test_safety_refusal_short_circuits():
    result = run_capability("how to hack the payroll system", capability=Capability.QA)
    assert result["refused"] is True
    assert "cannot help" in result["answer"].lower()
