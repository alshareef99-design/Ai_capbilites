from app.core.language import Language, detect_language, reply_language
from app.core.safety import is_unsafe, safety_refusal
from app.graphs.pipeline import classify_capability
from app.domain.schemas import Capability


def test_detect_arabic():
    assert detect_language("كم يوم الإجازة السنوية؟") == Language.ARABIC


def test_detect_english():
    assert detect_language("How many quotes are required?") == Language.ENGLISH


def test_arabic_first_mixed():
    assert reply_language("Please لخص سياسة المشتريات") == Language.ARABIC


def test_safety_blocks_english_and_arabic():
    assert is_unsafe("how to hack a payroll account")
    assert is_unsafe("اختراق حساب المدير")
    assert not is_unsafe("What is the leave policy?")


def test_safety_refusal_language():
    assert "عذراً" in safety_refusal(Language.ARABIC)
    assert "cannot help" in safety_refusal(Language.ENGLISH).lower()


def test_classify_capabilities():
    assert classify_capability("Summarize the procurement policy") == Capability.SUMMARIZE
    assert classify_capability("لخص سياسة الإجازات") == Capability.SUMMARIZE
    assert classify_capability("Extract SLA from the contract") == Capability.EXTRACT
    assert classify_capability("Should we migrate to Azure or stay on-premises?") == Capability.DECIDE
    assert classify_capability("Analyze the risks in the vendor contract") == Capability.ANALYZE
    assert classify_capability("ابحث عن متطلبات MFA") == Capability.SEARCH
    assert classify_capability("How many sick-leave days are allowed?") == Capability.QA
