from app.schemas import RepairRequest


def test_repair_attempt_range():
    assert RepairRequest(session_id="abcdefgh", attempt=3).attempt == 3
