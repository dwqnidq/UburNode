"""路由成功响应信封测试。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.core.codes import HttpStatus
from app.main import create_app
from app.schemas.audio import AudioResult, AudioTags, EvidenceLevel, SearchAudioData
from tests.test_exceptions import _assert_envelope


def test_search_returns_http_200_envelope() -> None:
    mock_service = MagicMock()
    mock_service.search_audio = AsyncMock(
        return_value=SearchAudioData(
            results=[
                AudioResult(
                    audio_url="https://cdn.example.com/a.mp3",
                    audio_name="雨声",
                    tags=AudioTags(),
                    evidence_level=EvidenceLevel.B,
                    recommend_weight=0.75,
                )
            ]
        )
    )
    mock_state = MagicMock()
    mock_state.audio_service = mock_service

    app = create_app()
    with patch("app.main.get_app_state", return_value=mock_state):
        response = TestClient(app).post(
            "/api/audio/search",
            json={"sleep_stage_tags": ["深睡"], "content_tags": ["雨声"], "top_k": 5},
        )

    assert response.status_code == 200
    body = response.json()
    _assert_envelope(body, HttpStatus.OK)
    assert body["msg"] == "检索成功"
    assert len(body["data"]["results"]) == 1
    assert response.status_code == body["code"]
