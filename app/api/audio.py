"""对外 HTTP 路由（唯一 audio 路由组，4 个端点）。

对应规范 §四：POST/PUT/DELETE /audio、POST /audio/search。
校验与序列化由 Pydantic + FastAPI 在边界完成，此处只做编排转发。
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.audio import (
    CreateAudioRequest,
    CreateAudioResponse,
    EmptyResponse,
    SearchAudioRequest,
    SearchAudioResponse,
    UpdateAudioRequest,
)
from app.services.audio import AudioService

router = APIRouter(prefix="/audio", tags=["audio"])


def get_audio_service() -> AudioService:
    """从进程单例取 AudioService；lifespan 未完成时返回 503。"""
    from app.main import get_app_state

    state = get_app_state()
    if state.audio_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务尚未就绪",
        )
    return state.audio_service


@router.post("", response_model=CreateAudioResponse, status_code=status.HTTP_201_CREATED)
async def create_audio(
    body: CreateAudioRequest,
    service: AudioService = Depends(get_audio_service),
) -> CreateAudioResponse:
    """创建音频：comm-service 写 Mongo → EsSync 同步 ES（含 embedding）。"""
    return await service.create_audio(body)


@router.put("/{material_id}", response_model=EmptyResponse)
async def update_audio(
    material_id: str,
    body: UpdateAudioRequest,
    service: AudioService = Depends(get_audio_service),
) -> EmptyResponse:
    """更新音频；material_id 走路径参数，不进请求体（规范 §四）。"""
    await service.update_audio(material_id, body)
    return EmptyResponse()


@router.delete("/{material_id}", response_model=EmptyResponse)
async def delete_audio(
    material_id: str,
    service: AudioService = Depends(get_audio_service),
) -> EmptyResponse:
    """删除音频：comm 删 Mongo 真值 + EsSync 删 ES 索引副本。"""
    await service.delete_audio(material_id)
    return EmptyResponse()


@router.post("/search", response_model=SearchAudioResponse)
async def search_audio(
    body: SearchAudioRequest,
    service: AudioService = Depends(get_audio_service),
) -> SearchAudioResponse:
    """三维度检索：只读 ES，不写 Mongo / ES（规范红线）。"""
    return await service.search_audio(body)
