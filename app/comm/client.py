"""comm-service gRPC 客户端（AudioMaterialService）。

UburNode 不直连 MongoDB；所有 CUD 经 comm-service（规范红线）。
proto 真源：仓库根 proto/bionode_comm.proto，变更后须重新 gen_proto.sh。
"""

from __future__ import annotations

import grpc
from loguru import logger

from app.comm.grpc_gen import bionode_comm_pb2, bionode_comm_pb2_grpc
from app.core.config import Settings


class CommClient:
    """封装 AudioMaterialService 的 gRPC 调用。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._channel: grpc.aio.Channel | None = None
        self._stub: bionode_comm_pb2_grpc.AudioMaterialServiceStub | None = None

    async def connect(self) -> None:
        target = self._settings.comm_grpc_target
        logger.info("正在连接 comm-service gRPC：{}", target)
        self._channel = grpc.aio.insecure_channel(target)
        self._stub = bionode_comm_pb2_grpc.AudioMaterialServiceStub(self._channel)

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None

    def _require_stub(self) -> bionode_comm_pb2_grpc.AudioMaterialServiceStub:
        if self._stub is None:
            raise RuntimeError("CommClient 未连接，请先在 lifespan 中调用 connect()")
        return self._stub

    async def get_audio_material(self, material_id: str) -> bionode_comm_pb2.AudioMaterialInfo:
        stub = self._require_stub()
        response = await stub.GetAudioMaterial(bionode_comm_pb2.IdReq(id=material_id))
        return response.material

    async def create_audio_material(
        self,
        *,
        category_code: int,
        noise_color: str,
        name: str,
        description: str,
        tags: list[str],
        audio_url: str,
    ) -> None:
        stub = self._require_stub()
        audio_info = bionode_comm_pb2.AudioMetaInfo(
            meta_data=bionode_comm_pb2.AudioMetaData(url=audio_url),
        )
        request = bionode_comm_pb2.CreateAudioMaterialReq(
            category_code=category_code,
            noise_color=noise_color,
            name=name,
            description=description,
            tags=tags,
            audio_info=audio_info,
        )
        await stub.CreateAudioMaterial(request)

    async def update_audio_material(
        self,
        material_id: str,
        *,
        category_code: int | None = None,
        noise_color: str | None = None,
        name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        audio_url: str | None = None,
        status: int | None = None,
    ) -> None:
        """Update 需先 Get 再合并：proto 要求全量字段，comm 不做部分更新。"""
        stub = self._require_stub()
        existing = await self.get_audio_material(material_id)

        audio_info = existing.audio_info
        if audio_url is not None:
            audio_info = bionode_comm_pb2.AudioMetaInfo(
                meta_data=bionode_comm_pb2.AudioMetaData(url=audio_url),
                is_loopable=existing.audio_info.is_loopable,
                is_voice=existing.audio_info.is_voice,
            )

        request = bionode_comm_pb2.UpdateAudioMaterialReq(
            id=material_id,
            category_code=category_code if category_code is not None else existing.category_code,
            noise_color=noise_color if noise_color is not None else existing.noise_color,
            name=name if name is not None else existing.name,
            description=description if description is not None else existing.description,
            tags=tags if tags is not None else list(existing.tags),
            audio_info=audio_info,
            status=status if status is not None else existing.status,
        )
        await stub.UpdateAudioMaterial(request)

    async def delete_audio_material(self, material_id: str) -> None:
        stub = self._require_stub()
        await stub.DeleteAudioMaterial(bionode_comm_pb2.IdReq(id=material_id))

    async def list_audio_materials_by_name(
        self, name: str
    ) -> list[bionode_comm_pb2.AudioMaterialInfo]:
        """Create 返回 EmptyRes 时的临时反查方案，待 proto 扩展后移除。"""
        stub = self._require_stub()
        from app.comm.grpc_gen import bionode_common_pb2

        response = await stub.ListAudioMaterials(
            bionode_comm_pb2.ListAudioMaterialsReq(
                page=bionode_common_pb2.PageRequest(page=1, page_size=10),
                name=name,
            )
        )
        return list(response.materials)
