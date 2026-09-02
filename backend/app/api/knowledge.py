"""知识库文件上传、列表、删除和重新索引接口。"""
import os
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException

from app.services import kb_service
from app.core.context import get_current_user
from app.core.metrics import metrics

router = APIRouter()

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
ALLOWED_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "application/octet-stream",  # 部分浏览器对 .md/.txt 返回此类型
}


@router.post("/kb/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    # 1) 文件名与扩展名校验
    """校验并保存上传文件，然后以后台任务方式启动索引。"""
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"仅支持 PDF / Word(.docx) / TXT / MD 文件，当前类型：{ext or '未知'}",
        )

    # 2) MIME 校验（宽松：octet-stream 放行）
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_MIMES:
        raise HTTPException(
            status_code=400,
            detail=f"文件 MIME 类型不被允许：{content_type}",
        )

    # 3) 读取并校验大小
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（{len(content) / 1024 / 1024:.1f} MB），上限 {MAX_UPLOAD_SIZE / 1024 / 1024:.0f} MB",
        )

    # 4) 先落盘建 pending 记录并立即返回；索引放到后台任务执行
    try:
        info = await kb_service.create_pending_file(filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存失败：{e}")

    metrics.incr("kb_uploads")
    background_tasks.add_task(kb_service.run_indexing, info["id"], filename, ext, get_current_user())
    return {"code": 0, "data": info}


@router.get("/kb/files")
async def list_files():
    """返回当前用户的知识库文件和索引进度。"""
    try:
        return {"code": 0, "data": await kb_service.list_files()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/kb/files/{file_id}/reindex", status_code=202)
async def reindex_file(file_id: str, background_tasks: BackgroundTasks):
    """启动指定文件的后台重新索引任务，并返回 202。"""
    try:
        job = await kb_service.prepare_reindex(file_id, get_current_user())
    except kb_service.ReindexConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except kb_service.RawFileMissing as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    background_tasks.add_task(kb_service.run_indexing, job[0], job[1], job[2], job[3], job[4])
    return {
        "code": 0,
        "data": {
            "id": file_id,
            "status": "indexing",
            "message": "重新索引任务已启动",
        },
    }

@router.delete("/kb/files/{file_id}")
async def delete_file(file_id: str):
    """删除指定知识库文件及其原文、向量索引。"""
    ok = await kb_service.delete_file(file_id)
    if not ok:
        raise HTTPException(status_code=404, detail="文件不存在")
    return {"code": 0, "message": "已删除"}
