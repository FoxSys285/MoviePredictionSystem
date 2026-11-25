from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse

from movie_rating.lab import GROUPS, LabRequest, LabService
from movie_rating.settings import PROJECT_ROOT


def lab_router(settings):
    router = APIRouter(prefix='/api/v1/experiment-lab', tags=['Experiment lab'])
    service = LabService(settings)

    @router.get('/catalog')
    def catalog():
        return service.catalog()

    @router.get('/runs')
    def runs(group: str | None = None):
        if group is not None and group not in GROUPS:
            raise HTTPException(422, 'Nhóm thuật toán không hợp lệ.')
        return service.list_runs(group)

    @router.post('/runs', status_code=202)
    def start(request: LabRequest):
        try:
            return service.start(request)
        except FileNotFoundError as error:
            raise HTTPException(503, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    @router.get('/runs/{run_id}')
    def get(run_id: str):
        try:
            return service.get(run_id)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error

    @router.post('/runs/{run_id}/cancel')
    def cancel(run_id: str):
        try:
            return service.cancel(run_id)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error

    @router.post('/runs/{run_id}/resume', status_code=202)
    def resume(run_id: str):
        try:
            return service.resume(run_id)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    @router.get('/runs/{run_id}/logs')
    def logs(run_id: str, cursor: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=500)):
        try:
            return service.logs(run_id, cursor, limit)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error

    @router.get('/runs/{run_id}/logs/download')
    def download_logs(run_id: str):
        try:
            path = service.logs_directory(run_id) / 'events.jsonl'
            if not path.is_file():
                raise FileNotFoundError('Lần chạy cũ chưa có nhật ký.')
            size = path.stat().st_size

            def chunks():
                # Download a fixed prefix even if the worker is still appending events.
                remaining = size
                with path.open('rb') as stream:
                    while remaining:
                        data = stream.read(min(65536, remaining))
                        if not data:
                            break
                        remaining -= len(data)
                        yield data

            return StreamingResponse(chunks(), media_type='application/x-ndjson', headers={
                'Content-Disposition': f'attachment; filename="{run_id}.jsonl"',
                'Content-Length': str(size),
            })
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error

    @router.get('/runs/{run_id}/export/{format}')
    def export(run_id: str, format: str):
        if format not in ('json', 'csv', 'xlsx'):
            raise HTTPException(422, 'Chỉ hỗ trợ JSON, CSV và Excel (XLSX).')
        try:
            job = service.get(run_id)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        if job['status'] != 'completed':
            raise HTTPException(409, 'Lần chạy chưa có kết quả hoàn tất.')
        if format == 'csv':
            return Response(service.csv_export(run_id), media_type='text/csv; charset=utf-8',
                            headers={'Content-Disposition': f'attachment; filename="{run_id}.csv"'})
        if format == 'xlsx':
            return Response(service.excel_export(run_id),
                            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            headers={'Content-Disposition': f'attachment; filename="{run_id}.xlsx"'})
        return FileResponse(service.directory / run_id / 'result.json', media_type='application/json', filename=f'{run_id}.json')

    @router.get('/document')
    def document():
        path = PROJECT_ROOT / 'docs/experiment-groups.docx'
        if not path.is_file():
            raise HTTPException(404, 'Chưa có tài liệu mô tả.')
        return FileResponse(path, filename='Mo-ta-thuc-nghiem.docx')

    return router
