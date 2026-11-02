from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

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

    @router.get('/runs/{run_id}/export/{format}')
    def export(run_id: str, format: str):
        if format not in ('json', 'csv'):
            raise HTTPException(422, 'Chỉ hỗ trợ JSON và CSV.')
        try:
            job = service.get(run_id)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        if job['status'] != 'completed':
            raise HTTPException(409, 'Lần chạy chưa có kết quả hoàn tất.')
        if format == 'csv':
            return Response(service.csv_export(run_id), media_type='text/csv; charset=utf-8',
                            headers={'Content-Disposition': f'attachment; filename="{run_id}.csv"'})
        return FileResponse(service.directory / run_id / 'result.json', media_type='application/json', filename=f'{run_id}.json')

    @router.get('/document')
    def document():
        path = PROJECT_ROOT / 'docs/experiment-groups.docx'
        if not path.is_file():
            raise HTTPException(404, 'Chưa có tài liệu mô tả.')
        return FileResponse(path, filename='Mo-ta-thuc-nghiem.docx')

    return router
