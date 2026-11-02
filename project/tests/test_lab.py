import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from movie_rating.api.app import create_app
from movie_rating.data.prepare import chronological_split, prepare
from movie_rating.lab import CATALOG, LabRequest, LabService
from movie_rating.lab_compact import (association, classification, clustering,
                                     clustering_scores, pca_experiment, rating,
                                     regression_scores, rule_scores)


@pytest.fixture
def lab_data():
    rng = np.random.default_rng(42)
    rows = []
    for user in range(1, 25):
        for movie in rng.choice(np.arange(1, 33), 24, replace=False):
            rating_value = .5 + .5*((user + int(movie)) % 10)
            rows.append((user, int(movie), rating_value))
    rng.shuffle(rows)
    frame = pd.DataFrame(rows, columns=['userId', 'movieId', 'rating'])
    frame['timestamp'] = np.arange(len(frame))+1_000_000
    movies = pd.DataFrame({'movieId': range(1, 33), 'title': [f'Movie {i}' for i in range(1, 33)],
                           'genres': ['Action|Comedy' if i % 2 else 'Drama|Romance' for i in range(1, 33)]})
    return chronological_split(frame, .7, .15), movies


def test_catalog_has_five_groups_and_exactly_one_chart_per_requested_metric():
    assert len(CATALOG['groups']) == 5
    assert sum(len(g['algorithms']) for g in CATALOG['groups']) == 11
    expected = {'rating': ['mae','mse','rmse','r2'], 'classification': ['accuracy','precision','recall','f1'],
                'clustering': ['silhouette','chi','dbi'], 'association': ['support','confidence','lift'],
                'pca': ['explained_variance_ratio','reconstruction_error','downstream_performance']}
    for group in CATALOG['groups']:
        assert [m['key'] for m in group['metrics']] == expected[group['id']]
        assert len(group['charts']) == len(group['metrics'])
        assert all(m['definition'] and m['interpretation'] and m['question'] for m in group['metrics'])
        assert all(c['axes'] and c['question'] for c in group['charts'])


@pytest.mark.parametrize('group,handler', [('rating', rating), ('classification', classification),
                                         ('clustering', clustering), ('association', association),
                                         ('pca', pca_experiment)])
def test_each_group_runs_real_algorithms_and_serializes_without_nan(group, handler, lab_data):
    from movie_rating.lab import native

    splits, movies = lab_data
    request = LabRequest(group=group, max_users=24, max_items=20, epochs=2)
    result = handler(splits, movies, request, lambda *_: None)
    encoded = json.dumps(native(result), allow_nan=False)
    assert encoded and result['rows'] and result['charts']
    definition = next(g for g in CATALOG['groups'] if g['id'] == group)
    assert len(result['charts']) == len(definition['metrics'])
    assert {r['split'] for r in result['rows']} == {'validation','test'}
    assert all([s['label'] for s in c['series']] == ['Validation','Test'] for c in result['charts'])
    assert all(set(r) == {'algorithm','name','split',*(m['key'] for m in definition['metrics'])} for r in result['rows'])
    if group == 'rating':
        assert {'linear','xgboost','svd','ncf'} <= {r['algorithm'] for r in result['rows']}
        assert all(.0 <= r['rmse'] <= 4.5 for r in result['rows'])
    if group == 'association':
        for split in ('validation','test'):
            a,b=[r for r in result['rows'] if r['split']==split]
            assert all(a[key] == pytest.approx(b[key]) if a[key] is not None else b[key] is None for key in ('support','confidence','lift'))


def test_undefined_cluster_scores_are_missing_not_zero():
    values = clustering_scores(np.asarray([[0.], [1.], [2.]]), np.asarray([-1, -1, -1]))
    assert values == {'silhouette':None,'chi':None,'dbi':None}


def test_regression_metrics_have_correct_units_and_keep_negative_r2():
    result=regression_scores([1,3],[2,3])
    assert result == pytest.approx({'mae':.5,'mse':.5,'rmse':np.sqrt(.5),'r2':.5})
    assert regression_scores([1,3],[5,5])['r2'] < 0
    assert regression_scores([3,3],[3,3])['r2'] is None


def test_association_metrics_are_measured_on_fixed_heldout_transactions():
    data=pd.DataFrame({1:[True,True,False,False],2:[True,False,True,False]})
    assert rule_scores(data,[(frozenset([1]),frozenset([2]))]) == pytest.approx({'support':.25,'confidence':.5,'lift':1})
    assert rule_scores(data,[]) == {'support':None,'confidence':None,'lift':None}


def test_job_api_persists_temporal_provenance_and_exports(configured):
    prepare(configured)
    with TestClient(create_app(configured)) as client:
        root = '/api/v1/experiment-lab'
        assert client.get(root+'/catalog').json()['data_ready']
        assert client.post(root+'/runs', json={'group': 'unknown'}).status_code == 422
        assert client.post(root+'/runs', json={'group': 'classification', 'max_users': 10001}).status_code == 422
        response = client.post(root+'/runs', json={'group': 'classification', 'max_users': 10})
        assert response.status_code == 202
        run_id = response.json()['id']
        for _ in range(200):
            job = client.get(root+'/runs/'+run_id).json()
            if job['status'] not in ('queued','running'):
                break
            time.sleep(.05)
        assert job['status'] == 'completed', job
        splits = job['result']['provenance']['splits']
        assert splits['train']['max_timestamp'] < splits['validation']['min_timestamp']
        assert splits['validation']['max_timestamp'] < splits['test']['min_timestamp']
        assert job['result']['provenance']['sample_sha256']
        assert client.get(root+f'/runs/{run_id}/export/json').status_code == 200
        csv = client.get(root+f'/runs/{run_id}/export/csv')
        assert csv.status_code == 200 and 'f1' in csv.text and 'macro_f1' not in csv.text
        assert len(job['result']['charts']) == 4
        assert client.post(root+'/runs', json={'group':'reinforcement'}).status_code == 422
        assert client.get(root+'/runs?group=classification').json()[0]['id'] == run_id
        assert client.get(root+'/runs/not-a-run').status_code == 404
        assert client.get(root+f'/runs/{run_id}/export/exe').status_code == 422
    restored = LabService(configured).get(run_id)
    assert restored['status'] == 'completed' and restored['result']['rows']


def test_missing_data_and_duplicate_active_jobs_are_reported(configured):
    service = LabService(configured)
    with pytest.raises(FileNotFoundError):
        service.start(LabRequest(group='rating'))
    prepare(configured)
    service.active = 'already-running'
    with pytest.raises(RuntimeError):
        service.start(LabRequest(group='rating'))


def test_cancellation_does_not_publish_partial_results(configured, monkeypatch):
    import threading

    prepare(configured)
    service = LabService(configured)
    entered, release = threading.Event(), threading.Event()
    original = service._load

    def blocked(request):
        entered.set()
        release.wait(3)
        return original(request)

    monkeypatch.setattr(service, '_load', blocked)
    job = service.start(LabRequest(group='classification', max_users=10))
    assert entered.wait(3)
    service.cancel(job['id'])
    release.set()
    for _ in range(100):
        finished = service.get(job['id'])
        if finished['status'] == 'cancelled':
            break
        time.sleep(.03)
    assert finished['status'] == 'cancelled'
    assert 'result' not in finished
    assert service.active is None


def test_prepared_sampling_is_shared_between_groups(configured):
    prepare(configured)
    service = LabService(configured)
    _, _, a = service._load(LabRequest(group='classification', max_users=10))
    _, _, b = service._load(LabRequest(group='rating', max_users=10))
    assert a['sample_sha256'] == b['sample_sha256']
    assert a['splits'] == b['splits']


def test_user_clustering_evaluates_both_heldout_splits(lab_data):
    splits, movies = lab_data
    result = clustering(splits, movies, LabRequest(group='clustering', cluster_object='users', repeats=2), lambda *_: None)
    assert len(result['rows']) == 4
    assert {r['split'] for r in result['rows']} == {'validation','test'}
    assert len(result['charts']) == 3


def test_atomic_report_write_retries_transient_windows_sharing_lock(tmp_path, monkeypatch):
    import movie_rating.lab as lab

    path=tmp_path/'job.json'
    original=lab.os.replace
    attempts=[]
    def busy_once(source,destination):
        attempts.append(1)
        if len(attempts)<3:
            raise PermissionError('File is temporarily being read')
        return original(source,destination)
    monkeypatch.setattr(lab.os,'replace',busy_once)
    monkeypatch.setattr(lab.time,'sleep',lambda _:None)
    lab.save_json(path,{'status':'completed'})
    assert json.loads(path.read_text()) == {'status':'completed'}
    assert len(attempts)==3 and not list(tmp_path.glob('*.tmp'))
