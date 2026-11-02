import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from movie_rating.api.app import create_app
from movie_rating.data.prepare import chronological_split, prepare
from movie_rating.lab import CATALOG, LabRequest, LabService
from movie_rating.lab_supervised import classification, paired_intervals, rating
from movie_rating.lab_unsupervised import association, cluster_metrics, clustering, pca_experiment
from movie_rating.lab_reinforcement import reinforcement


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


def test_catalog_has_exactly_twelve_algorithms_and_full_explanations():
    assert len(CATALOG['groups']) == 6
    assert sum(len(g['algorithms']) for g in CATALOG['groups']) == 12
    for group in CATALOG['groups']:
        assert group['metrics'] and group['charts']
        assert all(m['definition'] and m['interpretation'] and m['question'] for m in group['metrics'])
        assert all(c['axes'] and c['question'] for c in group['charts'])


@pytest.mark.parametrize('group,handler', [('rating', rating), ('classification', classification),
                                         ('clustering', clustering), ('association', association),
                                         ('pca', pca_experiment), ('reinforcement', reinforcement)])
def test_each_group_runs_real_algorithms_and_serializes_without_nan(group, handler, lab_data):
    from movie_rating.lab import native

    splits, movies = lab_data
    request = LabRequest(group=group, max_users=24, max_items=20, epochs=2, episodes=100)
    result = handler(splits, movies, request, lambda *_: None)
    encoded = json.dumps(native(result), allow_nan=False)
    assert encoded and result['rows'] and result['charts']
    assert all(r['fit_seconds'] >= 0 for r in result['rows'])
    assert set(range(len(next(g for g in CATALOG['groups'] if g['id'] == group)['charts']))) <= {c['index'] for c in result['charts']}
    if group == 'rating':
        assert {'linear','xgboost','svd','ncf'} <= {r['algorithm'] for r in result['rows']}
        assert all(.0 <= r['rmse'] <= 4.5 for r in result['rows'])
    if group == 'association':
        assert result['same_itemsets']
        assert len({r['rules'] for r in result['rows']}) == 1
    if group == 'reinforcement':
        assert all(r['repeat_fraction'] == 0 and r['invalid_fraction'] == 0 for r in result['rows'])


def test_undefined_cluster_scores_are_missing_not_zero():
    values = cluster_metrics(np.asarray([[0.], [1.], [2.]]), np.asarray([-1, -1, -1]))
    assert values['silhouette'] is None and values['davies_bouldin'] is None
    assert values['noise_fraction'] == 1


def test_paired_bootstrap_detects_known_prediction_difference(lab_data):
    splits, _ = lab_data
    frame = splits['test']
    perfect = frame.rating.to_numpy()
    result = paired_intervals(frame, {'perfect': perfect, 'offset': perfect+1}, 'perfect', 42)
    assert all(row['ci_low'] == pytest.approx(1) and row['ci_high'] == pytest.approx(1) for row in result)


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
        assert csv.status_code == 200 and 'macro_f1' in csv.text
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


def test_user_clustering_uses_same_representation_and_measures_seed_stability(lab_data):
    splits, movies = lab_data
    result = clustering(splits, movies, LabRequest(group='clustering', cluster_object='users', repeats=2), lambda *_: None)
    assert result['objects'] == splits['train'].userId.nunique()
    assert all(row['stability_ari'] is not None for row in result['rows'])
