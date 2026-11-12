"""Tree ensembles and linear models from scikit-learn, LightGBM and XGBoost."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

import montlok
from montlok import export

sklearn = pytest.importorskip("sklearn")
from sklearn import ensemble, linear_model, neural_network, pipeline, preprocessing, tree  # noqa: E402


def data(n=400, features=6, seed=0, missing=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, features))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] * x[:, 2] + 0.1 * rng.normal(size=n)
    if missing:
        x[rng.random(x.shape) < 0.1] = np.nan
    return x, y


def exact(model, x, expected):
    np.testing.assert_allclose(model.predict(x).reshape(expected.shape), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("estimator", [
    tree.DecisionTreeRegressor(max_depth=8, random_state=0),
    tree.ExtraTreeRegressor(max_depth=8, random_state=0),
    ensemble.RandomForestRegressor(n_estimators=30, max_depth=6, random_state=0, n_jobs=1),
    ensemble.ExtraTreesRegressor(n_estimators=30, max_depth=6, random_state=0, n_jobs=1),
    ensemble.GradientBoostingRegressor(n_estimators=60, max_depth=3, learning_rate=0.1, random_state=0),
    ensemble.GradientBoostingRegressor(n_estimators=20, init="zero", random_state=0),
    ensemble.GradientBoostingRegressor(n_estimators=20, loss="huber", random_state=0),
])
def test_sklearn_trees_are_exact(estimator):
    x, y = data()
    estimator.fit(x, y)
    model = export.from_sklearn(estimator).load()
    x_test, _ = data(seed=1)
    exact(model, x_test, estimator.predict(x_test))


def test_decision_tree_with_missing_values():
    x, y = data(missing=True)
    estimator = tree.DecisionTreeRegressor(max_depth=7, random_state=0).fit(x, y)
    x_test, _ = data(seed=2, missing=True)
    exact(export.from_sklearn(estimator).load(), x_test, estimator.predict(x_test))


@pytest.mark.parametrize("loss", ["squared_error", "poisson"])
def test_hist_gradient_boosting_with_missing_values(loss):
    x, y = data(missing=True)
    if loss == "poisson":
        y = np.exp(y / 3)
    estimator = ensemble.HistGradientBoostingRegressor(loss=loss, max_iter=50, random_state=0).fit(x, y)
    x_test, _ = data(seed=3, missing=True)
    exact(export.from_sklearn(estimator).load(), x_test, estimator.predict(x_test))


def test_linear_pipeline_with_scaler_is_float64_exact():
    x, y = data()
    x[:, 0] = x[:, 0] * 500 + 60000  # price-level feature: float32 would lose it
    fitted = pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.Ridge(alpha=0.3)).fit(x, y)
    model = export.from_sklearn(fitted).load()
    assert model.preprocess
    x_test = x[:50] + 0.25
    np.testing.assert_allclose(model.predict(x_test)[:, 0], fitted.predict(x_test), rtol=1e-12, atol=1e-12)


def test_logistic_regression_probabilities():
    x, y = data()
    binary = linear_model.LogisticRegression().fit(x, y > 0)
    np.testing.assert_allclose(export.from_sklearn(binary).load().predict(x)[:, 0], binary.predict_proba(x)[:, 1],
                               rtol=1e-12, atol=1e-14)
    multi = linear_model.LogisticRegression(max_iter=500).fit(x, np.digitize(y, [-0.5, 0.5]))
    np.testing.assert_allclose(export.from_sklearn(multi).load().predict(x), multi.predict_proba(x),
                               rtol=1e-12, atol=1e-14)


def test_mlp_regressor():
    x, y = data()
    mlp = neural_network.MLPRegressor(hidden_layer_sizes=(16, 8), activation="tanh", max_iter=300,
                                      random_state=0).fit(x, y)
    np.testing.assert_allclose(export.from_sklearn(mlp).load().predict(x)[:, 0], mlp.predict(x), rtol=1e-12,
                               atol=1e-12)


# --- LightGBM: dump_model() format, checked against LightGBM's decision rule ---

def lightgbm_reference(dump, x):
    """LightGBM Tree::NumericalDecision and GBDT::Predict, written out."""
    zero = float(np.float32(1e-35))
    k = dump["num_tree_per_iteration"]
    raw = np.zeros((len(x), k))
    for t, info in enumerate(dump["tree_info"]):
        for i, row in enumerate(x):
            node = info["tree_structure"]
            while "split_feature" in node:
                value = row[node["split_feature"]]
                missing = node["missing_type"]
                if math.isnan(value) and missing != "NaN":
                    value = 0.0
                if (missing == "Zero" and -zero <= value <= zero) or (missing == "NaN" and math.isnan(value)):
                    node = node["left_child"] if node["default_left"] else node["right_child"]
                else:
                    node = node["left_child"] if value <= node["threshold"] else node["right_child"]
            raw[i, t % k] += node["leaf_value"]
    return raw


def random_lightgbm_tree(rng, depth, features):
    if depth == 0 or rng.random() < 0.2:
        return {"leaf_value": float(rng.normal())}
    return {"split_feature": int(rng.integers(features)), "threshold": float(rng.normal()), "decision_type": "<=",
            "default_left": bool(rng.random() < 0.5), "missing_type": str(rng.choice(["None", "Zero", "NaN"])),
            "left_child": random_lightgbm_tree(rng, depth - 1, features),
            "right_child": random_lightgbm_tree(rng, depth - 1, features)}


def lightgbm_dump(objective, per_iteration, trees=24, features=5, seed=0):
    rng = np.random.default_rng(seed)
    return {"name": "tree", "version": "v4", "num_class": per_iteration, "num_tree_per_iteration": per_iteration,
            "max_feature_idx": features - 1, "objective": objective, "average_output": False,
            "feature_names": [f"x{i}" for i in range(features)],
            "tree_info": [{"tree_index": i, "tree_structure": random_lightgbm_tree(rng, 6, features)}
                          for i in range(trees)]}


def edge_inputs(features=5, seed=1):
    x = np.random.default_rng(seed).normal(size=(300, features))
    x[::7, 1] = np.nan
    x[::5, 2] = 0.0
    x[::11, 3] = 1e-36
    x[::13, 0] = -0.0
    return x


@pytest.mark.parametrize("objective, k", [("regression", 1), ("binary sigmoid:0.7", 1), ("multiclass num_class:3", 3)])
def test_lightgbm_dump(objective, k):
    dump = lightgbm_dump(objective, k)
    model = export.from_lightgbm(json.dumps(dump)).load()
    x = edge_inputs()
    raw = lightgbm_reference(dump, x)
    if objective.startswith("binary"):
        expected = 1 / (1 + np.exp(-0.7 * raw))
    elif objective.startswith("multiclass"):
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        expected = e / e.sum(axis=1, keepdims=True)
    else:
        expected = raw
    np.testing.assert_allclose(model.predict(x), expected, rtol=1e-12, atol=1e-14)
    assert model.spec["inputs"]["names"] == dump["feature_names"]


def test_lightgbm_rejects_categorical_splits():
    dump = lightgbm_dump("regression", 1, trees=1)
    dump["tree_info"][0]["tree_structure"] = {"split_feature": 0, "threshold": "1||3", "decision_type": "==",
                                              "left_child": {"leaf_value": 1.0}, "right_child": {"leaf_value": 2.0}}
    with pytest.raises(ValueError, match="categorical"):
        export.from_lightgbm(dump)


# --- XGBoost: get_dump(dump_format="json") format ---

def random_xgboost_tree(rng, features, depth=5):
    counter = iter(range(10_000))

    def node(level):
        nodeid = next(counter)
        if level == depth or rng.random() < 0.2:
            return {"nodeid": nodeid, "leaf": float(np.float32(rng.normal()))}
        left, right = node(level + 1), node(level + 1)
        return {"nodeid": nodeid, "depth": level, "split": f"f{rng.integers(features)}",
                "split_condition": float(np.float32(rng.normal())), "yes": left["nodeid"], "no": right["nodeid"],
                "missing": left["nodeid"] if rng.random() < 0.5 else right["nodeid"], "children": [left, right]}

    return node(0)


def xgboost_reference(trees, x, margin):
    out = np.full(len(x), margin, dtype=np.float64)
    xf = x.astype(np.float32)
    for tree_ in trees:
        for i, row in enumerate(xf):
            node = tree_
            while "children" in node:
                children = {c["nodeid"]: c for c in node["children"]}
                value = row[int(node["split"][1:])]
                if np.isnan(value):
                    node = children[node["missing"]]
                else:
                    node = children[node["yes"] if value < np.float32(node["split_condition"]) else node["no"]]
            out[i] += node["leaf"]
    return out


def test_xgboost_dump_logistic():
    rng = np.random.default_rng(0)
    trees = [random_xgboost_tree(rng, 5) for _ in range(20)]
    model = export.from_xgboost([json.dumps(t) for t in trees], features=5, objective="binary:logistic",
                                base_score=0.3).load()
    x = edge_inputs()
    margin = math.log(0.3 / 0.7)
    expected = 1 / (1 + np.exp(-xgboost_reference(trees, x, margin)))
    np.testing.assert_allclose(model.predict(x)[:, 0], expected, rtol=1e-12, atol=1e-14)


def test_streaming_a_tabular_model():
    x, y = data()
    estimator = ensemble.GradientBoostingRegressor(n_estimators=10, random_state=0).fit(x, y)
    model = export.from_sklearn(estimator).load()
    live = model.stream()
    for row in x[:20]:
        np.testing.assert_array_equal(live.step(row), model.predict(row))


def test_large_batch_threads_are_exact():
    x, y = data(n=2000)
    forest = ensemble.RandomForestRegressor(n_estimators=40, random_state=0, n_jobs=1).fit(x, y)
    model = export.from_sklearn(forest).load()
    big = np.tile(x, (10, 1))
    try:
        montlok.set_threads(1)
        serial = model.predict(big)
        montlok.set_threads(8)
        parallel = model.predict(big)
    finally:
        montlok.set_threads(0)
    np.testing.assert_array_equal(serial, parallel)
    np.testing.assert_allclose(serial[:, 0], forest.predict(big), rtol=1e-12, atol=1e-12)


def random_ensemble(rng, features, decision, cast, trees=40):
    """Trees over every missing-value mode, with thresholds the inputs can hit exactly."""
    grid = np.array([-1.5, -0.25, 0.0, 0.5, 2.0])
    nodes = export._Nodes()
    roots = []
    for _ in range(trees):
        roots.append(len(nodes.feature))
        depth = int(rng.integers(1, 9))

        def build(level):
            if level == depth or (level > 0 and rng.random() < 0.2):
                return nodes.add(value=float(rng.normal()))
            flags = int(rng.integers(2)) | (int(rng.integers(3)) << 1)
            index = nodes.add(feature=int(rng.integers(features)), threshold=float(rng.choice(grid)), flags=flags)
            nodes.left[index] = build(level + 1)
            nodes.right[index] = build(level + 1)
            return index

        build(0)
    model = export.tree_ensemble(nodes, roots, [0] * len(roots), features=features, decision=decision,
                                 feature_cast=cast, base_score=0.25).load()
    specials = [np.nan, 0.0, -0.0, 1e-35, -1e-36, np.inf, -np.inf, *grid]
    x = np.where(rng.random((200, features)) < 0.4, rng.choice(specials, size=(200, features)),
                 rng.normal(size=(200, features)))
    return model, x


@pytest.mark.parametrize("decision", ["le", "lt"])
@pytest.mark.parametrize("cast", ["f64", "f32"])
def test_single_rows_and_blocks_agree(decision, cast):
    model, x = random_ensemble(np.random.default_rng(len(decision) + len(cast)), 5, decision, cast)
    block = model.predict(x)
    rows = np.stack([model.predict(row) for row in x])
    np.testing.assert_array_equal(block.view(np.int64), rows.view(np.int64))


def test_shared_nodes_and_infinite_thresholds():
    nodes = export._Nodes()
    split = nodes.add(feature=0, threshold=-np.inf, flags=1 | (1 << 1))
    nodes.left[split] = nodes.add(value=1.0)
    nodes.right[split] = nodes.add(value=2.0)
    model = export.tree_ensemble(nodes, [split, split], [0, 0], features=1).load()
    x = np.array([[-np.inf], [-1e308], [0.0], [np.inf], [np.nan]])
    np.testing.assert_array_equal(model.predict(x)[:, 0], [2.0, 4.0, 4.0, 4.0, 2.0])
    np.testing.assert_array_equal(np.stack([model.predict(row) for row in x])[:, 0], [2.0, 4.0, 4.0, 4.0, 2.0])


def test_latency_threads_give_the_same_bits():
    from concurrent.futures import ThreadPoolExecutor

    model, x = random_ensemble(np.random.default_rng(9), 6, "lt", "f32", trees=300)
    rows = x[:60]
    expected = np.stack([model.predict(row) for row in rows]).view(np.int64)
    try:
        for threads in (1, 3, 7):
            montlok.set_latency_threads(threads)
            assert montlok.latency_threads() == threads
            got = np.stack([model.predict(row) for row in rows]).view(np.int64)
            np.testing.assert_array_equal(got, expected)
            np.testing.assert_array_equal(model.predict(rows[:5]).view(np.int64), expected[:5])
            with ThreadPoolExecutor(4) as pool:
                concurrent = np.stack(list(pool.map(model.predict, rows))).view(np.int64)
            np.testing.assert_array_equal(concurrent, expected)
    finally:
        montlok.set_latency_threads(0)
    assert montlok.latency_threads() == 0
    with pytest.raises(ValueError):
        montlok.set_latency_threads(-1)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork()")
def test_forked_child_runs_with_latency_threads_set():
    import signal
    import time
    import warnings

    model, x = random_ensemble(np.random.default_rng(10), 6, "le", "f64", trees=300)
    expected = model.predict(x[0])
    montlok.set_latency_threads(2)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pid = os.fork()
        if pid == 0:
            code = 1
            try:
                same = np.array_equal(model.predict(x[0]), expected) and montlok.latency_threads() == 0
                montlok.set_latency_threads(1)
                same = same and np.array_equal(model.predict(x[0]), expected)
                montlok.set_latency_threads(0)
                code = 0 if same else 1
            finally:
                os._exit(code)
        deadline = time.monotonic() + 30
        while True:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if time.monotonic() > deadline:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                pytest.fail("the forked child did not finish")
            time.sleep(0.01)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        np.testing.assert_array_equal(model.predict(x[0]), expected)
    finally:
        montlok.set_latency_threads(0)
