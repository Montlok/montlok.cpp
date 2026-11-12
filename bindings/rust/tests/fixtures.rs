//! The cross-language fixtures of tests/data through the Rust binding.

use std::path::PathBuf;

fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

fn read_f64(name: &str) -> Vec<f64> {
    let bytes = std::fs::read(data().join(name)).expect("fixture");
    bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn check(name: &str) {
    let batch = 3;
    let model = montlok::Model::open(data().join(format!("{name}.montlok"))).expect("model");
    let info = model.info();
    let io = read_f64(&format!("{name}.io"));
    let (x, expected) = io.split_at(batch * info.input_len());
    assert_eq!(expected.len(), batch * info.outputs);
    let y = model.predict(x, batch).expect("predict");
    for (a, b) in y.iter().zip(expected) {
        assert!(
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12 * b.abs().max(1.0),
            "{name}: {a} != {b}"
        );
    }
    if info.streamable {
        let mut stream = model.stream(1).expect("stream");
        let mut last = Vec::new();
        for step in x[..info.input_len()].chunks(info.features) {
            last = stream.step(step).expect("step");
        }
        assert_eq!(last, y[..info.outputs].to_vec(), "{name}: stream equals predict");
    }
    assert!(model.spec_json().contains("montlok.model/1"));
}

#[test]
fn gru_fixture() {
    check("gru");
}

#[test]
fn gbr_fixture() {
    check("gbr");
}

#[test]
fn models_are_shared_across_threads() {
    let model = std::sync::Arc::new(montlok::Model::open(data().join("gru.montlok")).unwrap());
    let x = vec![0.5f32; model.info().input_len()];
    let expected = model.predict(&x, 1).unwrap();
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let model = model.clone();
            let x = x.clone();
            std::thread::spawn(move || model.predict(&x, 1).unwrap())
        })
        .collect();
    for handle in handles {
        assert_eq!(handle.join().unwrap(), expected);
    }
}

#[test]
fn errors_carry_status_and_message() {
    let err = montlok::Model::from_bytes(b"not a model").err().expect("garbage is rejected");
    assert_eq!(err.status, 3);
    assert!(err.message.contains("safetensors"), "{}", err.message);
    let model = montlok::Model::open(data().join("gbr.montlok")).unwrap();
    assert_eq!(model.predict(&[0.0; 3], 1).err().unwrap().status, montlok::SHAPE);
    assert!(montlok::families().contains(&"tree_ensemble".to_string()));
}

#[test]
fn micro_engine_warms_up_and_rejects_bad_time() {
    use montlok::micro::{Event, FeatureEngine, BOOK, REFERENCE};
    assert_eq!(montlok::micro::schema(), "xstock-observable-l1-v2-32");
    let mut engine = FeatureEngine::new(1024).unwrap();
    let t = 1_000_000_000_000_000i64;
    let ms = 1_000_000;
    engine.update(&Event { receive_ns: t, kind: REFERENCE, side: 0, a: 100.0, b: 100.02, c: 1.5, d: 0.0 }).unwrap();
    let book = |ns| Event { receive_ns: ns, kind: BOOK, side: 0, a: 99.99, b: 100.03, c: 30.0, d: 10.0 };
    engine.update(&book(t + ms)).unwrap();
    assert!(engine.snapshot().is_none());
    engine.update(&book(t + 2500 * ms)).unwrap();
    let features = engine.snapshot().expect("ready after 2 s");
    assert!((features[9] - 0.5).abs() < 1e-15, "imbalance_l1");
    assert!(engine.update(&book(t)).is_err());
    engine.reset();
    engine.update(&book(t)).unwrap();
}

#[test]
fn latency_threads_keep_results() {
    montlok::set_latency_threads(2).expect("start latency threads");
    assert_eq!(montlok::latency_threads(), 2);
    check("gbr");
    montlok::set_latency_threads(0).expect("stop latency threads");
    assert_eq!(montlok::latency_threads(), 0);
}
