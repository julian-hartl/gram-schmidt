use gram_schmidt::{Vector, Vector4};
use criterion::{criterion_group, criterion_main, Criterion, black_box};

fn gram_schmit_benchmark(c: &mut Criterion) {

    c.bench_function("gram_schmidt", |b| b.iter(|| {
        let mut basis = black_box(
            [
                Vector4::new([1.0, 1.0, 1.0, 1.0]),
                Vector4::new([0.0, 1.0, 0.0, 1.0]),
                Vector4::new([0.0, 0.0, 1.0, 1.0]),
                Vector4::new([0.0, 0.0, 0.0, 1.0]),
            ].to_vec()
        );
        Vector4::gram_schmidt(&mut basis)
    }));
}

criterion_group!(benches, gram_schmit_benchmark);
criterion_main!(benches);