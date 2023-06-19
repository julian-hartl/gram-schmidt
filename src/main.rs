use gram_schmidt::{Vector, Vector4};

fn main() {
    let basis = [
        Vector4::new([1.0, 1.0, 1.0, 1.0]),
        Vector4::new([0.0, 1.0, 0.0, 1.0]),
        Vector4::new([0.0, 0.0, 1.0, 1.0]),
        Vector4::new([0.0, 0.0, 0.0, 1.0]),
    ];
    const iterations: usize = 1000000;
    for _ in 0..iterations {
        Vector4::gram_schmidt(basis.to_vec());
    }
}
