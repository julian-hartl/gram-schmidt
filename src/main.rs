use gram_schmidt::{Vector, Vector4};

fn main() {

    const iterations: usize = 1000000;
    for _ in 0..iterations {
        let mut basis = [
            Vector4::new([1.0, 1.0, 1.0, 1.0]),
            Vector4::new([0.0, 1.0, 0.0, 1.0]),
            Vector4::new([0.0, 0.0, 1.0, 1.0]),
            Vector4::new([0.0, 0.0, 0.0, 1.0]),
        ].to_vec();
        Vector4::gram_schmidt(&mut basis);
    }
}
