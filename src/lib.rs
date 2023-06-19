#![allow(clippy::needless_return)]

use std::iter::Sum;
use std::ops::{Add, Div, Index, Mul, Sub};

pub trait Vector where
    Self: Sized
    + Index<usize, Output=f64>
    + Clone
    + Mul<f64, Output=Self>
    + Add<Output=Self>
    + Sub<Output=Self>
    + Div<f64, Output=Self>
    + Sum<Self> {
    const DIM: usize;

    fn get_component(&self, index: usize) -> f64;

    fn dot_product(v1: &Self, v2: &Self) -> f64 {
        let mut sum = 0.0;
        for i in 0..Self::DIM {
            sum += v1[i] * v2[i];
        }
        return sum;
    }

    fn gram_schmidt(
        basis: Vec<Self>,
    ) -> Vec<Self> {
        let mut nb = Vec::with_capacity(basis.len());
        for (index, b) in basis.into_iter().enumerate() {
            if index == 0 {
                nb.push(b.normalize());
                continue;
            }

            let c =
                nb[0..index]
                    .iter()
                    .cloned()
                    .map(|v| {
                        let dot = Self::dot_product(&v, &b);
                        return v * dot;
                    })
                    .sum::<Self>() * -1. + b;
            nb.push(c.normalize());
        }
        return nb;
    }

    fn length(&self) -> f64 {
        return Self::dot_product(self, self).sqrt();
    }

    fn normalize(self) -> Self where Self: Sized {
        let len = self.length();
        return self / len;
    }

    fn scale(self, lambda: f64) -> Self {
        return self * lambda;
    }
}

macro_rules! vector {
    ($name:ident, $dim:expr) => {
        #[derive(Debug, PartialEq, Clone)]
        pub struct $name {
            pub components: [f64; $dim],
        }

        impl $name {
            pub const DIM: usize = $dim;

            pub fn new(components: [f64; Self::DIM]) -> Self {
                return Self { components };
            }

            pub fn empty() -> Self {
                return Self { components: [0.0; Self::DIM] };
            }
        }

        impl Add for $name {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                let mut components = [0.0; Self::DIM];
                for i in 0..Self::DIM {
                    components[i] = self.components[i] + rhs.components[i];
                }
                return Self { components };
            }
        }

        impl Sub for $name {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                let mut components = [0.0; Self::DIM];
                for i in 0..Self::DIM {
                    components[i] = self.components[i] - rhs.components[i];
                }
                return Self { components };
            }
        }

        impl Sum for $name {
            fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
                return iter.fold(
                    Self::empty(),
                    |a, b| a + b,
                );
            }
        }

        impl Mul<f64> for $name {
            type Output = Self;

            fn mul(self, rhs: f64) -> Self::Output {
                let mut components = [0.0; Self::DIM];
                for i in 0..Self::DIM {
                    components[i] = self.components[i] * rhs;
                }
                return Self { components };
            }
        }

        impl Div<f64> for $name {
            type Output = Self;

            fn div(self, rhs: f64) -> Self::Output {
                let mut components = [0.0; Self::DIM];
                for i in 0..Self::DIM {
                    components[i] = self.components[i] / rhs;
                }
                return Self { components };
            }
        }

        impl Index<usize> for $name {
            type Output = f64;

            fn index(&self, index: usize) -> &Self::Output {
                return &self.components[index];
            }
        }

        impl Vector for $name {

            const DIM: usize = $dim;

            fn get_component(&self, index: usize) -> f64 {
                return self.components[index];
            }
        }
    };
}

// Usage
vector!(Vector4, 4);
vector!(Vector3, 3);



#[cfg(test)]
mod vec3_test {
    use crate::{Vector, Vector4};

    #[test]
    fn test_dot_product() {
        let v1 = Vector4::new([1.0, 2.0, 3.0, 6.0]);
        let v2 = Vector4::new([3.0, 4.0, 5.0, 7.0]);
        assert_eq!(Vector4::dot_product(&v1, &v2), 68.0);
    }

    #[test]
    fn test_normalize() {
        let v1 = Vector4::new([4.0, 4.0, 4.0, 4.0]);
        assert_eq!(v1.normalize(), Vector4::new([4.0 / 8.0, 4.0 / 8.0, 4.0 / 8.0, 4.0 / 8.0]));
    }

    #[test]
    fn test_length() {
        let v1 = Vector4::new([4.0, 4.0, 4.0, 4.0]);
        let v2 = Vector4::new([3.0, 4.0, 5.0, 7.0]);
        assert_eq!(v1.length(), 8.0);
        assert_eq!(v2.length(), 99_f64.sqrt());
    }
}

#[cfg(test)]
mod grim_schmidt_test {
    use crate::{Vector, Vector4};

    #[test]
    fn basic_test() {
        let basis = [
            Vector4::new([1.0, 1.0, 1.0, 1.0]),
            Vector4::new([0.0, 1.0, 0.0, 1.0]),
            Vector4::new([0.0, 0.0, 1.0, 1.0]),
            Vector4::new([0.0, 0.0, 0.0, 1.0]),
        ];
        let nb = Vector4::gram_schmidt(basis.to_vec());
        assert_eq!(vec![
            Vector4::new([0.5, 0.5, 0.5, 0.5]),
            Vector4::new([-0.5, 0.5, -0.5, 0.5]),
            Vector4::new([-0.5, -0.5, 0.5, 0.5]),
            Vector4::new([0.5, -0.5, -0.5, 0.5]),
        ], nb);
    }
}