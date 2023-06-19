#![allow(clippy::needless_return)]

use std::iter::Sum;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};
use std::ptr;

pub trait Vector where
    Self: Sized
    + Index<usize, Output=f64>
    + IndexMut<usize>
    + Clone
    + Mul<f64, Output=Self>
    + Add<Output=Self>
    + Sub<Output=Self>
    + Div<f64, Output=Self>
    + Sum<Self> {
    const DIM: usize;

    fn get_component(&self, index: usize) -> f64;

    fn get_components_mut(&mut self) -> &mut [f64];

    fn dot_product(v1: &Self, v2: &Self) -> f64 {
        let mut sum = 0.0;
        for i in 0..Self::DIM {
            sum += v1[i] * v2[i];
        }
        return sum;
    }

    fn scale_with_dot_prod(&mut self, v2: &Self) {
        for i in 0..Self::DIM {
            self[i] = self[i] * self[i] * v2[i];
        }
    }

    #[inline(never)]
    fn gram_schmidt(
        basis: &mut Vec<Self>,
    ) {
        basis[0].normalize();
        for index in 1..basis.len() {
            let (first_half, second_half) = basis.split_at_mut(index);
            let a = &mut second_half[0];
            let sum: Self = first_half
                .iter()
                .cloned()
                .map(|b|
                    {
                        let dot = Vector::dot_product(a, &b);
                        b.scale(dot)
                    }
                )
                .sum();
            let mut c: Self = a.clone() - sum;
            c.normalize();
            // No conflict here because b is in second_half
            *a = c;
        }
    }

    fn length(&self) -> f64 {
        return Self::dot_product(self, self).sqrt();
    }

    fn normalize(&mut self) {
        let len = self.length();
        self.get_components_mut().iter_mut().for_each(|c| *c /= len);
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

        impl IndexMut<usize> for $name {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                return &mut self.components[index];
            }
        }

        impl Vector for $name {

            const DIM: usize = $dim;

            fn get_component(&self, index: usize) -> f64 {
                return self.components[index];
            }

            fn get_components_mut(&mut self) -> &mut [f64] {
                return &mut self.components;
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
    fn scale_with_dot_prod() {
        let mut v1 = Vector4::new([1.0, 2.0, 3.0, 6.0]);
        let v2 = Vector4::new([3.0, 4.0, 5.0, 7.0]);
        v1.scale_with_dot_prod(&v2);
        assert_eq!(v1, Vector4::new([3.0, 16.0, 45.0, 252.0]));
    }

    #[test]
    fn test_normalize() {
        let mut v1 = Vector4::new([4.0, 4.0, 4.0, 4.0]);
        v1.normalize();
        assert_eq!(v1, Vector4::new([4.0 / 8.0, 4.0 / 8.0, 4.0 / 8.0, 4.0 / 8.0]));
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
        let mut basis = vec![
            Vector4::new([1.0, 1.0, 1.0, 1.0]),
            Vector4::new([0.0, 1.0, 0.0, 1.0]),
            Vector4::new([0.0, 0.0, 1.0, 1.0]),
            Vector4::new([0.0, 0.0, 0.0, 1.0]),
        ];
        Vector4::gram_schmidt(&mut basis);
        assert_eq!(vec![
            Vector4::new([0.5, 0.5, 0.5, 0.5]),
            Vector4::new([-0.5, 0.5, -0.5, 0.5]),
            Vector4::new([-0.5, -0.5, 0.5, 0.5]),
            Vector4::new([0.5, -0.5, -0.5, 0.5]),
        ], basis);
    }
}