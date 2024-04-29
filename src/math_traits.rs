use num_traits::*;

pub trait Mathable: Copy + Signed + NumAssignOps + Default {
	fn mul_add(self, a: Self, b: Self) -> Self;
	fn sqrt(self) -> f32;
}

impl Mathable for f32 {
	fn mul_add(self, a: Self, b: Self) -> Self {
		self.mul_add(a, b)
	}
	fn sqrt(self) -> f32 {
		self.sqrt()
	}
}

impl Mathable for f64 {
	fn mul_add(self, a: Self, b: Self) -> Self {
		self.mul_add(a, b)
	}
	fn sqrt(self) -> f32 {
		self.sqrt() as f32
	}
}

use std::ops::{Sub, Add, Mul};

use crate::{Matrix, Vector};

// Trying out a default implementation, but it might be too much for complex numbers
// or vectors and matrices? By only using Clone and not Copy it should be fine though
pub trait PlsGiveSNFIAE<V>
where
	V: Clone + Sub<Output = V> + Add<Output = V> + Mul<f32, Output = V>
{
	fn lerp(u: V, v: V, t: f32) -> V {
		u.clone() + ((v - u) * t)
	}
}

impl PlsGiveSNFIAE<f32> for f32 {
	fn lerp(u: f32, v: f32, t: f32) -> f32 {
		t.mul_add(v - u, u)
	}
}

impl<const N: usize, K> PlsGiveSNFIAE<Vector<N, K>> for Vector<N, K>
where
	K: Mathable + Mul<f32, Output = K>
{
	// Default impl
}

impl<const M: usize, const N: usize, K> PlsGiveSNFIAE<Matrix<M, N, K>> for Matrix<M, N, K>
where
	K: Mathable + Mul<f32, Output = K>
{
	// Default impl
}

#[cfg(test)]
mod tests {
	use crate::vector;

use super::*;

	#[test]
	fn test_lerp_f32() {
		assert_eq!(f32::lerp(0., 1., 0.), 0.);
		assert_eq!(f32::lerp(0., 1., 1.), 1.);
		assert_eq!(f32::lerp(0., 1., 0.5), 0.5);
		assert_eq!(f32::lerp(0., 1., 2.0), 2.0);
		assert_eq!(f32::lerp(0., 1., -2.0), -2.0);
		assert!(abs_sub(f32::lerp(21., 42., 0.3), 27.3) < 0.00001);
		assert_eq!(f32::lerp(21., 42., -0.3), 14.7);
	}

	#[test]
	fn test_lerp_vector() {
		let a = Vector::from([2., 1.]);
		let b = Vector::from([4., 2.]);
		let expected = vector!(2.6, 1.3);
		let result = Vector::lerp(a, b, 0.3);
		assert!(abs_sub(result[0], expected[0]) < 0.00000000000001);
		assert!(abs_sub(result[1], expected[1]) < 0.00000000000001);
	}

	#[test]
	fn test_lerp_matrix() {
		let a = Matrix::from([
			[2., 1.],
			[3., 4.]
		]);
		let b = Matrix::from([
			[20., 10.],
			[30., 40.]
		]);
		let result = Matrix::lerp(a, b, 0.5);
		let expected = Matrix::from([
			[11.0, 5.5],
			[16.5, 22.0]
		]);
		assert!((result[0] - expected[0]).norm() < 0.00000000000001);
		assert!((result[1] - expected[1]).norm() < 0.00000000000001);
	}
}
