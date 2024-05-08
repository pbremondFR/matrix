use num_traits::*;
use std::ops::{Sub, Add, Mul};
use crate::{Matrix, Vector};

pub trait Mathable: Copy + Signed + NumAssignOps + Default {
	fn mul_add(self, a: Self, b: Self) -> Self {
		(self * a) + b
	}
	fn abs(self) -> Self;
	fn max(self, other: Self) -> Self;
	fn sqrt(self) -> Self;	// Don't bother implementing it for complex numbers!
}

impl Mathable for f32 {
	fn mul_add(self, a: Self, b: Self) -> Self {
		// Holy shit performance from this is LITERALLY 100 times worse. Just let the
		// compiler call the fused multiply-accumulate for you.
		// For some reason, at leat on my Ryzen 7 5700X, this does not work well AT ALL.
		self.mul_add(a, b)
	}
	fn abs(self) -> Self {
		self.abs()
	}
	fn max(self, other: Self) -> Self {
		self.max(other)
	}
	fn sqrt(self) -> Self {
		self.powf(0.5)	// For some reason, I'm allowed to use pow but not sqrt?!
	}
}

impl Mathable for f64 {
	fn mul_add(self, a: Self, b: Self) -> Self {
		self.mul_add(a, b)
	}
	fn abs(self) -> Self {
		self.abs()
	}
	fn max(self, other: Self) -> Self {
		self.max(other)
	}
	fn sqrt(self) -> Self {
		self.powf(0.5)	// For some reason, I'm allowed to use pow but not sqrt?!
	}
}

pub trait RealNumber: Mathable + PartialOrd {}
impl RealNumber for f32 {}
impl RealNumber for f64 {}

// Trying out a default implementation, but it might be too much for complex numbers
// or vectors and matrices? By only using Clone and not Copy it should be fine though
// TODO: Take in references for performance reasons?
pub trait Lerp<V>
where
	V: Clone + Sub<Output = V> + Add<Output = V> + Mul<f32, Output = V>
{
	fn lerp(u: V, v: V, t: f32) -> V {
		u.clone() + ((v - u) * t)
	}
}

impl Lerp<f32> for f32 {
	fn lerp(u: f32, v: f32, t: f32) -> f32 {
		t.mul_add(v - u, u)
	}
}

impl<const N: usize, K> Lerp<Vector<N, K>> for Vector<N, K>
where
	K: Mathable + Mul<f32, Output = K>
{
	// Default impl
}

impl<const M: usize, const N: usize, K> Lerp<Matrix<M, N, K>> for Matrix<M, N, K>
where
	K: Mathable + Mul<f32, Output = K>
{
	// Default impl
}

pub trait Norm<T>
where
	T: Mathable
{
	fn norm_1(self) -> T;
	fn norm(self) -> T;
	fn norm_inf(self) -> T;
}

#[cfg(test)]
mod tests {
	use crate::{macros::assert_approx_eq, vector};

use super::*;

	#[test]
	fn test_lerp_f32() {
		assert_eq!(f32::lerp(0., 1., 0.), 0.);
		assert_eq!(f32::lerp(0., 1., 1.), 1.);
		assert_eq!(f32::lerp(0., 1., 0.5), 0.5);
		assert_eq!(f32::lerp(0., 1., 2.0), 2.0);
		assert_eq!(f32::lerp(0., 1., -2.0), -2.0);
		assert_approx_eq!(f32::lerp(21., 42., 0.3), 27.3, 0.00001);
		assert_eq!(f32::lerp(21., 42., -0.3), 14.7);
	}

	#[test]
	fn test_lerp_vector() {
		let a = Vector::from([2., 1.]);
		let b = Vector::from([4., 2.]);
		let expected = vector!(2.6, 1.3);
		let result = Vector::lerp(a, b, 0.3);
		assert_approx_eq!(result[0], expected[0], 0.00000000000001);
		assert_approx_eq!(result[1], expected[1], 0.00000000000001);
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
