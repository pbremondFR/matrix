#![allow(dead_code)]

use std::{fmt, ops::{self}};

use crate::math_traits::{Mathable, Norm, RealNumber};

#[derive(Debug, Clone, Copy)]
pub struct Vector<const N: usize, K: Mathable = f32> {
	data: [K; N],
}

pub type Vec2 = Vector<2>;
pub type Vec3 = Vector<3>;

macro_rules! vector {
	($($elem:expr),*) => {{
		Vector::from([$($elem),*])
	}};
}

pub(crate) use vector;

impl<const N: usize, K: Mathable> Default for Vector<N, K> {
	fn default() -> Vector<N, K> {
		Vector::<N, K> { data: [K::default(); N] }
	}
}

impl<const N: usize, K: Mathable> Vector<N, K> {
	pub fn new() -> Self {
		Vector::<N, K> { ..Default::default() }
	}

	// NOTE: If needed, change this to accept a slice? This one has the advantage
	// of static array bounds checking
	pub fn from(array: [K; N]) -> Self {
		Vector::<N, K> { data: array }
	}

	pub fn from_slice(src: &[K]) -> Self {
		Self { data: src.try_into().expect("Bad slice length") }
	}

	/*
	 * "If you’re curious about vector spaces of complex numbers, and already know enough
	 * about complex numbers, you might want to look up the terms conjugate transpose,
	 * sesquilinear algebra, and Pre-Hilbert space."
	 *
	 * "Those interested in electronics, control systems in engineering,
	 * or quantum mechanics should definitely study complex numbers and
	 * sesquilinear algebra."
	 *
	 * TODO, I guess?
	 */
	pub fn dot(self, v: &Vector<N, K>) -> K {
		let mut res = self[0] * v[0];
		for i in 1..N {
			res = self[i].mul_add(v[i], res);
		}
		res
	}
}
impl<const N: usize, K> Norm<K> for Vector<N, K>
where K: Mathable + RealNumber
{
	default fn norm_1(self) -> K {
		// Would have been nice but don't want to implement whatever trait this is for K
		// self.data.into_iter().map(|x| x.abs()).sum()
		let mut res = self[0].abs();
		for i in 1..N {
			res += self[i].abs()
		}
		res
	}

	default fn norm(self) -> K {
		self.dot(&self).sqrt()
	}

	default fn norm_inf(self) -> K {
		let mut res = self[0].abs();
		for i in 1..N {
			res = res.max(self[i].abs());
		}
		res
	}
}

// Demonstration on how to specialize an impl so that I have sort of like SFNIAE but worse
// For testing, and bonuses I guess, even though I might not make them at all...
// impl<const N: usize, T: Mathable> Norm<T> for Vector<N, ComplexNum<T>>
// {
// 	fn norm_1(self) -> T {
// 		T::default()
// 	}
// 	fn norm(self) -> T {
// 		T::default()
// 	}
// 	fn norm_inf(self) -> T {
// 		T::default()
// 	}
// }

impl<const N: usize, K: Mathable> ops::Index<usize> for Vector<N, K> {
	type Output = K;

	fn index(&self, index: usize) -> &Self::Output {
		&self.data[index]
	}
}

impl<const N: usize, K: Mathable> ops::IndexMut<usize> for Vector<N, K> {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		&mut self.data[index]
	}
}

impl<const N: usize, K: Mathable> PartialEq for Vector<N, K> {
	fn eq(&self, rhs: &Self) -> bool {
		self.data == rhs.data
	}
}

impl<const N: usize, K: Mathable> ops::Add<Vector<N, K>> for Vector<N, K> {
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		let mut res = Self::new();
		for i in 0..N {	// Maybe you can do better with iterators or something? don't really care rn
			res.data[i] = self.data[i] + rhs.data[i];
		}
		res
	}
}

impl<const N: usize, K: Mathable> ops::Sub<Vector<N, K>> for Vector<N, K> {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		self + -rhs
	}
}

impl<const N: usize, K, T> ops::Mul<T> for Vector<N, K>
where
	K: Mathable + ops::Mul<T, Output = K>,
	T: Mathable
{
	type Output = Self;

	fn mul(self, rhs: T) -> Self {
		let mul: Vec<K> = self.data.iter().map(|&x| x * rhs).collect();
		Self::from_slice(&mul)
	}
}

impl<const N: usize, K, T> ops::Div<T> for Vector<N, K>
where
	K: Mathable + ops::Div<T, Output = K>,
	T: Mathable
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
		let res: Vec<K> = self.data.iter().map(|&v| v / rhs).collect();
        Self::from_slice(&res)
    }
}

impl<const N: usize, K: Mathable> ops::Neg for Vector<N, K> {
	type Output = Self;

	fn neg(self) -> Self::Output {
		let inverted: Vec<K> = self.data.iter().map(|&x| -x).collect();
		Self::from_slice(&inverted)
	}
}

// Looks like it also implements the to_string trait?
impl<const N: usize> fmt::Display for Vector<N> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		let mut output = String::new();
		for i in 0..self.data.len() {
			output += self.data[i].to_string().as_str();
			if i != self.data.len() - 1 {
				output += ", ";
			}
		}
		write!(f, "[{}]", output)
	}
}

#[cfg(test)]
mod tests {
	use num_traits::abs_sub;

	use super::*;

	#[test]
	fn vec_f64() {
		Vector::<3, f64>::from([1.0, 2.0, 3.0]);
	}

	#[test]
	fn test_add() {
		assert!(Vec2::from([1., 1.]) + Vec2::from([-1., -1.]) == Vec2::from([0., 0.]))
	}

	#[test]
	fn test_substract() {
		let test = Vec2::from([42., 42.]);
		assert!(test - test == Vec2::from([0., 0.]))
	}

	#[test]
	fn test_mul() {
		let foo = vector!(1., 1.);
		let bar = foo * 2.;
		assert!(bar == vector!(2., 2.));
		let bar = bar * 0.;
		assert!(bar == vector!(0., 0.));
	}

	#[test]
	fn test_eq_neq() {
		let test = Vec2::from([42., 42.]);
		assert!(test == test);
		let test2 = -test;
		assert!(test != test2);
	}

	#[test]
	fn test_dot_product() {
		let u = Vector::from([0., 0.]);
		let v = Vector::from([1., 1.]);
		assert_eq!(u.dot(&v), 0.0);

		let u = Vector::from([1., 1.]);
		let v = Vector::from([1., 1.]);
		assert_eq!(u.dot(&v), 2.0);


		let u = Vector::from([-1.0, 6.0]);
		let v = Vector::from([3.0, 2.0]);
		assert_eq!(u.dot(&v), 9.0);
	}

	#[test]
	fn test_norm() {
		assert_eq!(vector!(0., 1.).norm(), 1.);
		assert!(abs_sub(vector!(42., 42.).norm(), 59.39696961966999) < 0.00000000000001);

		let u = vector!(0.0, 0.0, 0.0);
		assert_eq!(u.norm_1(), 0.0);
		assert_eq!(u.norm(), 0.0);
		assert_eq!(u.norm_inf(), 0.0);

		let u = vector!(1.0, 2.0, 3.0);
		assert_eq!(u.norm_1(), 6.0);
		assert_eq!(u.norm(), 3.7416573867739413);
		assert_eq!(u.norm_inf(), 3.0);

		let u = vector!(-1.0, -2.0);
		assert_eq!(u.norm_1(), 3.0);
		assert_eq!(u.norm(), 2.23606797749979);
		assert_eq!(u.norm_inf(), 2.0);
	}
}
