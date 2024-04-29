#![allow(dead_code)]

use std::{fmt, ops::{self}};

use crate::math_traits::Mathable;

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

	pub fn norm(self) -> f32
	{
		let mut acc: K = self[0] * self[0];
		for i in 1..N {
			acc += self[i] * self[i];
		}
		acc.sqrt()
	}
}

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
	fn test_norm() {
		assert_eq!(vector!(0., 1.).norm(), 1.);
		assert!(abs_sub(vector!(42., 42.).norm(), 59.39696961966999) < 0.00000000000001);
	}
}
