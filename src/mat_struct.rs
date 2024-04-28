#![allow(dead_code)]

use crate::{math_traits::Mathable, Vector};
use std::{fmt, ops};

#[derive(Debug, Clone, Copy)]
/* M lines, N columns
 *    n
 *  ┌─────────────►
 * m│1;1  1;2  1;3
 *  |2;1  2;2  2;3
 *  |3;1  3;2  3;3
 *  ▼
 */
pub struct Matrix<const M: usize, const N: usize, K: Mathable = f32> {
	data: [Vector::<N, K>; M],
}

pub type Mat2 = Matrix<2, 2>;
pub type Mat3 = Matrix<3, 3>;

macro_rules! matrix_impl {
	($($body:tt)*) => {
		impl<const M: usize, const N: usize, K: Mathable> $($body)*
	};
}

// idk if I really want to use this trick. Looks like I'm too C++ minded? GIVE ME BACK #DEFINE
matrix_impl!( Default for Matrix<M, N, K> {
	fn default() -> Self {
		Self { data: [Vector::<N, K>::new(); M] }
	}
});

impl<const M: usize, const N: usize, K: Mathable> Matrix<M, N, K> {
	// Doesn't work, the fuck?
	// type VecType = Vector::<N, K>;

	pub fn new() -> Self {
		Self { ..Default::default() }
	}

	pub fn from(array: [[K; N]; M]) -> Self {
		let mut ret = Self::new();
		for (idx, &vec) in array.iter().enumerate() {
			ret.data[idx] = Vector::from(vec);
		}
		ret
	}
}

impl<const M: usize, const N: usize, K: Mathable> ops::Index<usize> for Matrix<M, N, K> {
	type Output = Vector::<N, K>;

	fn index(&self, index: usize) -> &Self::Output {
		&self.data[index]
	}
}

impl<const M: usize, const N: usize, K: Mathable> ops::IndexMut<usize> for Matrix<M, N, K> {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		&mut self.data[index]
	}
}

impl<const M: usize, const N: usize, K: Mathable> PartialEq for Matrix<M, N, K> {
	fn eq(&self, rhs: &Self) -> bool {
		for (i, &x) in rhs.data.iter().enumerate() {
			if self[i] != x {
				return false;
			}
		}
		return true;
	}
}

impl<const M: usize, const N: usize, K: Mathable> ops::Add<Matrix<M, N, K>> for Matrix<M, N, K> {
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		let mut res = Self::new();
		for i in 0..M {	// Maybe you can do better with iterators or something? don't really care rn
			res.data[i] = self.data[i] + rhs.data[i];
		}
		res
	}
}

impl<const M: usize, const N: usize, K: Mathable>
ops::Sub<Matrix<M, N, K>> for Matrix<M, N, K> {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		self + -rhs
	}
}

impl<const M: usize, const N: usize, K, T> ops::Mul<T> for Matrix<M, N, K>
where
	K: Mathable,
	T: Mathable + std::convert::Into<K>
{
	type Output = Self;

	fn mul(self, rhs: T) -> Self {
		let mut res = self.clone();
		for i in 0..M {
			res[i] = res[i] * rhs;
		}
		res
	}
}

impl<const M: usize, const N: usize, K, T> ops::Div<T> for Matrix<M, N, K>
where
	K: Mathable,
	T: Mathable + std::convert::Into<K>
{
	type Output = Self;

	fn div(self, rhs: T) -> Self {
		let mut res = self.clone();
		for i in 0..M {
			res[i] = res[i] / rhs;
		}
		res
	}
}

impl<const M: usize, const N: usize, K: Mathable> ops::Neg for Matrix<M, N, K> {
	type Output = Self;

	fn neg(self) -> Self::Output {
		let mut res = self.clone();
		// I tried with iterators and it sucks andf I'm BAD AT RUST FFS
		for i in 0..M {
			res[i] = -res[i];
		}
		res
	}
}

// Looks like it also implements the to_string trait?
impl<const M: usize, const N: usize> fmt::Display for Matrix<M, N> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		let mut output = String::new();
		for i in 0..self.data.len() {
			let vec_str = self.data[i].to_string();
			output += &format!("|{}|", &vec_str[1..vec_str.len() - 1]);
			if i != self.data.len() - 1 {
				output += "\n";
			}
		}
		write!(f, "{}", output)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_add() {
		let foo = Matrix::<2, 2>::from([
			[1., 2.],
			[3., 4.],
		]);
		let bar = Matrix::<2, 2>::from([
			[7., 4.],
			[-2., 2.],
		]);
		let expected = Matrix::<2, 2>::from([
			[8., 6.],
			[1., 6.],
		]);
		assert!(foo + bar == expected);
	}

	#[test]
	fn test_substract() {
		let foo = Matrix::<2, 2>::from([
			[1., 2.],
			[3., 4.],
		]);
		let expected = Matrix::<2, 2>::from([
			[0., 0.],
			[0., 0.],
		]);
		assert!(foo - foo == expected);
	}

	#[test]
	fn test_mul() {
		let foo = Matrix::<2, 2>::from([
			[1., 2.],
			[3., 4.],
		]);
		let expected = Matrix::<2, 2>::from([
			[2., 4.],
			[6., 8.],
		]);
		assert!(foo * 2. == expected);
	}

	#[test]
	fn test_eq_neq() {
		let mut foo = Matrix::<2, 2>::from([
			[1., 2.],
			[3., 4.],
		]);
		assert!(foo == foo);
		let bar = foo;
		foo = foo * 2.;
		assert!(foo != bar);
	}
}
