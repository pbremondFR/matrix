#![allow(dead_code)]

use crate::{funcs::linear_combination, math_traits::Mathable, Vector};
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

	pub fn as_slice(&self) -> &[Vector<N, K>] {
		&self.data
	}

	pub fn shape(&self) -> (usize, usize) {
		(M, N)
	}

	pub fn get_row(self, m: usize) -> Vector<N, K> {
		self[m]
	}

	pub fn set_row(&mut self, m: usize, vec: Vector<N, K>) {
		self[m] = vec;
	}

	pub fn get_column(self, n: usize) -> Vector<M, K> {
		let mut res = Vector::<M, K>::new();
		for i in 0..M {
			res[i] = self[i][n];
		}
		res
	}

	pub fn set_column(&mut self, n: usize, vec: Vector<M, K>) {
		for i in 0..M {
			self[n][i] = vec[i];
		}
	}

	pub fn transpose(&self) -> Matrix<N, M, K> {
		let mut res = Matrix::<N, M, K>::new();
		for i in 0..N {
			res.set_row(i, self.get_column(i));
		}
		res
	}

	pub fn is_row_echelon(&self) -> bool {
		let get_leading_entry = |i| -> usize {
			self[i].as_slice().iter().position(|&x| x != K::zero()).unwrap_or(N)
		};

		let mut leading_entry = get_leading_entry(0);
		for i in 1..M {
			let new_leading_entry = get_leading_entry(i);
			if new_leading_entry <= leading_entry && new_leading_entry != N {
				return false;
			}
			leading_entry = new_leading_entry;
		}
		return true;
	}
}

impl<const M: usize, K: Mathable> Matrix<M, M, K> {
	fn trace(&self) -> K {
		let mut res = self[0][0];
		for i in 1..M {
			res += self[i][i];
		}
		res
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

impl<const M: usize, const N: usize, K: Mathable> ops::Sub<Matrix<M, N, K>> for Matrix<M, N, K> {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self {
		self + -rhs
	}
}

// Matrix multiplication by number
impl<const M: usize, const N: usize, K, T> ops::Mul<T> for Matrix<M, N, K>
where
	K: Mathable + ops::Mul<T, Output = K>,
	T: Mathable
{
	type Output = Self;

	fn mul(self, rhs: T) -> Self::Output {
		let mut res = self.clone();
		for i in 0..M {
			res[i] = res[i] * rhs;
		}
		res
	}
}

// Matrix multiplication by vector
impl<const M: usize, const N: usize, K> ops::Mul<Vector<N, K>> for Matrix<M, N, K>
where
	K: Mathable
{
	type Output = Vector<M, K>;

	fn mul(self, vec: Vector<N, K>) -> Self::Output {
		let mut res = Vector::<M, K>::new();
		for i in 0..M {
			res[i] = vec.dot(&self[i]);
		}
		res
	}
}

// Matrix multiplication by other matrix
impl<const M: usize, const N: usize, const P: usize, K> ops::Mul<Matrix<N, P, K>> for Matrix<M, N, K>
where
	K: Mathable
{
	type Output = Matrix<M, P, K>;

	fn mul(self, matrix: Matrix<N, P, K>) -> Self::Output {
		// I think I'll leave matrix multiplication optimization to the mathematicians
		// and computer scientists.
		let mut res = Self::Output::new();
		for i in 0..M {
			for j in 0..P {
				res[i][j] = self.get_row(i).dot(&matrix.get_column(j));
			}
		}
		res
	}
}

impl<const M: usize, const N: usize, K, T> ops::Div<T> for Matrix<M, N, K>
where
	K: Mathable + ops::Div<T, Output = K>,
	T: Mathable
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
	use crate::vector;

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

	#[test]
	fn mul_vec() {
		let u = Matrix::from([
			[1., 0.],
			[0., 1.],
		]);
		let v = Vector::from([4., 2.]);
		assert_eq!(u * v, vector!(4.0, 2.0));

		let u = Matrix::from([
			[2., 0.],
			[0., 2.],
		]);
		let v = Vector::from([4., 2.]);
		assert_eq!(u * v, vector!(8.0, 4.0));

		let u = Matrix::from([
			[2., -2.],
			[-2., 2.],
		]);
		let v = Vector::from([4., 2.]);
		assert_eq!(u * v, vector!(4.0, -4.0));

		let u = Matrix::from([
			[1.0, -1.0, 2.0],
			[0.0, -3.0, 1.0]
		]);
		let v = Vector::from([2.0, 1.0, 0.0]);
		assert_eq!(u * v, vector!(1.0, -3.0));
	}

	#[test]
	fn mul_mat() {
		let u = Matrix::from([
			[1., 0.],
			[0., 1.],
		]);
		let v = Matrix::from([
			[1., 0.],
			[0., 1.],
		]);
		let expected = Matrix::from([
			[1., 0.],
			[0., 1.],
		]);
		assert_eq!(u * v, expected);

		let u = Matrix::from([
			[1., 0.],
			[0., 1.],
		]);
		let v = Matrix::from([
			[2., 1.],
			[4., 2.],
		]);
		let expected = Matrix::from([
			[2., 1.],
			[4., 2.],
		]);
		assert_eq!(u * v, expected);

		let u = Matrix::from([
			[3., -5.],
			[6.,  8.],
		]);
		let v = Matrix::from([
			[2., 1.],
			[4., 2.],
		]);
		let expected = Matrix::from([
			[-14., -7. ],
			[ 44.,  22.],
		]);
		assert_eq!(u * v, expected);
	}

	#[test]
	fn trace() {
		let u = Matrix::from([
			[1., 0.],
			[0., 1.],
		]);
		assert_eq!(u.trace(), 2.0);

		let u = Matrix::from([
			[ 2., -5., 0.],
			[ 4.,  3., 7.],
			[-2.,  3., 4.],
		]);
		assert_eq!(u.trace(), 9.0);

		let u = Matrix::from([
			[-2., -8.,  4.],
			[ 1., -23., 4.],
			[ 0.,  6.,  4.],
		]);
		assert_eq!(u.trace(), -21.0);
	}

	#[test]
	fn transpose() {
		let a = Matrix::from([
			[1.0, 3.0, 5.0],
			[2.0, 4.0, 6.0]
		]);
		let b = Matrix::from([
			[1.0, 2.0],
			[3.0, 4.0],
			[5.0, 6.0]
		]);
		assert_eq!(a.transpose(), b);
		assert_eq!(b.transpose(), a);

		let a = Matrix::from([
			[1.0, 2.0],
			[3.0, 4.0]
		]);
		let b = Matrix::from([
			[1.0, 3.0],
			[2.0, 4.0]
		]);
		assert_eq!(a.transpose(), b);
		assert_eq!(b.transpose(), a);

		let a = Matrix::from([
			[1.0, 2.0]
		]);
		let b = Matrix::from([
			[1.0],
			[2.0]
		]);
		assert_eq!(a.transpose(), b);
		assert_eq!(b.transpose(), a);
	}

	#[test]
	fn is_row_echelon() {
		let u = Matrix::from([
			[1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
			[0.0, 0.0, 1.0]
		]);
		assert!(u.is_row_echelon());

		let u = Matrix::from([
			[1.0, 2.0],
			[3.0, 4.0]
		]);
		assert_ne!(u.is_row_echelon(), true);
		let u = Matrix::from([
			[1.0, 0.0],
			[0.0, 1.0]
		]);
		assert!(u.is_row_echelon());

		let u = Matrix::from([
			[1., 2.],
			[2., 4.],
		]);
		assert_ne!(u.is_row_echelon(), true);
		let u = Matrix::from([
			[1., 2.],
			[0., 0.],
		]);
		assert!(u.is_row_echelon());

		let u = Matrix::from([
			[8., 5.,  -2.,  4.,  28.],
			[4., 2.5,  20., 4., -4. ],
			[8., 5.,   1.,  4.,  17.],
		]);
		assert_ne!(u.is_row_echelon(), true);
		let u = Matrix::from([
			[1.0, 0.625, 0.0, 0.0, -12.1666667],
			[0.0, 0.0,   1.0, 0.0, -3.6666667 ],
			[0.0, 0.0,   0.0, 1.0,  29.5      ],
		]);
		assert!(u.is_row_echelon());
	}
}
