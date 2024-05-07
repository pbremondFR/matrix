#![allow(dead_code)]

use crate::{funcs::linear_combination, math_traits::{Determinant, Mathable, RealNumber}, Vector};
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

	pub fn set_row(&mut self, m: usize, vec: Vector<N, K>) -> Self {
		self[m] = vec;
		*self
	}

	pub fn get_column(self, n: usize) -> Vector<M, K> {
		let mut res = Vector::<M, K>::new();
		for i in 0..M {
			res[i] = self[i][n];
		}
		res
	}

	pub fn set_column(&mut self, n: usize, vec: Vector<M, K>) -> Self {
		for i in 0..M {
			self[n][i] = vec[i];
		}
		*self
	}

	pub fn transpose(&self) -> Matrix<N, M, K> {
		let mut res = Matrix::<N, M, K>::new();
		for i in 0..N {
			res.set_row(i, self.get_column(i));
		}
		res
	}

	pub fn is_row_echelon(&self) -> bool {
		let get_pivot = |i| -> usize {
			self[i].as_slice().iter().position(|&x| x != K::zero()).unwrap_or(N)
		};

		let mut pivot = get_pivot(0);
		for i in 1..M {
			let new_pivot = get_pivot(i);
			if new_pivot <= pivot && new_pivot != N {
				return false;
			}
			pivot = new_pivot;
		}
		return true;
	}

	fn swap_rows(&mut self, a_idx: usize, b_idx: usize) -> Self {
		assert!(a_idx < M && b_idx < N,
			"a_idx: {a_idx}, M: {M}, b_idx: {b_idx}, N: {N}");

		let tmp = self.get_row(a_idx);
		self.set_row(a_idx, self.get_row(b_idx));
		self.set_row(b_idx, tmp);
		*self
	}

	/*
	 * Uses Gauss-Jordan elimination to obtain the (reduced?) row-echelon form of a given matrix.
	 * I lost almost a day because some stupid website gave me wrong results for calculating this...
	 * Great video: https://www.youtube.com/watch?v=PTii4TBh9kQ
	 */
	pub fn row_echelon(&self) -> Self where K: RealNumber {

		/*
		 * I can either get the first non-zero pivot, or the biggest abs() one.
		 * Wikipedia claims that the max abs() is the most numerically stable, but my small
		 * tests don't really corroborate that.
		 */
		let find_first_non_zero = |col: &[K], start_idx: usize| {
			for i in start_idx..col.len() {
				if col[i] != K::zero() {
					return i;
				}
			}
			start_idx
		};

		let find_max_pivot = |col: &[K], start_idx: usize| -> usize {
			let (mut max_val, mut max_idx) = (K::zero(), start_idx);
			for (idx, val) in col.iter().enumerate().skip(start_idx) {
				if val.abs() > max_val {
					max_val = *val;
					max_idx = idx;
				}
			}
			max_idx
		};

		let mut res = self.clone();
		let mut i: usize = 0;
		let mut j: usize = 0;
		while i < M && j < N {
			let pivot = find_first_non_zero(res.get_column(j).as_slice(), i);
			if res[pivot][j] == K::zero() {
				j += 1;
			} else {
				if pivot != i {
					res.swap_rows(pivot, i);
				}
				let coef = res[i][j];
				res[i] /= coef;
				for k in 0..M {
					if res[k][j] == K::zero() || k == i {
						continue;
					}
					let diff = res[i] * res[k][j];
					res[k] -= diff;
				}
				i += 1;
				j += 1;
			}
		}
		res
	}

	pub fn submatrix(&self, rm_row: usize, rm_col: usize) -> Matrix<{M - 1}, {N - 1}, K> {
		if rm_row >= M || rm_col >= N {
			panic!("submatrix: index out-of-bounds: Matrix size is {M}x{N}, rm_row: {rm_row}, \
				 rm_col: {rm_col}");
		}
		let mut res = Matrix::<{M - 1}, {N - 1}, K>::default();
		let mut i: usize = 0;
		for (_, vec) in self.data.iter().enumerate().filter(|(idx,_)| *idx != rm_row) {
			let mut j: usize = 0;
			for (_, val) in vec.as_slice().iter().enumerate().filter(|(idx,_)| *idx != rm_col) {
				res[i][j] = *val;
				j += 1;
			}
			i += 1;
		}
		res
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

/*
 * Default implementation of determinant calculation using Gauss-Jordan elimination.
 * Faster specializations for 1x1, 2x2, 3x3 and 4x4 matrices are defined below.
 */
impl<const N: usize, K: Mathable> Determinant<N, K> for Matrix<N, N, K> {
	default fn det(&self) -> K {
		/*
		 * Using first non-zero instead of greatest abs() because complex numbers don't
		 * have one definitive way to order/compare them
		 */
		 let find_first_non_zero = |col: &[K], start_idx: usize| {
			for i in start_idx..col.len() {
				if col[i] != K::zero() {
					return i;
				}
			}
			start_idx
		};

		/*
		 * Use Gauss-Jordan here too: https://fr.wikipedia.org/wiki/%C3%89limination_de_Gauss-Jordan#D%C3%A9terminant
		 * det(A) = (-1)^p * n∏j=1 (A[k,j])
		 */
		let mut product = K::one();
		let mut row_swap_factor = K::one();
		let mut tmp = self.clone();
		let mut i: usize = 0;
		while i < N {
			let pivot = find_first_non_zero(tmp.get_column(i).as_slice(), i);
			if pivot != i {
				tmp.swap_rows(pivot, i);
				row_swap_factor = -row_swap_factor;
			}
			product *= tmp[i][i];
			let coef = tmp[i][i];
			tmp[i] /= coef;
			for k in 0..N {
				if tmp[k][i] == K::zero() || k == i {
					continue;
				}
				let diff = tmp[i] * tmp[k][i];
				tmp[k] -= diff;
			}
			i += 1;
		}
		return row_swap_factor * product;

	}
}

impl<K: Mathable> Determinant<1, K> for Matrix<1, 1, K> {
	fn det(&self) -> K {
		self[0][0]
	}
}

impl<K: Mathable> Determinant<2, K> for Matrix<2, 2, K> {
	fn det(&self) -> K {
		/*
		 * |a b|
		 * |c d|
		 *
		 * ad - bc
		 */

		// Bencharmking this vs a "normal" multiplication reveals mul_add is
		// INCREDIBLY slower! Two orders of magnitude! 430ps vs 4ns on my Ryzen 7 5700X

		// self[0][0].mul_add(self[1][1], -self[0][1] * self[1][0])
		self[0][0] * self[1][1] - self[0][1] * self[1][0]
	}
}

impl<K: Mathable> Determinant<3, K> for Matrix<3, 3, K> {
	fn det(&self) -> K {
		/*
		 * |a b c|
		 * |d e f|
		 * |g h i|
		 *
		 * aei + bfg + cdh - ceg - bdi - afh
		 */
		self[0][0] * self[1][1] * self[2][2]	// aei
		+ self[0][1] * self[1][2] * self[2][0]	// bfg
		+ self[0][2] * self[1][0] * self[2][1]	// cdh
		- self[0][2] * self[1][1] * self[2][0]	// ceg
		- self[0][1] * self[1][0] * self[2][2]	// bdi
		- self[0][0] * self[1][2] * self[2][1]	// afh
	}
}

// Brainless hard-coded thing. TODO: stop plagiarizing Unreal Engine
// Maybe make an N-sized implementation to flex, if that's not too hard?
impl<K: Mathable> Determinant<4, K> for Matrix<4, 4, K> {
	fn det(&self) -> K {
		let temp0 = self[2][2] * self[3][3] - self[2][3] * self[3][2];
		let temp1 = self[1][2] * self[3][3] - self[1][3] * self[3][2];
		let temp2 = self[1][2] * self[2][3] - self[1][3] * self[2][2];
		let temp3 = self[0][2] * self[3][3] - self[0][3] * self[3][2];
		let temp4 = self[0][2] * self[2][3] - self[0][3] * self[2][2];
		let temp5 = self[0][2] * self[1][3] - self[0][3] * self[1][2];

		self[0][0] * (self[1][1] * temp0 - self[2][1] * temp1 + self[3][1] * temp2)
		- self[1][0] * (self[0][1] * temp0 - self[2][1] * temp3 + self[3][1] * temp4)
		+ self[2][0] * (self[0][1] * temp1 - self[1][1] * temp3 + self[3][1] * temp5)
		- self[3][0] * (self[0][1] * temp2 - self[1][1] * temp4 + self[2][1] * temp5)
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
impl<const M: usize, const N: usize, K: Mathable + fmt::Display> fmt::Display for Matrix<M, N, K> {
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
	use crate::{macros::assert_approx_eq, vector};
	use super::*;

	macro_rules! assert_matrices_approx_equal {
		($m1:expr, $m2:expr, $epsilon:expr) => {{
			let m1 = $m1;
			let m2 = $m2;
			let epsilon = $epsilon;

			assert_eq!(m1.shape(), m2.shape());
			let (m, n) = m1.shape();

			for i in 0..m {
				for j in 0..n {
					assert!((m1[i][j] - m2[i][j]).abs() < epsilon,
						"Matrices differ in coordinates {i},{j} with epsilon {epsilon}:\n{m1}\n\n{m2}");
				}
			}
		}};
	}

	#[test]
	fn sucka_ma_balls() {
		let fuck = Matrix::from([
			[0.0, 1.0, 2.0, 4.0,  5.0],
			[0.0, 2.0, 3.0, 5.0,  7.0],
			[2.0, 4.0, 4.0, 2.0,  0.0],
			[3.0, 7.0, 9.0, 11.0, 13.0]
		]);
		let you = Matrix::from([
			[1.0, 0.0, 0.0, 0.0,  1.0],
			[0.0, 1.0, 0.0, 0.0,  9.0],
			[0.0, 0.0, 1.0, 0.0, -12.0],
			[0.0, 0.0, 0.0, 1.0,  5.0]
		]);
		assert_matrices_approx_equal!(fuck.row_echelon(), you, f32::EPSILON);

		let fuck = Matrix::from([
			[3.0, 4.0, 9.0],
			[2.0, 5.0, 1.0],
			[9.0, 12.0, 27.0]
		]);
		let you = Matrix::from([
			[1.0, 0.0,  41.0/7.0],
			[0.0, 1.0, -15.0/7.0],
			[0.0, 0.0,  0.0     ]
		]);
		assert_eq!(fuck.row_echelon(), you);
	}

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

		let u = Matrix::from([
			[1.0, 2.0, 3.0, 4.0, 5.0],
			[0.0, 0.0, 2.0, 4.0, 5.0],
			[0.0, 0.0, 0.0, 1.0, 5.0],
			[0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 0.0, 0.0],
		]);
		assert!(u.is_row_echelon());

		let u = Matrix::from([
			[1.0, 2.0, 3.0, 4.0, 5.0],
			[0.0, 0.0, 2.0, 4.0, 5.0],
			[0.0, 0.0, 0.0, 0.0, 0.0],
			[0.0, 0.0, 0.0, 1.0, 5.0],
			[0.0, 0.0, 0.0, 0.0, 0.0],
		]);
		assert_ne!(u.is_row_echelon(), true);

		let u = Matrix::from([
			[1.0, 2.0, 3.0, 4.0, 5.0],
			[0.0, 0.0, 2.0, 4.0, 5.0],
			[0.0, 0.0, 0.0, 1.0, 5.0],
			[0.0, 0.0, 0.0, 1.0, 5.0],
			[0.0, 0.0, 0.0, 0.0, 0.0],
		]);
		assert_ne!(u.is_row_echelon(), true);
	}

	#[test]
	fn swap_rows() {
		let mut m = Matrix::from([
			[1.0, 2.0],
			[3.0, 4.0],
		]);
		let expected = Matrix::from([
			[3.0, 4.0],
			[1.0, 2.0],
		]);
		assert_eq!(m.swap_rows(0, 1), expected);
	}

	#[test]
	fn row_echelon() {
		let u = Matrix::from([
			[ 2.0, -1.0,  0.0],
			[-1.0,  2.0, -1.0],
			[ 0.0, -1.0,  2.0]
		]);
		let expected = Matrix::from([
			[1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
			[0.0, 0.0, 1.0]
		]);
		assert_eq!(u.row_echelon(), expected);

		let u = Matrix::from([
			[1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
			[0.0, 0.0, 1.0]
		]);
		assert_eq!(u.row_echelon(), u);

		let u = Matrix::from([
			[1.0, 2.0],
			[3.0, 4.0]
		]);
		let expected = Matrix::from([
			[1.0, 0.0],
			[0.0, 1.0]
		]);
		assert_eq!(u.row_echelon(), expected);

		let u = Matrix::from([
			[1., 2.],
			[2., 4.],
		]);
		let expected = Matrix::from([
			[1., 2.],
			[0., 0.],
		]);
		assert_eq!(u.row_echelon(), expected);

		let u = Matrix::from([
			[8., 5.,  -2.,  4.,  28.],
			[4., 2.5,  20., 4., -4. ],
			[8., 5.,   1.,  4.,  17.],
		]);
		let expected = Matrix::from([
			[1.0, 0.625, 0.0, 0.0, -12.1666667],
			[0.0, 0.0,   1.0, 0.0, -3.6666667 ],
			[0.0, 0.0,   0.0, 1.0,  29.5      ],
		]);
		assert_matrices_approx_equal!(u.row_echelon(), expected, 0.0000001);

		let u = Matrix::from([
			[ 2.0, -1.0,  0.0],
			[ 0.0, -1.0,  2.0],
			[-1.0,  2.0, -1.0],
		]);
		let expected = Matrix::from([
			[1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
			[0.0, 0.0, 1.0]
		]);
		assert_eq!(u.row_echelon(), expected);
	}

	#[test]
	fn determinant() {
		let u = Matrix::from([
			[ 1., -1.],
			[-1., 1.],
		]);
		assert_eq!(u.det(), 0.0);

		let u = Matrix::from([
			[4., 2.],
			[2., 1.],
		]);
		assert_eq!(u.det(), 0.0);

		let u = Matrix::from([
			[2., 0., 0.],
			[0., 2., 0.],
			[0., 0., 2.],
		]);
		assert_eq!(u.det(), 8.0);

		let u = Matrix::from([
			[8., 5., -2.],
			[4., 7., 20.],
			[7., 6., 1.],
		]);
		assert_eq!(u.det(), -174.0);

		let u = Matrix::from([
			[ 8., 5., -2., 4.],
			[ 4., 2.5, 20., 4.],
			[ 8., 5., 1., 4.],
			[28., -4., 17., 1.],
		]);
		assert_eq!(u.det(), 1032.0);

		let u = Matrix::from([
			[ 18.0,  3.0,  11.0, -11.0, -10.0, -7.0],
			[ -3.0,  2.0, -11.0, -19.0, -16.0, -5.0],
			[ 16.0, 14.0,   9.0,   0.0,   2.0, 15.0],
			[-14.0,  8.0,  -5.0, -12.0, -11.0, 19.0],
			[ -7.0, 10.0,   9.0,  11.0,  12.0, 16.0],
			[-14.0, 14.0,   0.0,  -5.0,  -8.0,  3.0],
		]);
		assert_approx_eq!(u.det(), 9916537.0, 0.00000001);
	}

	#[test]
	fn submatrix() {
		let u = Matrix::from([
			[1., 2., 3.],
			[4., 5., 6.],
			[7., 8., 9.],
		]);
		let expected = Matrix::from([
			[5., 6.],
			[8., 9.],
		]);
		assert_eq!(u.submatrix(0, 0), expected);

		let u = Matrix::from([
			[1., 2., 3.],
			[4., 5., 6.],
			[7., 8., 9.],
		]);
		let expected = Matrix::from([
			[1., 2.],
			[4., 5.],
		]);
		assert_eq!(u.submatrix(2, 2), expected);

		let u = Matrix::from([
			[1., 2., 3.],
			[4., 5., 6.],
			[7., 8., 9.],
		]);
		let expected = Matrix::from([
			[1., 3.],
			[7., 9.],
		]);
		assert_eq!(u.submatrix(1, 1), expected);

		let u = Matrix::from([
			[1., 2., 3.],
			[4., 5., 6.],
			[7., 8., 9.],
		]);
		let expected = Matrix::from([
			[2., 3.],
			[5., 6.],
		]);
		assert_eq!(u.submatrix(2, 0), expected);
	}

	#[test]
	#[should_panic]
	fn submatrix_panic() {
		let u = Matrix::from([
			[1., 2., 3.],
			[4., 5., 6.],
			[7., 8., 9.],
		]);
		u.submatrix(3, 0);
	}

	extern crate test;
	use test::Bencher;
	use rand::distributions::{Distribution, Uniform};
	#[bench]
	fn benchmark_det(b: &mut Bencher) {
		let mut rng = rand::thread_rng();
		let die = Uniform::from(-200.0_f32..200.0_f32);
		let u = Matrix::from([
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
		]);
		b.iter(|| {
			let n = test::black_box(42);	// Trick compiler into not optimizing everything away
			u.det()
		})
	}
}
