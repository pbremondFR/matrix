#![allow(dead_code)]

use crate::{math_traits::Mathable, Vector};
use std::{fmt, ops};

#[derive(Debug)]
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

// impl<const K: usize> PartialEq for Matrix<K> {
// 	fn eq(&self, rhs: &Matrix<K>) -> bool {
// 		for i in 0..K {
// 			if self.data[i] != rhs[i] {
// 				return false;
// 			}
// 		}
// 		return true;
// 	}
// }

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
