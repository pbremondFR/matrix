use crate::Vector;
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
pub struct Matrix<const M: usize, const N: usize, K: Copy = f32> {
	data: [Vector::<N, K>; M],
}

pub type Mat2 = Matrix<2, 2>;
pub type Mat3 = Matrix<3, 3>;

macro_rules! matrix_impl {
    ($($body:tt)*) => {
        impl<const N: usize, const M: usize, K: Copy> $($body)*
    };
}

// idk if I really want to use this trick. Looks like I'm too C++ minded?
matrix_impl!( Default for Matrix<N, M, K> {
    fn default() -> Self {
        Self { data: [Vector::<N, K>::new(); M] }
    }
});


impl<const N: usize, const M: usize, K: Copy> Matrix<N, M, K> {
	pub fn new() -> Self {
		Self { ..Default::default() }
	}

	pub fn from(array: [[f32; N]; M]) -> Self {
		let mut ret = Self::new();
		array.iter().enumerate().map(|(i, x)| ret[i] = x);
		// One of these is more idiomatic, I guess. The other, more readable, I think.
		// Pick your poison.

		// for i in 0..array.len() {
		// 	ret.data[i] = Vector::<N>::from(array[i]);
		// }
		ret
	}
}

// impl<const N: usize, const M: usize, K> ops::Index<usize> for Matrix<N, M, K> {
// 	type Output = Vector::<K>;

// 	fn index(&self, index: usize) -> &Self::Output {
// 		&self.data[index]
// 	}
// }

// impl<const K: usize> ops::IndexMut<usize> for Matrix<K> {
// 	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
// 		&mut self.data[index]
// 	}
// }

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
