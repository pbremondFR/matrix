use crate::Vector;
use std::{fmt, ops};

#[derive(Debug)]
pub struct Matrix<const K: usize> {
	data: [Vector::<K>; K],
}

pub type Mat2 = Matrix<2>;
pub type Mat3 = Matrix<3>;

impl<const K: usize> Default for Matrix<K> {
	fn default() -> Self {
		Matrix::<K> { data: [ Vector::<K>::new(); K] }
	}
}

impl<const K: usize> Matrix<K> {
	pub fn new() -> Self {
		Matrix::<K> { ..Default::default() }
	}

	pub fn from(array: [[f32; K]; K]) -> Self {
		let mut ret = Self::new();
		for i in 0..array.len() {
			ret.data[i] = Vector::<K>::from(array[i]);
		}
		ret
	}
}

impl<const K: usize> ops::Index<usize> for Matrix<K> {
	type Output = Vector::<K>;

	fn index(&self, index: usize) -> &Self::Output {
		&self.data[index]
	}
}

impl<const K: usize> ops::IndexMut<usize> for Matrix<K> {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		&mut self.data[index]
	}
}

impl<const K: usize> PartialEq for Matrix<K> {
	fn eq(&self, rhs: &Matrix<K>) -> bool {
		for i in 0..K {
			if self.data[i] != rhs[i] {
				return false;
			}
		}
		return true;
	}
}
