use std::{fmt, ops};

use crate::math_traits::Mathable;

#[derive(Debug, Clone, Copy)]
pub struct Vector<const N: usize, K: Mathable = f32> {
	data: [K; N],
}

pub type Vec2 = Vector<2>;
pub type Vec3 = Vector<3>;

impl<const N: usize, K: Mathable> Default for Vector<N, K> {
	fn default() -> Vector<N, K> {
		Vector::<N, K> { data: [K::from(0.); N] }
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
}

impl<const N: usize, K: Mathable> ops::Index<usize> for Vector<N, K> {
	type Output = K;

	fn index(&self, index: usize) -> &Self::Output {
		&self.data[index]
	}
}

impl<const N: usize> ops::IndexMut<usize> for Vector<N> {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		&mut self.data[index]
	}
}

impl<const N: usize> PartialEq for Vector<N> {
	fn eq(&self, rhs: &Vector<N>) -> bool {
		for i in 0..N {
			if self.data[i] != rhs.data[i] {
				return false;
			}
		}
		return true;
	}
}

impl<const N: usize> ops::Add<Vector<N>> for Vector<N> {
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		let mut res = Self::new();
		for i in 0..N {	// Maybe you can do better with iterators or something? don't really care rn
			res.data[i] = self.data[i] + rhs.data[i];
		}
		res
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
	use super::*;

	#[test]
	fn test_add() {
		assert!(Vec2::from([1., 1.]) + Vec2::from([-1., -1.]) == Vec2::from([0., 0.]))
	}
}
