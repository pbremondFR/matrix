use std::{fmt, ops};

#[derive(Debug, Clone, Copy)]
pub struct Vector<const K: usize> {
	data: [f32; K],
}

pub type Vec2 = Vector<2>;
pub type Vec3 = Vector<3>;

impl<const K: usize> Default for Vector<K> {
	fn default() -> Self {
		Vector::<K> { data: [0.0; K] }
	}
}

impl<const K: usize> Vector<K> {
	pub fn new() -> Self {
		Vector::<K> { ..Default::default() }
	}

	// NOTE: If needed, change this to accept a slice? This one has the advantage
	// of static array bounds checking
	pub fn from(array: [f32; K]) -> Self {
		Vector::<K> { data: array }
	}
}

impl<const K: usize> ops::Index<usize> for Vector<K> {
	type Output = f32;

	fn index(&self, index: usize) -> &Self::Output {
		&self.data[index]
	}
}

impl<const K: usize> ops::IndexMut<usize> for Vector<K> {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		&mut self.data[index]
	}
}

impl<const K: usize> PartialEq for Vector<K> {
	fn eq(&self, rhs: &Vector<K>) -> bool {
		for i in 0..K {
			if self.data[i] != rhs.data[i] {
				return false;
			}
		}
		return true;
	}
}

impl<const K: usize> ops::Add<Vector<K>> for Vector<K> {
	type Output = Self;

	fn add(self, rhs: Self) -> Self {
		let mut res = Self::new();
		for i in 0..K {	// Maybe you can do better with iterators or something? don't really care rn
			res.data[i] = self.data[i] + rhs.data[i];
		}
		res
	}
}

// Looks like it also implements the to_string trait?
impl<const K: usize> fmt::Display for Vector<K> {
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
