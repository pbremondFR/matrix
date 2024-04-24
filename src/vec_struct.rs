use std::ops;

#[derive(Debug)]
pub struct Vector<const K: usize> {
	pub data: [f32; K],
}

pub type Vec3 = Vector<3>;
pub type Vec2 = Vector<2>;

impl<const K: usize> Default for Vector<K> {
	fn default() -> Self {
		Vector::<K> { data: [0.0; K] }
	}
}

impl<const K: usize> Vector<K> {
	pub fn new() -> Self {
		Vector::<K> { ..Default::default() }
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
	type Output = Vector<K>;

	fn add(self, rhs: Vector<K>) -> Vector<K> {
		let mut res = Vector::<K>::new();
		for i in 0..K {
			res.data[i] = self.data[i] + rhs.data[i];
		}
		res
	}
}

mod tests {
	use super::*;

	#[test]
	fn test_add() {
		assert_eq!(Vec2{data: [1., 1.]} + Vec2{data: [-1., -1.]}, Vec2{data: [42., 0.]});
	}
}
