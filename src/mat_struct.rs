#[derive(Debug)]
pub struct Matrix<const K: usize> {
	pub data: [[f32; K]; K],
}

pub type Mat3 = Matrix<3>;
pub type Mat2 = Matrix<3>;

impl<const K: usize> Default for Matrix<K> {
	fn default() -> Self {
		Matrix::<K> { data: [[0.0; K]; K] }
	}
}

impl<const K: usize> Matrix<K> {
	pub fn new() -> Self {
		Matrix::<K> { ..Default::default() }
	}
}
