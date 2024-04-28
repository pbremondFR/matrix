use num_traits::*;

pub trait Mathable: Copy + Signed + NumAssignOps + Default{
	fn mul_add(self, a: Self, b: Self) -> Self;
}

impl Mathable for f32 {
	fn mul_add(self, a: Self, b: Self) -> Self {
		self.mul_add(a, b)
	}
}

impl Mathable for f64 {
	fn mul_add(self, a: Self, b: Self) -> Self {
		self.mul_add(a, b)
	}
}
