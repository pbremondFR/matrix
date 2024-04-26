use num_traits::*;

pub trait Mathable: Copy + Signed {
	fn from(x: f32) -> Self;
}

impl Mathable for f32 {
	fn from(x: f32) -> Self {
		x
	}
}
