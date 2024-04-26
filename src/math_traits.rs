pub trait Mathable: Copy {
	fn from(x: f32) -> Self;
}

impl Mathable for f32 {
	fn from(x: f32) -> Self {
		x
	}
}
