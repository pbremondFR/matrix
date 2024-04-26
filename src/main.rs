mod math_traits;
mod mat_struct;
mod vec_struct;

use vec_struct::*;
use mat_struct::*;

// use crate::math_traits::Mathable;

fn main() {
	println!("Hello, world!");

	{
		let foo: Vec<f32> = vec![21., 42.];
		let bar: Vec<f32> = foo.iter().map(|x| -x).collect();

		println!("{foo:?}\n{bar:?}");
	}
	{
		let mut foo = Vector::<3>::new();
		let mut bar = Matrix::<3, 3>::new();

		foo[1] = 1.;
		bar[0][0] = 42.;

		println!("{}", foo);
		println!("{}", bar);
	}
	{
		let foo = Vec3::from([0., 1., 2.]);
		let bar = Mat3::from([
			[1., 0., 0.],
			[0., 2., 0.],
			[0., 0., 3.],
		]);

		println!("{}", foo);
		println!("{}", bar);
	}
}
