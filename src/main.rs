mod mat_struct;
mod vec_struct;

use vec_struct::*;
use mat_struct::*;

fn main() {
	println!("Hello, world!");

	{
		let mut foo = Vector::<3>::new();
		let mut bar = Matrix::<3>::new();

		foo[1] = 1.;
		bar[0][0] = 42.;

		println!("{}", foo);
		println!("{:?}", bar);
	}
	{
		let foo = Vec3::from([0., 1., 2.]);
		let bar = Mat3::from([
			[1., 0., 0.],
			[0., 1., 0.],
			[0., 0., 1.],
		]);

		println!("{:?}", foo);
		println!("{:?}", bar);
	}
	{
		let a = Vec2::from( [1., 1.] );
		let b = Vec2::from( [1., -1.] );
		let res = a + b;

		println!("{:?}", res);
	}
}
