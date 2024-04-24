mod mat_struct;
mod vec_struct;

use vec_struct::*;
use mat_struct::*;

fn main() {
	println!("Hello, world!");

	let mut foo = Vector::<3>::new();
	let mut bar = Matrix::<3>::new();

	foo.data[1] = 1.;
	bar.data[0][0] = 42.;

	println!("{:?}", foo);
	println!("{:?}", bar);

	{
		let a = Vec2{data: [1., 1.]};
		let b = Vec2{data: [1., -1.]};
		let res = a + b;

		println!("{:?}", res);
	}
}
