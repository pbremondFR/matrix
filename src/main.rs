#![feature(min_specialization)]
#![feature(associated_type_defaults)]
#![feature(generic_const_exprs)]
mod math_traits;
mod mat_struct;
mod vec_struct;
mod funcs;
mod macros;

use vec_struct::*;
use mat_struct::*;
use math_traits::*;

// use crate::math_traits::Mathable;

fn main() {
	println!("Hello, world!");

	{
		let u = Matrix::from([
			[ 1.]
		]);
		let _ = u.det();

		let u = Matrix::from([
			[ 1., -1.],
			[-1., 1.],
		]);
		let _ = u.det();

		let u = Matrix::from([
			[2., 0., 0.],
			[0., 2., 0.],
			[0., 0., 2.],
		]);
		let _ = u.det();
	}
}
