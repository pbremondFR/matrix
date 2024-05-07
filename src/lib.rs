#![feature(min_specialization)]
#![feature(associated_type_defaults)]
#![feature(generic_const_exprs)]
#![feature(test)]
pub mod math_traits;
pub mod mat_struct;
pub mod vec_struct;
pub mod funcs;
pub mod macros;

pub use vec_struct::*;
pub use mat_struct::*;
pub use math_traits::*;

use rand::{distributions::{Distribution, Uniform}};

// fn main() {
// 	println!("Hello, world!");
// 	{
// 		const ITER_LEN: usize = 10_000;
// 		let mut rng = rand::thread_rng();
// 		let die = Uniform::from(-20.0_f32..20.0_f32);
// 		let mut results = Vec::<f32>::with_capacity(ITER_LEN);
// 		for i in 0..ITER_LEN {
// 			let u = Matrix::from([
// 				[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
// 				[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
// 				[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
// 				[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
// 			]);
// 			results.push(u.det());
// 		}
// 	}
// }
