#![feature(test)]
#![allow(unused_variables)]
use criterion::{black_box, criterion_group, criterion_main, Criterion};

extern crate test;
use matrix::*;
use rand::distributions::{Distribution, Uniform};

fn inverse(c: &mut Criterion) {
	// c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
	let mut rng = rand::thread_rng();
	let die = Uniform::from(-200.0_f32..200.0_f32);

	let mut c = c.benchmark_group("Inverse");
	// c.sample_size(1000);

	// c.bench_function("2x2 hardcoded matrix inverse", |b| {
	// 	let u = Matrix::<2, 2, f64>::from([
	// 		[4., 2.],
	// 		[2., 1.],
	// 	]);
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		u.inverse_unchecked()
	// 	})
	// });

	// c.bench_function("3x3 hardcoded matrix inverse", |b| {
	// 	let u = Matrix::<3, 3, f64>::from([
	// 		[8., 5., -2.],
	// 		[4., 7., 20.],
	// 		[7., 6., 1.],
	// 	]);
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		u.inverse_unchecked()
	// 	})
	// });

	c.bench_function("2x2 matrix inverse", |b| {
		let mut u = Matrix::from([
			[die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng)],
		]);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			u.inverse()
			// if u.det() != 0.0 {
			// 	u.inverse_unchecked();
			// } else {
			// 	panic!("fuck");
			// }
			// u
		})
	});

	c.bench_function("2x2 unchecked matrix inverse", |b| {
		let u = Matrix::from([
			[die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng)],
		]);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			u.inverse_unchecked()
		})
	});

	c.bench_function("3x3 matrix inverse", |b| {
		let mut u = Matrix::from([
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
		]);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			u.inverse()
		})
	});

	c.bench_function("3x3 unchecked matrix inverse", |b| {
		let u = Matrix::from([
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
		]);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			u.inverse_unchecked()
		})
	});

	// c.bench_function("4x4 matrix inverse", |b| {
	// 	let u = Matrix::from([
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 	]);
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		u.inverse()
	// 	})
	// });

	// c.bench_function("5x5 matrix inverse", |b| {
	// 	let u = Matrix::from([
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 	]);
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		u.inverse()
	// 	})
	// });

	// c.bench_function("6x6 matrix inverse", |b| {
	// 	let mut u = Matrix::from([
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 	]);
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		u.inverse()
	// 	})
	// });

	// c.bench_function("7x7 matrix inverse", |b| {
	// 	let mut u = Matrix::from([
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 	]);
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		u.inverse()
	// 	})
	// });

	// c.bench_function("8x8 matrix inverse", |b| {
	// 	let mut u = Matrix::from([
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 	]);
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		u.inverse()
	// 	})
	// });

	// c.bench_function("9x9 matrix inverse", |b| {
	// 	let mut u = Matrix::from([
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 	]);
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		u.inverse()
	// 	})
	// });

	// c.bench_function("20x20 matrix inverse", |b| {
	// 	const SIZE: usize = 20;
	// 	let mut u = Matrix::<SIZE, SIZE>::new();
	// 	for i in 0..SIZE {
	// 		u[i] = Vector::<SIZE>::from_iter((0..SIZE).map(|_| die.sample(&mut rng)));
	// 	}
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		u.inverse()
	// 	})
	// });

	// c.bench_function("2x2 matrix in-place inverse", |b| {
	// 	let mut u = Matrix::from([
	// 		[die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng)],
	// 	]);
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		let stfuborrowchecker = u.inverse_inplace();
	// 	})
	// });

	// c.bench_function("3x3 matrix in-place inverse", |b| {
	// 	let mut u = Matrix::from([
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 	]);
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		let stfuborrowchecker = u.inverse_inplace();
	// 	})
	// });

	// c.bench_function("4x4 matrix in-place inverse", |b| {
	// 	let mut u = Matrix::from([
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 	]);
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		let stfuborrowchecker = u.inverse_inplace();
	// 	})
	// });

	// c.bench_function("5x5 matrix in-place inverse", |b| {
	// 	let mut u = Matrix::from([
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 	]);
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		let stfuborrowchecker = u.inverse_inplace();
	// 	})
	// });

	// c.bench_function("6x6 matrix in-place inverse", |b| {
	// 	let mut u = Matrix::from([
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 	]);
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		u.inverse_inplace()
	// 	})
	// });

	// c.bench_function("7x7 matrix in-place inverse", |b| {
	// 	let mut u = Matrix::from([
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 	]);
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		u.inverse_inplace()
	// 	})
	// });

	// c.bench_function("8x8 matrix in-place inverse", |b| {
	// 	let mut u = Matrix::from([
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 	]);
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		u.inverse_inplace()
	// 	})
	// });

	// c.bench_function("9x9 matrix in-place inverse", |b| {
	// 	let mut u = Matrix::from([
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 	]);
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		u.inverse_inplace()
	// 	})
	// });

	// c.bench_function("20x20 matrix in-place inverse", |b| {
	// 	const SIZE: usize = 20;
	// 	let mut u = Matrix::<SIZE, SIZE>::new();
	// 	for i in 0..SIZE {
	// 		u[i] = Vector::<SIZE>::from_iter((0..SIZE).map(|_| die.sample(&mut rng)));
	// 	}
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		u.inverse_inplace()
	// 	})
	// });
}

criterion_group!(benches, inverse);
criterion_main!(benches);
