#![feature(test)]
#![allow(unused_variables)]
use criterion::{black_box, criterion_group, criterion_main, Criterion};

extern crate test;
use matrix::*;
use rand::distributions::{Distribution, Uniform};

fn inverse(c: &mut Criterion) {
	// c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
	let mut rng = rand::thread_rng();
	let die = Uniform::from(-200.0_f64..200.0_f64);

	let mut c = c.benchmark_group("Inverse");
	c.sample_size(1000);

	c.bench_function("2x2 hardcoded matrix inverse", |b| {
		let u = Matrix::<2, 2, f64>::from([
			[4., 2.],
			[2., 1.],
		]);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			u.inverse_unchecked()
		})
	});

	c.bench_function("3x3 hardcoded matrix inverse", |b| {
		let u = Matrix::<3, 3, f64>::from([
			[8., 5., -2.],
			[4., 7., 20.],
			[7., 6., 1.],
		]);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			u.inverse_unchecked()
		})
	});

	c.bench_function("2x2 matrix inverse", |b| {
		let u = Matrix::<2, 2, f64>::from([
			[die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng)],
		]);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			u.inverse_unchecked()
		})
	});

	c.bench_function("3x3 matrix inverse", |b| {
		let u = Matrix::<3, 3, f64>::from([
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
	// 		u.det()
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
	// 		u.det()
	// 	})
	// });

	// c.bench_function("6x6 matrix inverse", |b| {
	// 	let u = Matrix::from([
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 		[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
	// 	]);
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		u.det()
	// 	})
	// });

	// c.bench_function("7x7 matrix inverse", |b| {
	// 	let u = Matrix::from([
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
	// 		u.det()
	// 	})
	// });

	// c.bench_function("100x100 matrix inverse", |b| {
	// 	const SIZE: usize = 100;
	// 	let mut u = Matrix::<SIZE, SIZE, f32>::new();
	// 	for i in 0..SIZE {
	// 		let vec: Vec<f32> = (0..SIZE).map(|_| die.sample(&mut rng)).collect();
	// 		u[i] = Vector::<SIZE, f32>::from_slice(&vec);
	// 	}
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		u.det()
	// 	})
	// });

	// c.bench_function("200x200 matrix inverse", |b| {
	// 	const SIZE: usize = 200;
	// 	let mut u = Matrix::<SIZE, SIZE, f32>::new();
	// 	for i in 0..SIZE {
	// 		let vec: Vec<f32> = (0..SIZE).map(|_| die.sample(&mut rng)).collect();
	// 		u[i] = Vector::<SIZE, f32>::from_slice(&vec);
	// 	}
	// 	b.iter(|| {
	// 		let n = black_box(42);	// Trick compiler into not optimizing everything away
	// 		u.det()
	// 	})
	// });
}

criterion_group!(benches, inverse);
criterion_main!(benches);
