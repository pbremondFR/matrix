#![feature(test)]
use criterion::{black_box, criterion_group, criterion_main, Criterion};

extern crate test;
use matrix::*;
use rand::distributions::{Distribution, Uniform};

fn criterion_benchmark(c: &mut Criterion) {
	// c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
	let mut rng = rand::thread_rng();
	let die = Uniform::from(-200.0_f32..200.0_f32);

	c.bench_function("2x2 manual hardcoded matrix determinant", |b| {
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			4.0 * 1.0 - 2.0 * 2.0
		})
	});

	c.bench_function("2x2 hardcoded matrix determinant", |b| {
		let u = Matrix::from([
			[4., 2.],
			[2., 1.],
		]);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			u.det()
		})
	});

	c.bench_function("3x3 hardcoded matrix determinant", |b| {
		let u = Matrix::from([
			[8., 5., -2.],
			[4., 7., 20.],
			[7., 6., 1.],
		]);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			u.det()
		})
	});

	c.bench_function("2x2 matrix determinant", |b| {
		let u = Matrix::from([
			[die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng)],
		]);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			u.det()
		})
	});

	c.bench_function("3x3 matrix determinant", |b| {
		let u = Matrix::from([
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
		]);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			u.det()
		})
	});

	c.bench_function("4x4 matrix determinant", |b| {
		let u = Matrix::from([
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
		]);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			u.det()
		})
	});

	c.bench_function("5x5 matrix determinant", |b| {
		let u = Matrix::from([
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
		]);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			u.det()
		})
	});

	c.bench_function("6x6 matrix determinant", |b| {
		let u = Matrix::from([
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
		]);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			u.det()
		})
	});

	c.bench_function("7x7 matrix determinant", |b| {
		let u = Matrix::from([
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
		]);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			u.det()
		})
	});
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
