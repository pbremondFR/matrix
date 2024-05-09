#![feature(test)]
#![allow(unused_variables)]
use criterion::{black_box, criterion_group, criterion_main, Criterion};

extern crate test;
use matrix::*;
use rand::distributions::{Distribution, Uniform};

fn inverse(c: &mut Criterion) {
	let mut rng = rand::thread_rng();
	let die = Uniform::from(-200.0_f32..200.0_f32);

	let mut c = c.benchmark_group("Scalar multiplication");
	c.sample_size(1000);

	c.bench_function("Vector2 scalar multiplication", |b| {
		let u = Vector::from([die.sample(&mut rng), die.sample(&mut rng)]);
		let v = die.sample(&mut rng);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			u * v
		})
	});

	c.bench_function("Vector2 reference scalar multiplication", |b| {
		let u = Vector::from([die.sample(&mut rng), die.sample(&mut rng)]);
		let v = die.sample(&mut rng);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			&u * v
		})
	});

	c.bench_function("Vector3 scalar multiplication", |b| {
		let u = Vector::from([die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)]);
		let v = die.sample(&mut rng);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			u * v
		})
	});

	c.bench_function("Vector3 reference scalar multiplication", |b| {
		let u = Vector::from([die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)]);
		let v = die.sample(&mut rng);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			&u * v
		})
	});

	c.bench_function("Matrix 2x2 scalar multiplication", |b| {
		let u = Matrix::from([
			[die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng)],
		]);
		let v = die.sample(&mut rng);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			&u * v
		})
	});

	c.bench_function("Matrix 3x3 scalar multiplication", |b| {
		let u = Matrix::from([
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
			[die.sample(&mut rng), die.sample(&mut rng), die.sample(&mut rng)],
		]);
		let v = die.sample(&mut rng);
		b.iter(|| {
			let n = black_box(42);	// Trick compiler into not optimizing everything away
			&u * v
		})
	});

}

criterion_group!(benches, inverse);
criterion_main!(benches);
