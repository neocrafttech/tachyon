use std::hint::black_box;
use std::mem::size_of;

use compute::bit_vector::BitVector;
use compute::column::{Column, VecArray};
use compute::data_type::DataType;
use compute::error::ErrorMode;
use compute::evaluate::{Device, evaluate};
use compute::expr::Expr;
use compute::operator::Operator;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::Rng;
use tokio::runtime::Builder;

fn single_thread_runtime() -> tokio::runtime::Runtime {
    Builder::new_current_thread().enable_all().build().unwrap()
}

fn random_i32(len: usize) -> Vec<i32> {
    let mut rng = rand::rng();
    (0..len).map(|_| rng.random_range(i32::MIN..i32::MAX)).collect::<Vec<i32>>()
}

fn random_f64(len: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    (0..len).map(|_| rng.random_range(-1_000_000.0..1_000_000.0)).collect::<Vec<f64>>()
}

fn make_i32_column(name: &str, len: usize) -> Column<u64> {
    use std::sync::Arc;
    let values = Arc::new(VecArray { data: random_i32(len), datatype: DataType::I32 });
    Column::new(name, values, Some(BitVector::<u64>::new_all_valid(len)))
}

fn make_f64_column(name: &str, len: usize) -> Column<u64> {
    use std::sync::Arc;
    let values = Arc::new(VecArray { data: random_f64(len), datatype: DataType::F64 });
    Column::new(name, values, Some(BitVector::<u64>::new_all_valid(len)))
}

fn bench_aggregate_i32(c: &mut Criterion, rt: &tokio::runtime::Runtime, len: usize) {
    let col_a = make_i32_column("a", len);
    let ops = [Operator::Min, Operator::Max, Operator::Sum, Operator::Count];
    let bytes = (len * size_of::<i32>()) as u64;

    let mut group_rows = c.benchmark_group("aggregate_i32_rows");
    group_rows.throughput(Throughput::Elements(len as u64));
    for op in ops {
        let expr = Expr::aggregate(op, Expr::col("a"), false);
        let bench_id = BenchmarkId::new(format!("{:?}", op), len);
        group_rows.bench_with_input(bench_id, &op, |bch, &_op| {
            bch.iter(|| {
                let out = rt.block_on(async {
                    evaluate(Device::GPU, ErrorMode::Tachyon, &expr, black_box(&[col_a.clone()]))
                        .await
                });
                black_box(out).unwrap();
            });
        });
    }
    group_rows.finish();

    let mut group_bytes = c.benchmark_group("aggregate_i32_bytes");
    group_bytes.throughput(Throughput::Bytes(bytes));
    for op in ops {
        let expr = Expr::aggregate(op, Expr::col("a"), false);
        let bench_id = BenchmarkId::new(format!("{:?}", op), len);
        group_bytes.bench_with_input(bench_id, &op, |bch, &_op| {
            bch.iter(|| {
                let out = rt.block_on(async {
                    evaluate(Device::GPU, ErrorMode::Tachyon, &expr, black_box(&[col_a.clone()]))
                        .await
                });
                black_box(out).unwrap();
            });
        });
    }
    group_bytes.finish();
}

fn bench_aggregate_f64(c: &mut Criterion, rt: &tokio::runtime::Runtime, len: usize) {
    let col_a = make_f64_column("a", len);
    let expr = Expr::aggregate(Operator::Avg, Expr::col("a"), false);
    let bytes = (len * size_of::<f64>()) as u64;

    let mut group_rows = c.benchmark_group("aggregate_f64_rows");
    group_rows.throughput(Throughput::Elements(len as u64));
    let bench_id_rows = BenchmarkId::new("Avg", len);
    group_rows.bench_with_input(bench_id_rows, &len, |bch, &_len| {
        bch.iter(|| {
            let out = rt.block_on(async {
                evaluate(Device::GPU, ErrorMode::Tachyon, &expr, black_box(&[col_a.clone()])).await
            });
            black_box(out).unwrap();
        });
    });
    group_rows.finish();

    let mut group_bytes = c.benchmark_group("aggregate_f64_bytes");
    group_bytes.throughput(Throughput::Bytes(bytes));
    let bench_id_bytes = BenchmarkId::new("Avg", len);
    group_bytes.bench_with_input(bench_id_bytes, &len, |bch, &_len| {
        bch.iter(|| {
            let out = rt.block_on(async {
                evaluate(Device::GPU, ErrorMode::Tachyon, &expr, black_box(&[col_a.clone()])).await
            });
            black_box(out).unwrap();
        });
    });
    group_bytes.finish();
}

fn bench_all_aggregate(c: &mut Criterion) {
    let rt = single_thread_runtime();
    let len = 1_000_000;
    bench_aggregate_i32(c, &rt, len);
    bench_aggregate_f64(c, &rt, len);
}

criterion_group!(benches, bench_all_aggregate);
criterion_main!(benches);
