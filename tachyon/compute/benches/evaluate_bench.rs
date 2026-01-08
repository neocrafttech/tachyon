use std::hint::black_box;

use compute::data_type::DataType;
use compute::error::ErrorMode;
use compute::evaluate::{Device, evaluate};
use compute::expr::Expr;
use compute::operator::Operator;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tokio::runtime::Builder;

//TODO: Export macro
#[macro_export]
macro_rules! create_column {
    ($vec:expr, $bit_vec:expr, $name:expr, $data_type:expr) => {{
        use std::sync::Arc;

        use compute::column::{Column, VecArray};
        let arr = Arc::new(VecArray { data: $vec.clone(), datatype: $data_type });
        Column::new($name, arr, $bit_vec)
    }};
}

macro_rules! generators {
    ($($fn_name:ident => $t:ty),+ $(,)?) => {
        $(
            pub fn $fn_name(len: usize) -> Vec<$t> {
                use rand::Rng;
                let mut rng = rand::rng();
                (0..len).map(|_| rng.random_range(<$t>::MIN..<$t>::MAX)).collect::<Vec<$t>>()
            }
        )+
    };
}

mod generators {
    generators! {
        random_f32 => f32,
        random_f64 => f64,
        random_u8 => u8,
        random_u16 => u16,
        random_u32 => u32,
        random_u64 => u64,
        random_i8 => i8,
        random_i16 => i16,
        random_i32 => i32,
        random_i64 => i64,
    }
}

fn single_thread_runtime() -> tokio::runtime::Runtime {
    Builder::new_current_thread().enable_all().build().unwrap()
}

struct EvaluateBenchmark {
    len: usize,
}

impl EvaluateBenchmark {
    fn new(len: usize) -> Self {
        Self { len }
    }

    fn column_types(&self) -> [DataType; 1] {
        [DataType::I32]
    }

    fn op_types(&self) -> [Operator; 4] {
        [Operator::Add, Operator::Sub, Operator::Mul, Operator::Div]
    }

    fn run(&self, c: &mut Criterion) {
        let rt = single_thread_runtime();

        for column_type in self.column_types() {
            let mut group = c.benchmark_group(format!("evaluate_metrics_{:?}", column_type));

            for op in self.op_types() {
                let op_name = format!("{:?}", op);
                let bench_id =
                    BenchmarkId::new(format!("{:?}_{:?}", column_type, op_name), self.len);

                group.bench_with_input(bench_id, &op, |bch, &op| {
                    bch.iter(|| {
                        use crate::generators::random_i32;
                        let col_a: compute::column::Column<u64> =
                            create_column!(random_i32(self.len), None, "a", column_type);
                        let col_b: compute::column::Column<u64> =
                            create_column!(random_i32(self.len), None, "b", column_type);

                        let expr = Expr::binary(op, Expr::col("a"), Expr::col("b"));
                        let result = rt.block_on(async {
                            evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &vec![col_a, col_b])
                                .await
                        });
                        println!("{:?}", result.is_ok());
                    });
                });
            }
            group.finish();
        }
    }
}

fn bench_all_evaluate(c: &mut Criterion) {
    let vb = EvaluateBenchmark::new(1_000_000);
    vb.run(c);
}

criterion_group!(benches, bench_all_evaluate);
criterion_main!(benches);
