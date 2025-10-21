macro_rules! test_eval_kernel_matrix {
    (
        $operation_name:expr,
        $epsilon:expr,
        $verify_fn:expr,
        [
            $(
                ( $test_name:ident, $data_type:ty, $size:expr )
            ),* $(,)?
        ]
    ) => {
        $(
            test_eval_kernel_fn!(
                $test_name,
                $operation_name,
                $data_type,
                $size,
                $epsilon,
                $verify_fn
            );
        )*
    };
}

macro_rules! test_eval_kernel_fn {
    (
        $test_name:ident,
        $operation_name:expr,
        $data_type:ty,
        $size:expr,
        $epsilon:expr,
        $verify_fn:expr
    ) => {
        #[cfg(feature = "gpu")]
        #[test]
        fn $test_name() {
            use gpu::cuda_launcher;
            use rand;
            use rand::Rng;
            let n = $size;
            let mut rng = rand::rng();

            let a_vec: Vec<$data_type> = (0..$size)
                .map(|_| {
                    rng.random_range(
                        <$data_type>::MIN / 2 as $data_type..<$data_type>::MAX / 2 as $data_type,
                    )
                })
                .collect();

            let b_vec: Vec<$data_type> = (0..$size)
                .map(|_| {
                    rng.random_range(
                        <$data_type>::MIN / 2 as $data_type..<$data_type>::MAX / 2 as $data_type,
                    )
                })
                .collect();

            let input = vec![&a_vec, &b_vec];
            let res = cuda_launcher::launch::<$data_type>(&input);

            assert!(
                res.is_ok(),
                "CUDA kernel launch failed for {}",
                $operation_name
            );

            let output = res.unwrap();
            assert_eq!(output.len(), 1, "Expected 1 output vector");

            println!(
                "Testing {} with type {} and size {}",
                $operation_name,
                stringify!($data_type),
                n
            );
            println!("First 10 results: {:?}", &output[0][0..10.min(n)]);

            for i in 0..$size {
                let expected = $verify_fn(a_vec[i], b_vec[i]);
                let actual = output[0][i];
                let diff = if expected > actual {
                    expected - actual
                } else {
                    actual - expected
                };

                assert!(
                    diff <= $epsilon as $data_type,
                    "Mismatch at index {}: expected {}, got {}, diff {}",
                    i,
                    expected,
                    actual,
                    diff
                );
            }
        }
    };
}

#[cfg(feature = "gpu")]
fn add_verify<T: std::ops::Add<Output = T>>(a: T, b: T) -> T {
    a + b
}

test_eval_kernel_matrix!(
    "vector_addition",
    1e-6,
    add_verify,
    [
        (test_add_i8_small, i8, 100),
        (test_add_i8_medium, i8, 100 * 10),
        (test_add_i8_large, i8, 100 * 1000),
        (test_add_u8_small, u8, 100),
        (test_add_u8_medium, u8, 100 * 10),
        (test_add_u8_large, u8, 100 * 1000),
        (test_add_i16_small, i16, 500),
        (test_add_i16_medium, i16, 500 * 10),
        (test_add_i16_large, i16, 500 * 1000),
        (test_add_u16_small, u16, 500),
        (test_add_u16_medium, u16, 500 * 10),
        (test_add_u16_large, u16, 500 * 1000),
        (test_add_i32_small, i32, 1024),
        (test_add_i32_medium, i32, 1024 * 10),
        (test_add_i32_large, i32, 1024 * 100),
        (test_add_u32_small, u32, 512),
        (test_add_u32_medium, u32, 512 * 10),
        (test_add_u32_large, u32, 512 * 100),
        (test_add_i64_small, i64, 1024),
        (test_add_i64_medium, i64, 1024 * 10),
        (test_add_i64_large, i64, 1024 * 100),
        (test_add_u64_small, u64, 1024),
        (test_add_u64_medium, u64, 1024 * 10),
        (test_add_u64_large, u64, 1024 * 100),
        (test_add_f32_small, f32, 1024),
        (test_add_f32_medium, f32, 1024 * 10),
        (test_add_f32_large, f32, 1024 * 100),
        (test_add_f64_small, f64, 1024),
        (test_add_f64_medium, f64, 1024 * 10),
    ]
);
