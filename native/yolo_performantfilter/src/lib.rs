// #[rustler::nif]
// fn add(a: i64, b: i64) -> i64 {
//     a + b
// }

// rustler::init!("Elixir.Yolo.PerformantFilter");


use rustler::{Binary, Env, NifResult, Encoder, Term, OwnedBinary};

// This macro expands to the code that makes the function visible to Elixir.
#[rustler::nif(schedule = "DirtyCpu")]
fn filter_greater_binary<'a>(
    env: Env<'a>,
    tensor_data: Binary,   // raw bytes of the Nx tensor
    threshold: f32,        // threshold as float64
) -> NifResult<Term<'a>> {
    // Convert raw bytes -> slice of f32.
    let float_count = tensor_data.len() / 4;
    let float_slice = unsafe {
        std::slice::from_raw_parts(tensor_data.as_ptr() as *const f32, float_count)
    };

    let threshold_f32 = threshold;

    // Collect indices of values above threshold into a Vec<u64>.
    let mut out_vec = Vec::new();
    out_vec.reserve(float_count);
    for (index, &value) in float_slice.iter().enumerate() {
        if value > threshold_f32 {
            out_vec.push(index as u64);
        }
    }

    // Convert Vec<u64> to raw bytes in a new Binary for Elixir.
    let out_bytes = unsafe {
        // Transmute the u64 Vec into a u8 Vec
        let ptr = out_vec.as_mut_ptr() as *mut u8;
        let len = out_vec.len() * std::mem::size_of::<u64>();
        let cap = out_vec.capacity() * std::mem::size_of::<u64>();
        std::mem::forget(out_vec); // Prevent Rust from freeing the memory
        Vec::from_raw_parts(ptr, len, cap)
    };

    // Create an OwnedBinary to hold the bytes
    let mut owned_binary = OwnedBinary::new(out_bytes.len()).unwrap();

    // Copy the data from the Vec<u8> to the OwnedBinary
    owned_binary.as_mut_slice().copy_from_slice(&out_bytes);

    // Create a Binary from the OwnedBinary
    let out_binary = owned_binary.release(env);

    Ok(out_binary.encode(env))
}

rustler::init!("Elixir.Yolo.PerformantFilter");