# atomic-backoff

[![Crates.io](https://img.shields.io/crates/v/atomic-backoff.svg)](https://crates.io/crates/atomic-backoff)
[![Documentation](https://docs.rs/atomic-backoff/badge.svg)](https://docs.rs/atomic-backoff)
[![License](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue.svg)](https://github.com/wyfo/atomic-backoff#license)

Customizable backoff strategies for compare-and-swap loops and spin loops.

Compare-and-swap (CAS) loops and spin loops can often be optimized by adding backoff at each
iteration, i.e. waiting a bit before the next iteration, in order to reduce the contention on
the CPU's cache lines.

As the optimal backoff strategy depends on multiple factors, especially the expected
contention, this crate provides a generic `BackoffStrategy` trait to help customize algorithms
using CAS/spin loops. Typical backoff strategies like `ExponentialBackoff` are also provided.

Atomic types are extended with `try_update_with_backoff`/`update_with_backoff` methods,
mirroring their std `try_update`/`update` counterparts.

For handwritten CAS loops, see `BackoffStrategy::backoff_reload` and `BackoffState`;
for spin loops, see `BackoffStrategy::backoff_until`, or `BoundedBackoffStrategy` to spin
a bounded number of iterations before falling back to a slower waiting mechanism.

## Example

```rust
use std::{
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
    thread,
    time::{Duration, Instant},
};

use atomic_backoff::{AtomicExt, BackoffStrategy, ExponentialBackoff, NoBackoff};

fn parallel_increment<S: BackoffStrategy>(threads: usize, iterations: usize) -> Duration {
    let counter = AtomicUsize::new(0);
    let start = Instant::now();
    thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                for _ in 0..iterations {
                    counter.update_with_backoff(Relaxed, Relaxed, |x| x + 1, S::default());
                }
            });
        }
    });
    assert_eq!(counter.load(Relaxed), threads * iterations);
    start.elapsed()
}

let no_backoff = parallel_increment::<NoBackoff>(4, 10_000);
let exponential = parallel_increment::<ExponentialBackoff<6, 4>>(4, 10_000);
println!("no backoff: {no_backoff:?}, exponential backoff: {exponential:?}");
// no backoff: 2.08ms, exponential backoff: 646µs
```

## Retry strategies

Unlike most backoff implementations, a `BackoffStrategy` doesn't only decide *how long* to wait
after a failed CAS, but also *what to do with the atomic value* before retrying, through the
`RetryStrategy` variant it returns:

- `NoReload`: retry with the value returned by the failed CAS;
- `Reload`: reload the atomic and retry with the up-to-date value;
- `ReloadUntilUnchanged`: reload the atomic and keep backing off while its value changes between
  reloads, then retry with the up-to-date value.

`ReloadUntilUnchanged` avoids attempting a CAS while the atomic is being actively modified, which
significantly reduces the number of failed CAS under contention, and the contention itself. However, it should only be
returned for a bounded number of iterations, as it could otherwise lead to starvation under sustained contention.

## Comparison with [`crossbeam::utils::Backoff`](https://docs.rs/crossbeam/latest/crossbeam/utils/struct.Backoff.html)

`crossbeam::utils::Backoff` is strictly equivalent to `ExponentialBackoff<6>` when `Backoff::spin` is used, and `ExponentialBackoff<10, 0, 7>` when `Backoff::snooze` is used. However, it is not customizable, and especially it doesn't convey the way the atomic value should be reloaded. 

On the other hand, `ExponentialBackoff` provides a `UNTIL_UNCHANGED_LIMIT` parameter to wait until the atomic stops being contended, which significantly impacts the overall contention.

Running the example above with `ExponentialBackoff<6, N>` for different values of `N`:

| Strategy                                     | Time   |
|----------------------------------------------|--------|
| `NoBackoff`                                  | 2.08ms |
| `ExponentialBackoff<6>` (crossbeam's `spin`) | 1.15ms |
| `ExponentialBackoff<6, 4>`                   | 646µs  |
| `ExponentialBackoff<6, 8>`                   | 569µs  |

## Features

- `std` (default): enables `std::thread::yield_now` in `ExponentialBackoff` (`YIELD_AFTER`
  parameter). Without it, the crate is `no_std` and `ExponentialBackoff` keeps spinning instead
  of yielding.
- `portable-atomic`: extends [`portable-atomic`](https://docs.rs/portable-atomic) atomic types to support `try_update_with_backoff`/`update_with_backoff`.

## Loom support

[`loom`](https://docs.rs/loom) atomic types are also extended to support  `try_update_with_backoff`/`update_with_backoff` when compiled with
`--cfg loom`, so that algorithms built on this crate can be model-checked with loom without any
feature flag.

## License

Licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT license](LICENSE-MIT)

at your option.
