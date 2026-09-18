//! # atomic-backoff
//!
//! Customizable backoff strategies for compare-and-swap loops and spin loops.
//!
//! Compare-and-swap (CAS) loops and spin loops can often be optimized by adding backoff at each
//! iteration, i.e. waiting a bit before the next iteration, in order to reduce the contention on
//! the CPU's cache lines.
//!
//! As the optimal backoff strategy depends on multiple factors, especially the expected
//! contention, this crate provides a generic [`BackoffStrategy`] to help customize algorithms
//! using CAS/spin loops. Typical backoff strategies like [`ExponentialBackoff`] are also provided.
//!
//! Atomic types are extended with [`try_update_with_backoff`]/[`update_with_backoff`] methods,
//! mirroring their std `try_update`/`update` counterparts.
//!
//! For handwritten CAS loops, see [`BackoffStrategy::backoff_reload`] and [`BackoffState`];
//! for spin loops, see [`BackoffStrategy::backoff_until`], or [`BoundedBackoffStrategy`] to spin
//! a bounded number of iterations before falling back to a slower waiting mechanism.
//!
//! # Examples
//!
//! ```rust
//! use std::{
//!     sync::atomic::{AtomicUsize, Ordering::Relaxed},
//!     thread,
//!     time::{Duration, Instant},
//! };
//!
//! use atomic_backoff::{AtomicWithBackoffExt, BackoffStrategy, ExponentialBackoff, NoBackoff};
//!
//! fn parallel_increment<S: BackoffStrategy>(threads: usize, iterations: usize) -> Duration {
//!     let counter = AtomicUsize::new(0);
//!     let start = Instant::now();
//!     thread::scope(|s| {
//!         for _ in 0..threads {
//!             s.spawn(|| {
//!                 for _ in 0..iterations {
//!                     counter.update_with_backoff(Relaxed, Relaxed, |x| x + 1, S::default());
//!                 }
//!             });
//!         }
//!     });
//!     assert_eq!(counter.load(Relaxed), threads * iterations);
//!     start.elapsed()
//! }
//!
//! let no_backoff = parallel_increment::<NoBackoff>(4, 10_000);
//! let exponential = parallel_increment::<ExponentialBackoff<6, 4>>(4, 10_000);
//! println!("no backoff: {no_backoff:?}, exponential backoff: {exponential:?}");
//! // no backoff: 2.08ms, exponential backoff: 646µs
//! ```
//!
//! [`try_update_with_backoff`]: AtomicWithBackoffExt::try_update_with_backoff
//! [`update_with_backoff`]: AtomicWithBackoffExt::update_with_backoff
#![no_std]

#[cfg(feature = "std")]
extern crate std;

use core::{hint::spin_loop, sync::atomic::Ordering};

/// Backoff strategy to be used after an atomic compare-and-swap (CAS) failure and in spin loops.
///
/// Waiting before retrying a failed CAS can greatly reduce the contention on the atomic's cache
/// line, and improve the performance of CAS loops.
///
/// Spin loops also benefit from backoff as it avoids keeping the CPU 100% busy while waiting,
/// at little latency cost.
pub trait BackoffStrategy: Default + Send + Sync + 'static {
    /// Whether the strategy does backoff or not.
    ///
    /// Some algorithms may have a different behavior depending on whether backoff is used; for
    /// example switching between an unbounded spin loop or a thread parking algorithm. This
    /// constant can be used for this purpose.
    ///
    /// It should be set to `false` only for [`NoBackoff`].
    const BACKOFF: bool = true;

    /// Performs backoff and returns how the CAS should be retried.
    ///
    /// [`will_reload`](Self::will_reload) should also be implemented accordingly.
    ///
    /// In spin loops, the returned value can simply be ignored.
    fn backoff(&mut self) -> RetryStrategy;

    /// Returns `true` if the next call to [`backoff`](Self::backoff) will not return
    /// [`RetryStrategy::NoReload`].
    ///
    /// This hint can be used to downgrade the failure ordering of the CAS to `Relaxed` when the
    /// returned value is overwritten anyway.
    #[inline]
    fn will_reload(&self) -> bool {
        false
    }

    /// Performs backoff after a failed CAS and reloads the atomic value according to the returned
    /// [`RetryStrategy`].
    ///
    /// [`ReloadUntilUnchanged`] causes this function to loop until the value stops changing. If the
    /// value returned by the failed CAS must be tested between reloads, use [`BackoffState`]
    /// instead.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use core::sync::atomic::{AtomicUsize, Ordering};
    /// # use atomic_backoff::BackoffStrategy;
    /// #
    /// fn update_with_backoff(
    ///     atomic: &AtomicUsize,
    ///     set_order: Ordering,
    ///     fetch_order: Ordering,
    ///     mut f: impl FnMut(usize) -> usize,
    ///     mut strategy: impl BackoffStrategy,
    /// ) -> usize {
    ///     let mut current = atomic.load(fetch_order);
    ///     loop {
    ///         let failure_order = if strategy.will_reload() {
    ///             Ordering::Relaxed
    ///         } else {
    ///             fetch_order
    ///         };
    ///         match atomic.compare_exchange_weak(current, f(current), set_order, failure_order) {
    ///             Ok(x) => return x,
    ///             Err(cur) => current = strategy.backoff_reload(cur, || atomic.load(fetch_order)),
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// [`ReloadUntilUnchanged`]: RetryStrategy::ReloadUntilUnchanged
    #[inline]
    fn backoff_reload<T: PartialEq, F: FnMut() -> T>(
        &mut self,
        mut current: T,
        mut reload: F,
    ) -> T {
        loop {
            match self.backoff() {
                RetryStrategy::NoReload => return current,
                RetryStrategy::Reload => return reload(),
                RetryStrategy::ReloadUntilUnchanged => {
                    let reloaded = reload();
                    if reloaded == current {
                        return current;
                    }
                    current = reloaded;
                }
            }
        }
    }

    /// Loops until a condition is satisfied, performing backoff at each iteration.
    #[inline]
    fn backoff_until<C: BackoffUntilCondition, F: FnMut() -> C>(&mut self, mut f: F) -> C::Result {
        loop {
            if let Some(res) = f().into_result() {
                return res;
            }
            self.backoff();
        }
    }
}

/// Retry strategy of a failed atomic compare-and-swap (CAS).
///
/// It is returned by [`BackoffStrategy::backoff`] and tells what to do with the value returned
/// by the failed CAS before retrying.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryStrategy {
    /// Retry with the value returned by the failed CAS.
    NoReload,
    /// Reload the atomic and retry the CAS with the up-to-date value.
    Reload,
    /// Reload the atomic and keep backing off while the value changes between reloads, then retry
    /// with the up-to-date value.
    ReloadUntilUnchanged,
}

/// A condition checked in [`BackoffStrategy::backoff_until`].
///
/// It should typically be a `bool` or an `Option<T>`.
pub trait BackoffUntilCondition {
    /// The result to return when the condition is satisfied.
    type Result;
    /// Converts the condition into a result to be returned.
    fn into_result(self) -> Option<Self::Result>;
}

impl BackoffUntilCondition for bool {
    type Result = ();
    #[inline]
    fn into_result(self) -> Option<Self::Result> {
        if self { Some(()) } else { None }
    }
}

impl<T> BackoffUntilCondition for Option<T> {
    type Result = T;
    #[inline]
    fn into_result(self) -> Option<Self::Result> {
        self
    }
}

/// No backoff.
///
/// Retries immediately with the value returned by the failed CAS.
#[derive(Debug, Default)]
pub struct NoBackoff;

impl BackoffStrategy for NoBackoff {
    const BACKOFF: bool = false;
    #[inline]
    fn backoff(&mut self) -> RetryStrategy {
        RetryStrategy::NoReload
    }
}

impl BoundedBackoffStrategy for NoBackoff {
    #[inline]
    fn is_completed(&self) -> bool {
        true
    }
}

/// A [`BackoffStrategy`] which completes after a bounded number of iterations.
///
/// It is typically used to spin a bit before falling back to a slower waiting mechanism, like
/// parking the thread.
///
/// # Examples
///
/// ```rust
/// use std::sync::atomic::{AtomicBool, Ordering::Acquire};
///
/// use atomic_backoff::{BackoffLimit, BoundedBackoffStrategy, ExponentialBackoff};
///
/// fn wait(flag: &AtomicBool, park: impl Fn()) {
///     let mut backoff = BackoffLimit::<ExponentialBackoff<6>, 10>::default();
///     // Spin a bit, then park the thread if the flag is still not set.
///     while backoff.try_backoff_until(|| flag.load(Acquire)).is_none() {
///         park();
///     }
/// }
/// ```
pub trait BoundedBackoffStrategy: BackoffStrategy {
    /// Returns `true` if the bounded number of iterations has been reached.
    ///
    /// [`backoff`](BackoffStrategy::backoff) can still be called afterward.
    fn is_completed(&self) -> bool;

    /// Loops until a condition is satisfied or the backoff is completed, performing backoff at
    /// each iteration.
    ///
    /// Returns `None` if the backoff completed before the condition was satisfied.
    #[inline]
    fn try_backoff_until<C: BackoffUntilCondition, F: FnMut() -> C>(
        &mut self,
        mut f: F,
    ) -> Option<C::Result> {
        loop {
            if let Some(res) = f().into_result() {
                return Some(res);
            }
            if self.is_completed() {
                return None;
            }
            self.backoff();
        }
    }
}

/// Wraps a [`BackoffStrategy`] to make it a [`BoundedBackoffStrategy`] completing after `LIMIT`
/// iterations.
#[derive(Debug, Default)]
pub struct BackoffLimit<S, const LIMIT: usize> {
    strategy: S,
    iter: usize,
}

impl<S: BackoffStrategy, const LIMIT: usize> BackoffLimit<S, LIMIT> {
    /// Wraps the given strategy.
    pub fn new(strategy: S) -> Self {
        Self { strategy, iter: 0 }
    }
}

impl<S: BackoffStrategy, const LIMIT: usize> BackoffStrategy for BackoffLimit<S, LIMIT> {
    const BACKOFF: bool = S::BACKOFF;

    #[inline]
    fn backoff(&mut self) -> RetryStrategy {
        let retry = self.strategy.backoff();
        self.iter = self.iter.saturating_add(1);
        retry
    }

    #[inline]
    fn will_reload(&self) -> bool {
        self.strategy.will_reload()
    }
}

impl<S: BackoffStrategy, const LIMIT: usize> BoundedBackoffStrategy for BackoffLimit<S, LIMIT> {
    #[inline]
    fn is_completed(&self) -> bool {
        self.iter >= LIMIT
    }
}

/// Emits a [`spin_loop`] and reloads the atomic value before retrying the CAS.
#[derive(Debug, Default)]
pub struct SpinBackoff;

impl BackoffStrategy for SpinBackoff {
    #[inline]
    fn backoff(&mut self) -> RetryStrategy {
        spin_loop();
        RetryStrategy::Reload
    }

    #[inline]
    fn will_reload(&self) -> bool {
        true
    }
}

/// Performs exponential backoff.
///
/// Each backoff iteration `iter` (starting from 0) calls [`spin_loop`] `1 << iter.min(SPIN_LIMIT)`
/// times.
///
/// During the first `UNTIL_UNCHANGED_LIMIT` backoff iterations, backoff continues until the
/// atomic's value stops changing between reloads; after that, the CAS is retried after a single
/// reload to avoid starvation.
///
/// After `YIELD_AFTER` backoff iterations (and if the `std` feature is enabled), [`yield_now`] is
/// called instead of spinning.
///
/// For reference, [`crossbeam::utils::Backoff`] is equivalent to `ExponentialBackoff<6>` with
/// `Backoff::spin`, and `ExponentialBackoff<10, 0, 7>` with `Backoff::snooze`. However,
/// `UNTIL_UNCHANGED_LIMIT` should also be used in contended CAS loop to further reduce contention.
///
/// [`yield_now`]: https://doc.rust-lang.org/std/thread/fn.yield_now.html
/// [`crossbeam::utils::Backoff`]: https://docs.rs/crossbeam/latest/crossbeam/utils/struct.Backoff.html
#[derive(Debug, Default)]
pub struct ExponentialBackoff<
    const SPIN_LIMIT: usize,
    const UNTIL_UNCHANGED_LIMIT: usize = 0,
    const YIELD_AFTER: usize = { usize::MAX },
> {
    iter: usize,
}

impl<const SPIN_LIMIT: usize, const UNTIL_UNCHANGED_LIMIT: usize, const YIELD_AFTER: usize>
    ExponentialBackoff<SPIN_LIMIT, UNTIL_UNCHANGED_LIMIT, YIELD_AFTER>
{
    const ASSERT_SPIN_LIMIT: () = assert!(
        SPIN_LIMIT < usize::BITS as usize,
        "SPIN_LIMIT must be lower than usize::BITS"
    );

    /// Starts an exponential backoff at the given iteration (starting from 0).
    ///
    /// This constructor can be used to skip the first smaller iterations.
    pub fn starts_at(iter: usize) -> Self {
        ExponentialBackoff { iter }
    }

    /// Returns the current count of backoff iterations performed.
    ///
    /// It can be used for example to switch to another algorithm, like thread parking, after a
    /// given iteration count.
    pub fn iter_count(&self) -> usize {
        self.iter
    }
}

impl<const SPIN_LIMIT: usize, const UNTIL_UNCHANGED_LIMIT: usize, const YIELD_AFTER: usize>
    BackoffStrategy for ExponentialBackoff<SPIN_LIMIT, UNTIL_UNCHANGED_LIMIT, YIELD_AFTER>
{
    #[inline]
    fn backoff(&mut self) -> RetryStrategy {
        let () = Self::ASSERT_SPIN_LIMIT;
        if cfg!(feature = "std") && self.iter >= YIELD_AFTER {
            #[cfg(feature = "std")]
            std::thread::yield_now();
        } else {
            for _ in 0..1usize << self.iter.min(SPIN_LIMIT) {
                spin_loop();
            }
        }
        self.iter = self.iter.saturating_add(1);
        if self.iter <= UNTIL_UNCHANGED_LIMIT {
            RetryStrategy::ReloadUntilUnchanged
        } else {
            RetryStrategy::Reload
        }
    }

    #[inline]
    fn will_reload(&self) -> bool {
        true
    }
}

/// A wrapper around a [`BackoffStrategy`] to be used in CAS loops when the atomic value must be
/// checked after each reload.
///
/// Contrary to [`BackoffStrategy::backoff_reload`], it allows checking for a termination condition
/// and early exiting the loop before performing the backoff.
///
/// In order to avoid code duplication with the checks after the reloads,
/// [`BackoffState::backoff_reload`] should be called at every iteration of the CAS loop before the
/// CAS. However, to avoid performing a backoff before any CAS failure, `BackoffState` is
/// initialized as disabled, and enabled after the first `backoff_reload` call.
///
/// # Examples
///
/// ```rust
/// # use core::sync::atomic::{AtomicUsize, Ordering};
/// # use atomic_backoff::{BackoffState, BackoffStrategy};
/// #
/// fn try_update_with_backoff(
///     atomic: &AtomicUsize,
///     set_order: Ordering,
///     fetch_order: Ordering,
///     mut f: impl FnMut(usize) -> Option<usize>,
///     strategy: impl BackoffStrategy,
/// ) -> Result<usize, usize> {
///     let mut backoff = BackoffState::new(strategy);
///     let mut current = atomic.load(fetch_order);
///     loop {
///         // Check the termination condition before backing off.
///         let new = f(current).ok_or(current)?;
///         // If the value has been reloaded, `new` must be recomputed.
///         if backoff.backoff_reload(&mut current, || atomic.load(fetch_order)) {
///             continue;
///         }
///         match atomic.compare_exchange_weak(current, new, set_order, fetch_order) {
///             Ok(x) => return Ok(x),
///             Err(cur) => current = cur,
///         }
///     }
/// }
/// ```
#[derive(Debug, Default)]
pub struct BackoffState<S> {
    strategy: S,
    enabled: bool,
}

impl<S: BackoffStrategy> BackoffState<S> {
    /// Creates a new `BackoffState` with the given backoff strategy.
    ///
    /// The backoff starts as disabled so the first iteration before any CAS failure doesn't wait.
    pub fn new(strategy: S) -> Self {
        Self {
            strategy,
            enabled: false,
        }
    }

    /// Starts the backoff in enabled mode.
    ///
    /// It is useful when the first CAS iteration is inlined in a hot function, and the complete CAS
    /// loop with the backoff is outlined in a cold function, so the backoff must start enabled
    /// after a CAS failure.
    pub fn enable(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Enables the backoff for the next iteration or perform a backoff if already enabled.
    ///
    /// Returns `true` if the current atomic value has been updated after a reload, in which case
    /// the new atomic value should be recomputed before retrying the CAS.
    ///
    /// The backoff can be temporarily disabled after a reload triggered by
    /// [`RetryStrategy::Reload`] in order to execute the CAS with the reloaded value at the next
    /// iteration.
    #[inline]
    pub fn backoff_reload<T: PartialEq, F: FnOnce() -> T>(
        &mut self,
        current: &mut T,
        reload: F,
    ) -> bool {
        if !self.enabled {
            self.enabled = true;
            return false;
        }
        let retry = self.strategy.backoff();
        if retry == RetryStrategy::NoReload {
            return false;
        }
        let reloaded = reload();
        if reloaded == *current {
            return false;
        }
        *current = reloaded;
        self.enabled = retry == RetryStrategy::ReloadUntilUnchanged;
        true
    }
}

impl<S: BackoffStrategy> From<S> for BackoffState<S> {
    fn from(strategy: S) -> Self {
        Self::new(strategy)
    }
}

/// An atomic type.
pub trait Atomic {
    /// The value of the atomic.
    type Value: Copy + PartialEq;
    /// Load the value of the atomic.
    fn load(&self, ordering: Ordering) -> Self::Value;
    /// Stores a value into the atomic if the current value is the same as the `current` value.
    fn compare_exchange_weak(
        &self,
        current: Self::Value,
        new: Self::Value,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Self::Value, Self::Value>;
}

/// Extension trait providing CAS loop methods using a given [`BackoffStrategy`].
pub trait AtomicWithBackoffExt: Atomic {
    /// Fetches the value, and applies a function to it that returns an optional new value.
    fn try_update_with_backoff<S, F>(
        &self,
        set_order: Ordering,
        fetch_order: Ordering,
        mut f: F,
        strategy: S,
    ) -> Result<Self::Value, Self::Value>
    where
        S: BackoffStrategy,
        F: FnMut(Self::Value) -> Option<Self::Value>,
    {
        let mut backoff = BackoffState::new(strategy);
        let mut current = self.load(fetch_order);
        loop {
            let new = f(current).ok_or(current)?;
            if backoff.backoff_reload(&mut current, || self.load(fetch_order)) {
                continue;
            }
            match self.compare_exchange_weak(current, new, set_order, fetch_order) {
                Ok(x) => return Ok(x),
                Err(cur) => current = cur,
            }
        }
    }

    /// Fetches the value, applies a function to it that returns a new value.
    fn update_with_backoff<S, F>(
        &self,
        set_order: Ordering,
        fetch_order: Ordering,
        mut f: F,
        mut strategy: S,
    ) -> Self::Value
    where
        S: BackoffStrategy,
        F: FnMut(Self::Value) -> Self::Value,
    {
        let mut current = self.load(fetch_order);
        loop {
            let failure_order = if strategy.will_reload() {
                Ordering::Relaxed
            } else {
                fetch_order
            };
            match self.compare_exchange_weak(current, f(current), set_order, failure_order) {
                Ok(x) => return x,
                Err(cur) => current = strategy.backoff_reload(cur, || self.load(fetch_order)),
            }
        }
    }
}

impl<T: Atomic> AtomicWithBackoffExt for T {}

macro_rules! impl_atomic {
    ($($($atomic:ident)::+ $(<$t:ident>)? => $value:ty,)*) => {$(
        impl$(<$t>)? Atomic for $($atomic)::+$(<$t>)? {
            type Value = $value;

            #[inline(always)]
            fn load(&self, ordering: Ordering) -> Self::Value {
                self.load(ordering)
            }

            #[inline(always)]
            fn compare_exchange_weak(
                &self,
                current: Self::Value,
                new: Self::Value,
                success: Ordering,
                failure: Ordering,
            ) -> Result<Self::Value, Self::Value> {
                self.compare_exchange_weak(current, new, success, failure)
            }
        }
    )*};
}

macro_rules! impl_core_atomic {
    ($($size:literal: $atomic:ident $(<$t:ident>)? => $value:ty,)*) => {$(
        #[cfg(target_has_atomic = $size)]
        impl_atomic!(core::sync::atomic::$atomic $(<$t>)? => $value,);
    )*};
}

impl_core_atomic! {
    "8": AtomicBool => bool,
    "8": AtomicI8 => i8,
    "8": AtomicU8 => u8,
    "16": AtomicI16 => i16,
    "16": AtomicU16 => u16,
    "32": AtomicI32 => i32,
    "32": AtomicU32 => u32,
    "64": AtomicI64 => i64,
    "64": AtomicU64 => u64,
    "ptr": AtomicIsize => isize,
    "ptr": AtomicUsize => usize,
    "ptr": AtomicPtr<T> => *mut T,
}

#[cfg(feature = "portable-atomic")]
macro_rules! impl_portable_atomic {
    ($($cfg:ident: $atomic:ident $(<$t:ident>)? => $value:ty,)*) => {
        portable_atomic::cfg_has_atomic_cas! {$(
            portable_atomic::$cfg! {
                impl_atomic!(portable_atomic::$atomic $(<$t>)? => $value,);
            }
        )*}
    };
}

#[cfg(feature = "portable-atomic")]
impl_portable_atomic! {
    cfg_has_atomic_8: AtomicBool => bool,
    cfg_has_atomic_8: AtomicI8 => i8,
    cfg_has_atomic_8: AtomicU8 => u8,
    cfg_has_atomic_16: AtomicI16 => i16,
    cfg_has_atomic_16: AtomicU16 => u16,
    cfg_has_atomic_32: AtomicI32 => i32,
    cfg_has_atomic_32: AtomicU32 => u32,
    cfg_has_atomic_64: AtomicI64 => i64,
    cfg_has_atomic_64: AtomicU64 => u64,
    cfg_has_atomic_128: AtomicI128 => i128,
    cfg_has_atomic_128: AtomicU128 => u128,
    cfg_has_atomic_ptr: AtomicIsize => isize,
    cfg_has_atomic_ptr: AtomicUsize => usize,
    cfg_has_atomic_ptr: AtomicPtr<T> => *mut T,
}

// loom only provides 64-bit atomics on 64-bit targets.
#[cfg(loom)]
macro_rules! impl_loom_atomic {
    ($($($width:literal:)? $atomic:ident $(<$t:ident>)? => $value:ty,)*) => {$(
        $(#[cfg(target_pointer_width = $width)])?
        impl_atomic!(loom::sync::atomic::$atomic $(<$t>)? => $value,);
    )*};
}

#[cfg(loom)]
impl_loom_atomic! {
    AtomicBool => bool,
    AtomicI8 => i8,
    AtomicU8 => u8,
    AtomicI16 => i16,
    AtomicU16 => u16,
    AtomicI32 => i32,
    AtomicU32 => u32,
    "64": AtomicI64 => i64,
    "64": AtomicU64 => u64,
    AtomicIsize => isize,
    AtomicUsize => usize,
    AtomicPtr<T> => *mut T,
}

#[cfg(test)]
mod tests {
    extern crate std;

    use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    use std::{sync::Arc, thread};

    use crate::{AtomicWithBackoffExt, BackoffLimit, BoundedBackoffStrategy, NoBackoff};

    /// Spawns two threads incrementing the same atomic, initialized to 0,
    /// and returns what each of them got.
    fn increment_twice<R: Send + 'static>(
        f: impl Fn(&AtomicUsize) -> R + Copy + Send + 'static,
    ) -> [R; 2] {
        let atomic = Arc::new(AtomicUsize::new(0));
        let spawn = || {
            let atomic = atomic.clone();
            thread::spawn(move || f(&atomic))
        };
        let (t1, t2) = (spawn(), spawn());
        [t1.join().unwrap(), t2.join().unwrap()]
    }

    #[test]
    fn update() {
        let results = increment_twice(|atomic| {
            atomic.update_with_backoff(Relaxed, Relaxed, |x| x + 1, NoBackoff)
        });
        assert!(results == [0, 1] || results == [1, 0], "{:?}", results);
    }

    #[test]
    fn try_update() {
        let results = increment_twice(|atomic| {
            let incr = |x| if x != 1 { Some(x + 1) } else { None };
            atomic.try_update_with_backoff(Relaxed, Relaxed, incr, NoBackoff)
        });
        assert!(
            results == [Ok(0), Err(1)] || results == [Err(1), Ok(0)],
            "{:?}",
            results
        );
    }

    #[test]
    fn backoff_limit() {
        let mut backoff = BackoffLimit::<NoBackoff, 2>::default();
        let mut calls = 0;
        let res = backoff.try_backoff_until(|| {
            calls += 1;
            false
        });
        assert_eq!(res, None);
        assert_eq!(calls, 3);
        assert!(backoff.is_completed());
        assert_eq!(backoff.try_backoff_until(|| Some(42)), Some(42));
        assert!(NoBackoff.is_completed());
    }
}
