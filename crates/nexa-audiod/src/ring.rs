//! Bounded SPSC capture queue with a nonblocking producer and a sleeping consumer.

use std::future::poll_fn;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use std::task::Poll;

use futures_util::task::AtomicWaker;
use rtrb::{Consumer, Producer, RingBuffer};
use thiserror::Error;

use crate::dsp::FRAME_SAMPLES;

#[derive(Debug, Default)]
struct Signal {
    readable: AtomicWaker,
    writable: AtomicWaker,
    consumer_closed: AtomicBool,
    producer_closed: AtomicBool,
}

#[derive(Debug)]
pub struct CaptureProducer {
    ring: Producer<f32>,
    capacity: usize,
    signal: Arc<Signal>,
}

#[derive(Debug)]
pub struct CaptureConsumer {
    ring: Consumer<f32>,
    signal: Arc<Signal>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PushError {
    #[error("capture ring is full")]
    Full,
    #[error("capture consumer disconnected")]
    Closed,
}

/// Allocate the shared 80 ms queue once. Transfers use caller-owned slices.
#[must_use]
pub fn capture_ring() -> (CaptureProducer, CaptureConsumer) {
    capture_ring_with_frames(1)
}

/// Allocate a shared capture ring sized for `frames` whole 80 ms capture frames.
#[must_use]
pub fn capture_ring_with_frames(frames: usize) -> (CaptureProducer, CaptureConsumer) {
    assert!(frames >= 1, "capture ring requires at least 1 frame");
    let capacity = FRAME_SAMPLES * frames;
    let (producer, consumer) = RingBuffer::new(capacity);
    let signal = Arc::new(Signal::default());
    (
        CaptureProducer {
            ring: producer,
            capacity,
            signal: Arc::clone(&signal),
        },
        CaptureConsumer {
            ring: consumer,
            signal,
        },
    )
}

impl CaptureProducer {
    /// Total ring capacity in samples.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Wait for room for the complete slice. Dropping the wait leaves the ring intact.
    pub async fn push(&mut self, samples: &[f32]) -> Result<(), PushError> {
        if samples.len() > self.capacity {
            return Err(PushError::Full);
        }
        poll_fn(|cx| {
            self.signal.writable.register(cx.waker());
            let closed = self.signal.consumer_closed.load(Ordering::Acquire);
            let result = if closed {
                Err(PushError::Closed)
            } else {
                self.try_push(samples)
            };
            if result == Err(PushError::Full) {
                Poll::Pending
            } else {
                self.signal.writable.take();
                Poll::Ready(result)
            }
        })
        .await
    }

    /// Enqueue the entire block, or return immediately without writing if it does not fit.
    pub fn try_push(&mut self, samples: &[f32]) -> Result<(), PushError> {
        if self.ring.is_abandoned() || self.signal.consumer_closed.load(Ordering::Acquire) {
            return Err(PushError::Closed);
        }
        self.ring
            .push_entire_slice(samples)
            .map_err(|_| PushError::Full)?;
        if !samples.is_empty() {
            self.signal.readable.wake();
        }
        Ok(())
    }
}

impl Drop for CaptureProducer {
    fn drop(&mut self) {
        self.signal.producer_closed.store(true, Ordering::Release);
        self.signal.readable.wake();
    }
}

impl CaptureConsumer {
    /// Drain whatever is immediately available into the supplied buffer.
    pub fn try_read(&mut self, samples: &mut [f32]) -> usize {
        let count = self.ring.pop_partial_slice(samples).0.len();
        if count != 0 {
            self.signal.writable.wake();
        }
        count
    }

    /// Wait for data, returning zero on a drained, disconnected queue or an empty output slice.
    /// The receive can be cancelled while waiting without consuming samples.
    pub async fn read(&mut self, samples: &mut [f32]) -> usize {
        if samples.is_empty() {
            return 0;
        }
        poll_fn(|cx| {
            self.signal.readable.register(cx.waker());
            // Read closure before data so the last producer write is visible at EOF.
            let closed = self.signal.producer_closed.load(Ordering::Acquire);
            let read = self.try_read(samples);
            if read != 0 || closed {
                self.signal.readable.take();
                Poll::Ready(read)
            } else {
                Poll::Pending
            }
        })
        .await
    }
}

impl Drop for CaptureConsumer {
    fn drop(&mut self) {
        self.signal.consumer_closed.store(true, Ordering::Release);
        self.signal.writable.wake();
    }
}
