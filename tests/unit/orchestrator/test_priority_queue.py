"""
Tests for PriorityRequestQueue - Priority-based request queue.
"""

import pytest
import asyncio
import time
from unittest.mock import MagicMock

from src.orchestrator.priority_queue import (
    Priority,
    Request,
    RequestStatus,
    PriorityRequestQueue
)


class TestPriority:
    """Tests for Priority enum"""

    def test_priority_ordering(self):
        # Lower value = higher priority
        assert Priority.CRITICAL.value < Priority.HIGH.value
        assert Priority.HIGH.value < Priority.MEDIUM.value
        assert Priority.MEDIUM.value < Priority.LOW.value

    def test_priority_values(self):
        assert Priority.CRITICAL.value == 0
        assert Priority.HIGH.value == 1
        assert Priority.MEDIUM.value == 2
        assert Priority.LOW.value == 3


class TestRequest:
    """Tests for Request dataclass"""

    def test_create_request(self):
        req = Request(
            request_id="req_1",
            user_id="user_1",
            text="Prende la luz",
            priority=Priority.HIGH
        )
        assert req.request_id == "req_1"
        assert req.user_id == "user_1"
        assert req.priority == Priority.HIGH
        assert req.status == RequestStatus.PENDING

    def test_request_ordering(self):
        """Requests should order by (priority, timestamp)"""
        req1 = Request(
            request_id="req_1",
            user_id="user_1",
            text="Low priority",
            priority=Priority.LOW
        )
        time.sleep(0.01)
        req2 = Request(
            request_id="req_2",
            user_id="user_2",
            text="High priority",
            priority=Priority.HIGH
        )

        # Higher priority (lower value) should come first
        assert req2 < req1

    def test_same_priority_fifo(self):
        """Same priority requests should be FIFO"""
        req1 = Request(
            request_id="req_1",
            user_id="user_1",
            text="First",
            priority=Priority.MEDIUM
        )
        time.sleep(0.01)
        req2 = Request(
            request_id="req_2",
            user_id="user_2",
            text="Second",
            priority=Priority.MEDIUM
        )

        # Earlier timestamp should come first
        assert req1 < req2

    def test_cancel_request(self):
        req = Request(
            request_id="req_1",
            user_id="user_1",
            text="Test",
            priority=Priority.LOW
        )

        req.cancel()
        assert req.status == RequestStatus.CANCELLED
        assert req.is_cancelled

    def test_complete_request(self):
        req = Request(
            request_id="req_1",
            user_id="user_1",
            text="Test",
            priority=Priority.LOW
        )

        req.complete("Result text")
        assert req.status == RequestStatus.COMPLETED
        assert req.result == "Result text"

    def test_fail_request(self):
        req = Request(
            request_id="req_1",
            user_id="user_1",
            text="Test",
            priority=Priority.LOW
        )

        req.fail("Error message")
        assert req.status == RequestStatus.FAILED
        assert req.error == "Error message"


class TestPriorityRequestQueue:
    """Tests for PriorityRequestQueue"""

    def test_create_queue(self):
        queue = PriorityRequestQueue(max_queue_size=100)
        assert len(queue) == 0

    def test_enqueue_request(self):
        queue = PriorityRequestQueue()

        req = queue.enqueue(
            user_id="user_1",
            text="Test request",
            priority=Priority.MEDIUM
        )

        assert req is not None
        assert req.user_id == "user_1"
        assert len(queue) == 1

    def test_priority_ordering_in_queue(self):
        queue = PriorityRequestQueue()

        # Enqueue in wrong order
        req_low = queue.enqueue("user_1", "Low", Priority.LOW)
        req_high = queue.enqueue("user_2", "High", Priority.HIGH)
        req_critical = queue.enqueue("user_3", "Critical", Priority.CRITICAL)

        # Dequeue should return by priority
        assert queue.dequeue() == req_critical
        assert queue.dequeue() == req_high
        assert queue.dequeue() == req_low

    def test_auto_cancel_previous(self):
        queue = PriorityRequestQueue(auto_cancel_previous=True)

        # First request from user
        req1 = queue.enqueue("user_1", "First request", Priority.LOW)

        # Second request from same user
        req2 = queue.enqueue("user_1", "Second request", Priority.LOW)

        # First should be cancelled
        assert req1.status == RequestStatus.CANCELLED
        assert req2.status == RequestStatus.PENDING

    def test_no_auto_cancel(self):
        queue = PriorityRequestQueue(auto_cancel_previous=False)

        req1 = queue.enqueue("user_1", "First request", Priority.LOW)
        req2 = queue.enqueue("user_1", "Second request", Priority.LOW)

        # Both should be pending
        assert req1.status == RequestStatus.PENDING
        assert req2.status == RequestStatus.PENDING

    def test_cancel_user_request(self):
        queue = PriorityRequestQueue(auto_cancel_previous=False)

        req1 = queue.enqueue("user_1", "Request 1", Priority.LOW)
        req2 = queue.enqueue("user_2", "Request 2", Priority.LOW)

        # Cancel user_1's request
        cancelled = queue.cancel_user_request("user_1")

        assert cancelled is True
        assert req1.is_cancelled
        assert not req2.is_cancelled

    def test_dequeue_skips_cancelled(self):
        queue = PriorityRequestQueue()

        req1 = queue.enqueue("user_1", "Will cancel", Priority.HIGH)
        req2 = queue.enqueue("user_2", "Will keep", Priority.LOW)

        req1.cancel()

        # Should skip cancelled and return req2
        result = queue.dequeue()
        assert result == req2

    def test_dequeue_empty(self):
        queue = PriorityRequestQueue()
        result = queue.dequeue(timeout=0.1)
        assert result is None

    def test_get_position(self):
        queue = PriorityRequestQueue()

        req1 = queue.enqueue("user_1", "First", Priority.LOW)
        req2 = queue.enqueue("user_2", "Second", Priority.LOW)
        req3 = queue.enqueue("user_3", "Third", Priority.LOW)

        assert queue.get_position(req1.request_id) == 1
        assert queue.get_position(req2.request_id) == 2
        assert queue.get_position(req3.request_id) == 3

    def test_get_position_with_priority(self):
        queue = PriorityRequestQueue()

        req_low = queue.enqueue("user_1", "Low", Priority.LOW)
        req_high = queue.enqueue("user_2", "High", Priority.HIGH)

        # High priority should be position 1
        assert queue.get_position(req_high.request_id) == 1
        assert queue.get_position(req_low.request_id) == 2

    def test_callbacks(self):
        queue = PriorityRequestQueue()

        completed_results = []
        cancelled_results = []

        def on_complete(req):
            completed_results.append(req.result)

        def on_cancel(req):
            cancelled_results.append(req.request_id)

        req = queue.enqueue(
            "user_1",
            "Test",
            Priority.MEDIUM,
            on_complete=on_complete,
            on_cancel=on_cancel
        )

        # Test complete callback
        req.complete("Done!")
        assert "Done!" in completed_results

        # Create another and test cancel callback
        req2 = queue.enqueue(
            "user_2",
            "Test 2",
            Priority.MEDIUM,
            on_cancel=on_cancel
        )
        req2.cancel()
        assert req2.request_id in cancelled_results

    def test_get_stats(self):
        queue = PriorityRequestQueue()

        queue.enqueue("user_1", "Req 1", Priority.HIGH)
        queue.enqueue("user_2", "Req 2", Priority.LOW)

        queue.enqueue("user_3", "Req 3", Priority.MEDIUM)
        queue.cancel_user_request("user_3")

        stats = queue.get_stats()
        assert stats["queue_size"] == 2  # Excluding cancelled
        assert stats["total_enqueued"] == 3
        assert stats["total_cancelled"] >= 1

    def test_max_size_limit(self):
        queue = PriorityRequestQueue(max_queue_size=2)

        queue.enqueue("user_1", "Req 1", Priority.HIGH)
        queue.enqueue("user_2", "Req 2", Priority.HIGH)

        # Third LOW priority should fail when queue is full
        from src.orchestrator.priority_queue import QueueFullError
        import pytest
        with pytest.raises(QueueFullError):
            queue.enqueue("user_3", "Req 3", Priority.LOW)


class TestPriorityRequestQueueAsync:
    """Async tests for PriorityRequestQueue"""

    @pytest.mark.asyncio
    async def test_dequeue_async(self):
        queue = PriorityRequestQueue()

        req = queue.enqueue("user_1", "Test", Priority.HIGH)

        result = await queue.dequeue_async(timeout=1.0)
        assert result == req

    @pytest.mark.asyncio
    async def test_dequeue_async_waits(self):
        queue = PriorityRequestQueue()

        async def delayed_enqueue():
            await asyncio.sleep(0.1)
            queue.enqueue("user_1", "Delayed", Priority.HIGH)

        # Start dequeue before enqueue
        task = asyncio.create_task(queue.dequeue_async(timeout=2.0))
        await asyncio.create_task(delayed_enqueue())

        result = await task
        assert result is not None
        assert result.text == "Delayed"

    @pytest.mark.asyncio
    async def test_dequeue_async_timeout(self):
        queue = PriorityRequestQueue()

        result = await queue.dequeue_async(timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test queue with multiple concurrent producers/consumers"""
        queue = PriorityRequestQueue()
        results = []

        async def producer(user_id, count):
            for i in range(count):
                queue.enqueue(user_id, f"Msg {i}", Priority.MEDIUM)
                await asyncio.sleep(0.01)

        async def consumer():
            while True:
                req = await queue.dequeue_async(timeout=0.5)
                if req is None:
                    break
                results.append(req)

        # Start producers
        producers = [
            asyncio.create_task(producer(f"user_{i}", 3))
            for i in range(3)
        ]

        # Start consumer
        consumer_task = asyncio.create_task(consumer())

        # Wait for producers
        await asyncio.gather(*producers)

        # Give consumer time to process
        await asyncio.sleep(0.2)
        consumer_task.cancel()

        try:
            await consumer_task
        except asyncio.CancelledError:
            pass

        # Should have processed all 9 requests
        assert len(results) == 9
