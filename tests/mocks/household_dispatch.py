"""A RequestDispatcher whose leaves count every C02 effect (P3-F2 matrix harness).

Lists and reminders use the real stores on disk, so mutations are counted as
changed rows; timers use a real started NamedTimerManager (T030-U1), counted
as new timers and state changes; the other leaves are spies. The home-action and slow-path
boundaries are replaced by counting stubs: a call there is the effect.
"""

from __future__ import annotations

import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.core.task_outcome import PersistenceState, TaskOutcome, TaskStatus
from src.lists.list_manager import ListManager
from src.lists.list_store import ListStore
from src.orchestrator.context_manager import ContextManager
from src.orchestrator.dispatcher import DispatchResult, PathType, Priority, RequestDispatcher
from src.orchestrator.priority_queue import PriorityRequestQueue
from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_store import ReminderScope, ReminderStore
from src.timers.named_timers import NamedTimerManager, TimerState
from tests.mocks.c02_matrix import effects

LIGHT_MATCH = {"entity_id": "light.living", "domain": "light", "service": "turn_off", "data": {},
               "description": "Luz apagada", "similarity": 1.0}


class CountedLeaves:
    """Spies and real stores shared by a dispatcher (or an orchestrator's dispatcher)."""

    def __init__(self, root) -> None:
        self.lists_db = str(root / "lists.db")
        self.reminders_db = str(root / "reminders.db")
        self.list_store = ListStore(self.lists_db)
        self.reminder_store = ReminderStore(self.reminders_db)
        self.list_manager = ListManager(self.list_store)
        self.reminder_manager = ReminderManager(self.reminder_store)
        self.chroma = MagicMock()
        self.chroma.asearch_command = AsyncMock(return_value=None)
        self.chroma.search_command = MagicMock(return_value=None)
        self.ha = AsyncMock()
        self.music = MagicMock()
        self.music.process = AsyncMock(return_value=SimpleNamespace(success=True, response="ok", details={},
                                                                    intent=SimpleNamespace(value="pause")))
        self.router = MagicMock()
        self.router.complete = AsyncMock(return_value="hola")
        self.router.generate = MagicMock(return_value=["hola"])
        self.routines = MagicMock()
        self.routines.handle = AsyncMock(return_value={"handled": True, "success": True, "response": "ok"})
        self.responses = MagicMock()
        self.responses.aspeak = AsyncMock()
        self.home = AsyncMock(return_value=SimpleNamespace(
            receipt=SimpleNamespace(status="confirmed", task_outcome=TaskOutcome(TaskStatus.COMPLETED)),
            feedback_attempted=False, feedback=None))
        self.cloud = AsyncMock(return_value=DispatchResult(
            PathType.SLOW_LLM, Priority.LOW, True, "Respuesta del modelo.", intent="conversation",
            outcome=TaskOutcome(TaskStatus.COMPLETED, persistence=PersistenceState.NOT_REQUIRED)))
        self.ring = AsyncMock()
        self.timers = NamedTimerManager(tts_callback=self.ring)
        # What main passes (pipeline.voice_services.timer_fallback_zone with zone routes): the shared-notice zone.
        self.timer_fallback_zone = "zone_living"
        self.cancels: list[bool] = []
        self.queue: PriorityRequestQueue | None = None

    async def start(self) -> None:
        await self.list_store.initialize()
        await self.reminder_store.initialize()
        await self.timers.start()

    async def close(self) -> None:
        await self.timers.stop()
        await self.list_store.close()
        await self.reminder_store.close()

    def dispatcher(self, **kwargs) -> RequestDispatcher:
        """A RequestDispatcher over these leaves with counting boundaries."""
        queue = PriorityRequestQueue()
        dispatcher = RequestDispatcher(
            chroma_sync=self.chroma, ha_client=self.ha, routine_manager=self.routines, router=self.router,
            llm=None, context_manager=ContextManager(), priority_queue=queue, music_dispatcher=self.music,
            list_manager=self.list_manager, reminder_manager=self.reminder_manager,
            response_handler=self.responses,
            **{"timer_manager": self.timers, "timer_fallback_zone": self.timer_fallback_zone, **kwargs})
        self.attach(dispatcher)
        return dispatcher

    def attach(self, dispatcher: RequestDispatcher) -> None:
        """Replace the effect boundaries of ``dispatcher`` with the counting stubs."""
        dispatcher._dispatch_home_action = self.home
        dispatcher._slow_path = self.cloud
        self.queue = dispatcher.queue
        original = dispatcher.queue.cancel_user_request

        def cancel(*args, **kwargs):
            cancelled = original(*args, **kwargs)
            self.cancels.append(bool(cancelled))
            return cancelled
        dispatcher.queue.cancel_user_request = cancel

    async def apply_state(self, dispatcher: RequestDispatcher, state: str | None) -> None:
        """Named fixtures from the entry conditions (';'-separated)."""
        for name in filter(None, (part.strip() for part in (state or "").split(";"))):
            if name == "shared_list_compras":
                created = await self.list_store.create_list("compras", "shared", None)
                await self.list_store.add_item(created.id, "pan")
            elif name == "queued_request":
                context = dispatcher.context_manager.get_household(None)
                dispatcher.queue.enqueue("unknown", "synthetic queued request", Priority.LOW,
                                         context_key=context.user_id)
            elif name.startswith("reminder_shared_active:"):
                await self.reminder_manager.create("household", name.split(":", 1)[1], time.time() + 3600,
                                                   scope=ReminderScope.SHARED)
            elif name == "light_match":
                self.chroma.asearch_command.return_value = LIGHT_MATCH
            elif name.startswith(("timer_running:", "timer_paused:")):
                kind, label = name.split(":", 1)
                timer = self.timers.create_timer(label, 600, user_id="household", zone_id="default")
                if kind == "timer_paused":
                    self.timers.pause_timer(timer.timer_id)
            elif name == "timer_running_unnamed":
                self.timers.create_timer("", 300, user_id="household", zone_id="default")
            elif name == "feature_manager_start_failed":
                await self.timers.stop()
            else:
                raise ValueError(f"unknown state fixture {name}")

    def snapshot(self) -> tuple[dict, dict, dict, dict]:
        with sqlite3.connect(self.lists_db) as db:
            lists = {row[0]: row[1:] for row in db.execute("SELECT id, name, owner_type FROM lists")}
            items = {row[0]: row[1:] for row in db.execute("SELECT id, list_id, text, completed FROM list_items")}
        with sqlite3.connect(self.reminders_db) as db:
            reminders = {row[0]: row[1] for row in db.execute("SELECT id, state FROM reminders")}
        timers = {timer.timer_id: TimerState(timer.state) for timer in self.timers.get_all_timers()}
        return lists, items, reminders, timers

    def effects(self, before: tuple[dict, dict, dict, dict], *, llm_8101_calls: int = 0,
                spoken: int | None = None) -> dict:
        """Every C02 counter since ``before``.

        list_mutations counts changed rows (lists and items, inserted,
        deleted or updated); reminder_transitions and timer_transitions count
        state changes of reminders and timers that existed before; timer_delta
        counts timers created. There is no ledger on the orchestrated
        dispatcher path, so ledger_claims is structurally zero.
        """
        lists0, items0, reminders0, timers0 = before
        lists1, items1, reminders1, timers1 = self.snapshot()
        list_mutations = (len(lists0.keys() ^ lists1.keys()) + len(items0.keys() ^ items1.keys())
                          + sum(1 for key in items0.keys() & items1.keys() if items0[key] != items1[key]))
        return effects(
            ha_service_calls=self.home.await_count + sum(1 for call in self.ha.mock_calls if "call_service" in call[0]),
            vector_searches=self.chroma.asearch_command.await_count + self.chroma.search_command.call_count,
            music_calls=self.music.process.await_count,
            list_mutations=list_mutations,
            reminder_rows_delta=len(reminders1) - len(reminders0),
            reminder_transitions=sum(1 for key in reminders0.keys() & reminders1.keys()
                                     if reminders0[key] != reminders1[key]),
            timer_delta=len(timers1) - len(timers0),
            timer_transitions=sum(1 for key in timers0.keys() & timers1.keys() if timers0[key] != timers1[key]),
            queue_cancels=sum(self.cancels),
            routine_runs=self.routines.handle.await_count,
            routine_changes=sum(1 for call in self.routines.method_calls if call[0] != "handle"),
            llm_8101_calls=self.router.complete.await_count + self.router.generate.call_count + llm_8101_calls,
            cloud_calls=self.cloud.await_count,
            spoken_utterances=self.responses.aspeak.await_count if spoken is None else spoken,
        )
