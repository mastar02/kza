"""
KZA Routines Module
Sistema completo de rutinas y automatizaciones
"""

from .routine_manager import RoutineManager, Routine, RoutineTrigger, RoutineAction, TriggerType
from .routine_scheduler import RoutineScheduler, ScheduledRoutine, TriggerType as SchedulerTriggerType
from .routine_executor import RoutineExecutor, ActionResult, ActionType
from .voice_routine_handler import VoiceRoutineHandler, VoiceRoutineIntent

__all__ = [
    # Routine Manager (original)
    'RoutineManager',
    'Routine',
    'RoutineTrigger',
    'RoutineAction',
    'TriggerType',

    # Scheduler
    'RoutineScheduler',
    'ScheduledRoutine',
    'SchedulerTriggerType',

    # Executor
    'RoutineExecutor',
    'ActionResult',
    'ActionType',

    # Voice Handler
    'VoiceRoutineHandler',
    'VoiceRoutineIntent',
]
