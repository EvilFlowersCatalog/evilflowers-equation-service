"""The task contract with the analyzer: name, queue and argument shape."""

import inspect

from evilflowers_equation.worker import app

TASKS = {"evilflowers_equation.equations": "evilflowers_equation"}


def test_every_task_is_registered_and_routed():
    app.loader.import_default_modules()
    for name, queue in TASKS.items():
        assert name in app.tasks, name
        assert app.amqp.router.route({}, name)["queue"].name == queue


def test_tasks_accept_arguments_a_newer_analyzer_adds():
    app.loader.import_default_modules()
    for name in TASKS:
        parameters = inspect.signature(app.tasks[name].run).parameters.values()
        assert any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
        assert app.tasks[name].acks_late


def test_redelivery_settings():
    assert app.conf.worker_prefetch_multiplier == 1
    assert app.conf.task_reject_on_worker_lost is True
    assert app.conf.broker_transport_options["visibility_timeout"] >= 3600
