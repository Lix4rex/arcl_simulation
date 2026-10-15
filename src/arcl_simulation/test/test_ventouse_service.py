"""Exercise the compiled service with a ROS backend, without Gazebo or a ROS clock."""

import os
from pathlib import Path
import subprocess
import time

from ament_index_python.packages import get_package_prefix
import pytest
import rclpy
from std_msgs.msg import String
from std_srvs.srv import SetBool


@pytest.fixture
def backend(tmp_path, monkeypatch, request):
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path / 'ros_logs'))
    monkeypatch.setenv('ROS_DOMAIN_ID', str(100 + os.getpid() % 100))
    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')
    rclpy.init()
    node = rclpy.create_node('ventouse_test_backend')
    commands = []
    publisher = node.create_publisher(String, '/robot_omni/ventouse/resultat', 10)

    def on_command(msg):
        version, identifier, operation, _ = msg.data.split()
        if operation == 'PING' and getattr(request, 'param', True):
            publisher.publish(String(data=f'{version} {identifier} OK READY'))
        else:
            commands.append(msg)

    subscription = node.create_subscription(
        String, '/robot_omni/ventouse/commande', on_command, 10)
    client = node.create_client(SetBool, '/robot_omni/ventouse/set')
    executable = os.environ.get('VENTOUSE_SERVICE_EXECUTABLE') or str(
        Path(get_package_prefix('arcl_simulation')) / 'lib/arcl_simulation/ventouse_service')
    process = subprocess.Popen([executable])

    def wait(predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            assert process.poll() is None, 'service process exited'
        assert predicate(), 'timed out waiting for service/backend'

    try:
        wait(lambda: client.service_is_ready() and publisher.get_subscription_count() > 0
             and subscription.get_publisher_count() > 0)
        yield node, commands, publisher, client, wait
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        node.destroy_node()
        rclpy.shutdown()


def request(client, take):
    return client.call_async(SetBool.Request(data=take))


def command(commands, wait, count):
    wait(lambda: len(commands) == count)
    version, identifier, operation, expiration = commands[-1].data.split()
    assert version == '1'
    assert 0 < int(expiration) - int(time.time() * 1000) <= 2000
    return identifier, operation


def result(publisher, identifier, status, code):
    publisher.publish(String(data=f'1 {identifier} {status} {code}'))


@pytest.mark.parametrize('backend', [False], indirect=True)
def test_unresponsive_backend_at_startup_allows_retry(backend):
    _, commands, publisher, client, wait = backend
    first = request(client, True)
    wait(first.done, timeout=4.0)
    assert not first.result().success
    assert first.result().message == 'Backend de ventouse indisponible'
    assert commands and all(msg.data.split()[2] == 'PING' for msg in commands)
    old_id = commands[0].data.split()[1]

    second = request(client, False)
    wait(lambda: any(msg.data.split()[1] != old_id for msg in commands))
    new_id = commands[-1].data.split()[1]
    result(publisher, old_id, 'OK', 'READY')  # stale readiness must not send an action
    result(publisher, new_id, 'OK', 'READY')
    result(publisher, new_id, 'OK', 'READY')  # duplicate readiness must not resend it
    def actions():
        return [msg.data.split() for msg in commands if msg.data.split()[2] != 'PING']

    wait(lambda: len(actions()) >= 1)
    assert [parts[1:3] for parts in actions()] == [[new_id, 'RELEASE']]
    result(publisher, new_id, 'OK', 'ALREADY_FREE')
    wait(second.done)
    assert second.result().success
    assert len(actions()) == 1


def test_concurrency_correlation_and_deferred_result(backend):
    node, commands, publisher, client, wait = backend
    first = request(client, True)
    identifier, operation = command(commands, wait, 1)
    assert operation == 'TAKE'
    assert not first.done()

    second = request(client, False)
    wait(second.done, timeout=1.0)
    assert not second.result().success
    assert second.result().message == 'Ventouse occupée'
    assert len(commands) == 1

    result(publisher, 'old-id', 'OK', 'TAKEN')
    result(publisher, identifier, 'OK', 'TIMEOUT')  # invalid status/code combination
    result(publisher, identifier, 'OK', 'TAKEN extra')
    deadline = time.monotonic() + 0.15
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
    assert not first.done()

    result(publisher, identifier, 'OK', 'TAKEN')
    wait(first.done)
    assert first.result().success
    third = request(client, False)
    new_id, operation = command(commands, wait, 2)
    assert new_id != identifier and operation == 'RELEASE'
    result(publisher, identifier, 'OK', 'TAKEN')  # duplicate old result
    result(publisher, new_id, 'OK', 'RELEASED')
    wait(third.done)
    assert third.result().success


@pytest.mark.parametrize('take,status,code,success', [
    (True, 'OK', 'ALREADY_HELD', True),
    (False, 'OK', 'ALREADY_FREE', True),
    (True, 'ERROR', 'TIMEOUT', False),
    (False, 'ERROR', 'NOT_READY', False),
    (True, 'ERROR', 'BUSY', False),
    (True, 'ERROR', 'INVALID_COMMAND', False),
])
def test_terminal_results(backend, take, status, code, success):
    _, commands, publisher, client, wait = backend
    future = request(client, take)
    identifier, _ = command(commands, wait, 1)
    result(publisher, identifier, status, code)
    wait(future.done)
    assert future.result().success == success
    assert future.result().message


def test_watchdog_keeps_reservation_until_late_terminal_result(backend):
    _, commands, publisher, client, wait = backend
    start = time.monotonic()
    first = request(client, True)
    identifier, _ = command(commands, wait, 1)
    wait(first.done, timeout=4.0)
    assert 2.8 <= time.monotonic() - start < 4.5
    assert not first.result().success
    assert first.result().message == 'Résultat de la simulation indisponible'

    blocked = request(client, False)
    wait(blocked.done)
    assert blocked.result().message == 'Ventouse occupée'
    assert len(commands) == 1
    result(publisher, identifier, 'ERROR', 'TIMEOUT')
    # Publishing a late result and the next request uses different DDS endpoints;
    # wait until the subscription has processed the result before making that request.
    deadline = time.monotonic() + 0.15
    while time.monotonic() < deadline:
        rclpy.spin_once(backend[0], timeout_sec=0.02)
    next_request = request(client, False)
    next_id, _ = command(commands, wait, 2)
    result(publisher, next_id, 'OK', 'ALREADY_FREE')
    wait(next_request.done)
    assert next_request.result().success
    assert first.result().message == 'Résultat de la simulation indisponible'
