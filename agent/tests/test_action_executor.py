"""Exercise policy action dispatch through typed ROS client contracts."""
import asyncio
import math
import sys
import time
import unittest
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

from agent.architecture.action_executor import ActionExecutor
from agent.tests.test_training_flow import ToyBackend


class Future:
    def __init__(self, value=None, ready=True):
        self.value, self.ready = value, ready

    def done(self):
        return self.ready

    def result(self):
        return self.value


class Handle:
    accepted = True

    def __init__(self):
        self.result_future = Future(ready=False)
        self.cancel_calls = 0
        self.cancel_future = Future(NS(goals_canceling=[object()]))

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_calls += 1
        return self.cancel_future


class Client:
    def __init__(self, result):
        self.future = Future(result)
        self.requests = []
        self.ready = True

    def server_is_ready(self):
        return self.ready

    service_is_ready = server_is_ready

    def send_goal_async(self, goal):
        self.requests.append(goal)
        return self.future

    call_async = send_goal_async


def ros_modules():
    action, srv = ModuleType('robot_msgs.action'), ModuleType('std_srvs.srv')
    action.GoToPose = NS(Goal=lambda: NS())
    srv.SetBool = NS(Request=lambda: NS(data=False))
    return {'robot_msgs': ModuleType('robot_msgs'), 'robot_msgs.action': action,
            'std_srvs': ModuleType('std_srvs'), 'std_srvs.srv': srv}


class ActionDispatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, ros_modules())
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.executor = ActionExecutor()
        self.executor._node = object()  # typed clients injected below
        self.world = ToyBackend().reset()['observations']['blue']
        self.handle = Handle()
        self.executor._move_client = Client(self.handle)

    async def move(self):
        return await self.executor.dispatch(
            {'action': 'MOVE', 'world_sequence': self.world['sequence'],
             'target_pose': {'x': 1., 'y': 1., 'sin': 2., 'cos': 0.}}, self.world)

    async def test_move_uses_action_and_waits_for_remote_result(self):
        self.assertEqual((await self.move())['status'], 'issuing')
        goal = self.executor._move_client.requests[0]
        self.assertEqual((goal.x, goal.y, goal.theta), (1., 1., math.pi / 2))
        await self.executor.update(self.world)
        self.assertEqual(self.executor.active['status'], 'running')
        self.handle.result_future.value = NS(status=4, result=NS(success=True, message='arrived'))
        self.handle.result_future.ready = True
        await self.executor.update(self.world)
        self.assertEqual(self.executor.last_result['status'], 'succeeded')
        self.assertIsNone(self.executor.active)

    async def test_cancel_before_acceptance_waits_for_canceled_result(self):
        self.executor._move_client.future.ready = False
        await self.move()
        result = await self.executor.dispatch({'action': 'CANCEL', 'world_sequence': self.world['sequence']}, self.world)
        self.assertEqual(result['status'], 'cancel_requested')
        self.assertEqual(self.handle.cancel_calls, 0)
        self.executor._move_client.future.ready = True
        await self.executor.update(self.world)
        self.assertEqual(self.handle.cancel_calls, 1)
        self.assertIsNotNone(self.executor.active)
        self.handle.result_future.value = NS(status=5, result=NS(success=False, message='canceled'))
        self.handle.result_future.ready = True
        await self.executor.update(self.world)
        self.assertEqual(self.executor.last_result['status'], 'canceled')

    async def test_rejected_and_unavailable_server(self):
        self.executor._move_client.ready = False
        self.assertEqual((await self.move())['status'], 'failed')
        self.executor._move_client.ready = True
        self.handle.accepted = False
        await self.move()
        await self.executor.update(self.world)
        self.assertEqual(self.executor.last_result['status'], 'rejected')

    async def test_timeout_keeps_lock_until_late_abort(self):
        await self.move()
        self.executor.active['started_at'] = time.monotonic() - 100
        await self.executor.update(self.world)
        self.assertEqual(self.executor.active['status'], 'unknown')
        self.assertEqual(self.handle.cancel_calls, 1)
        self.handle.result_future.value = NS(status=6, result=NS(success=False, message='timeout'))
        self.handle.result_future.ready = True
        await self.executor.update(self.world)
        self.assertEqual(self.executor.last_result['status'], 'failed')

    async def test_cancel_rejection_preserves_running_goal(self):
        await self.move()
        await self.executor.update(self.world)
        self.handle.cancel_future.value = NS(goals_canceling=[])
        await self.executor.dispatch({'action': 'CANCEL', 'world_sequence': self.world['sequence']}, self.world)
        await self.executor.update(self.world)
        self.assertEqual(self.executor.active['status'], 'running')
        self.assertTrue(self.executor.active['cancelable'])
        self.assertIsNone(self.executor.last_result)

    async def test_service_failure_and_unavailability_are_reported(self):
        client = Client(NS(success=False, message='backend unavailable'))
        self.executor._suction_all_client = client
        for ready in (True, False):
            client.ready = ready
            await self.executor.dispatch({'action': 'SUCTION_ALL_OFF',
                                          'world_sequence': self.world['sequence']}, self.world)
            await asyncio.sleep(0)
            await self.executor.update(self.world)
            self.assertEqual(self.executor.last_result['status'], 'failed')
            self.assertIsNone(self.executor.active)

    async def test_every_suction_and_wrist_request(self):
        clients = {i: Client(NS(success=True, message='ok')) for i in range(1, 4)}
        aggregate, wrist = Client(NS(success=True, message='ok')), Client(NS(success=True, message='ok'))
        self.executor._suction_clients = clients
        self.executor._suction_all_client = aggregate
        self.executor._wrist_client = wrist
        for index in (1, 2, 3, 'ALL'):
            for state in ('ON', 'OFF'):
                action = f'SUCTION_{index}_{state}'
                await self.executor.dispatch({'action': action, 'world_sequence': self.world['sequence']}, self.world)
                await asyncio.sleep(0)
                await self.executor.update(self.world)
                client = aggregate if index == 'ALL' else clients[index]
                self.assertEqual(client.requests[-1].data, state == 'ON')
                self.assertEqual(self.executor.last_result['status'], 'succeeded')
        for angle, expected in ((0., True), (math.pi / 2, False)):
            self.world['robot']['wrist_angle_rad'] = angle
            await self.executor.dispatch({'action': 'TOGGLE_WRIST', 'world_sequence': self.world['sequence']}, self.world)
            await asyncio.sleep(0)
            await self.executor.update(self.world)
            self.assertEqual(wrist.requests[-1].data, expected)
        result = await self.executor.dispatch({'action': 'CANCEL', 'world_sequence': self.world['sequence']}, self.world)
        self.assertEqual(result['status'], 'rejected')
