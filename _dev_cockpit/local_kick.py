"""Cloud-only one-shot action controller (kicks + episodic tricks like roulade).
Never resets/teleports. Kick uses ball_kick_left/right; tricks run an episodic
ONNX policy (e.g. roulade) for its own duration then recover to stand.
"""
import numpy as np

KICK_POLICIES = {'kickL': 'ball_kick_left', 'kickR': 'ball_kick_right'}
TRICK_POLICIES = {'roulade': 'roulade'}
TRICK_DUR_S = {'roulade': 1.4}
# Pose skills. 'sit' ships with the scene pack (alpha_sitstand); 'standup' is wired
# ahead of the policy existing, so the token is only offered once a stand-up ONNX
# is trained and exported. Availability filtering handles that automatically.
POSE_POLICIES = {'sit': 'alpha_sitstand', 'standup': 'alpha_standup'}
POSE_DUR_S = {'sit': 1.8, 'standup': 2.5}
# Skills that are allowed to fire while the duck is already down. STAND_UP is the
# whole point of being down; everything else (kicks, sit, tricks) is not.
RECOVERY_REQUESTS = ('standup',)
POLICY_FOR = {**KICK_POLICIES, **TRICK_POLICIES, **POSE_POLICIES}


class LocalKick:
    def __init__(self):
        self.name = None
        self.phase = None
        self.elapsed = 0.0
        self.stable_time = 0.0
        self.events = []

    def emit(self, status, message):
        self.events.append({'type': 'action_status', 'name': self.name,
                            'status': status, 'phase': self.phase,
                            'executor': 'cloud', 'message': message})

    def request(self, name, available, fallen):
        reason = None
        if name in KICK_POLICIES or name in TRICK_POLICIES or name in POSE_POLICIES:
            if POLICY_FOR[name] not in available:
                reason = 'Action policy is unavailable; check the server log.'
            elif self.name:
                reason = 'Another action is in progress; wait for recovery.'
            elif fallen and name not in RECOVERY_REQUESTS:
                reason = 'Duck is down; stand up first.'
        else:
            reason = 'This action requires an X5 connection in this build.'
        if reason:
            return {'type': 'action_status', 'name': name, 'status': 'rejected',
                    'executor': 'cloud', 'message': reason}
        self.name, self.phase = name, 'preparing'
        self.elapsed = self.stable_time = 0.0
        return {'type': 'action_status', 'name': name, 'status': 'accepted',
                'phase': self.phase, 'executor': 'cloud',
                'message': 'Slowing to stand before action.'}

    def finish(self, status, message):
        if self.name:
            self.emit(status, message)
        self.name = self.phase = None
        self.elapsed = self.stable_time = 0.0

    def _dur(self):
        return TRICK_DUR_S.get(self.name, POSE_DUR_S.get(self.name, 0.5))

    def select(self, dt, command, speed, gyro, upright):
        if not self.name:
            return ('alpha_walking' if np.linalg.norm(command) >= .05 else 'alpha_stand'), command
        # A recovery skill is supposed to start from a fallen pose, so the balance
        # guard must not fire for it - and it must skip the 'settle before acting'
        # gate, which can never be satisfied while lying on the ground.
        down_ok = self.name in RECOVERY_REQUESTS
        if not np.isfinite([speed, gyro, upright]).all() or (upright < .5 and not down_ok):
            self.finish('failed', 'Action aborted: duck lost balance.')
            return 'alpha_stand', np.zeros(3, dtype=np.float32)
        stable = speed < .08 and gyro < .6 and upright > .94
        self.stable_time = self.stable_time + dt if stable else 0.0
        if self.phase == 'preparing':
            if down_ok:
                self.phase, self.elapsed = 'kicking', 0.0
                self.emit('running', 'Recovery policy running.')
            elif self.elapsed >= .3 and self.stable_time >= .12:
                self.phase, self.elapsed = 'kicking', 0.0
                self.emit('running', 'Action policy running.')
            elif self.elapsed >= 2.0:
                self.finish('failed', 'Could not settle before action; stop and retry.')
        elif self.phase == 'kicking' and self.elapsed >= self._dur() - 1e-8:
            self.phase, self.elapsed, self.stable_time = 'recovering', 0.0, 0.0
            self.emit('recovering', 'Returning to standing before resuming movement.')
        elif self.phase == 'recovering':
            if self.elapsed >= .4 and self.stable_time >= .12:
                self.finish('completed', 'Action cycle finished; movement command restored.')
            elif self.elapsed >= 2.0:
                self.finish('failed', 'Recovery did not settle; check the duck before moving.')
        if not self.name:
            return 'alpha_stand', np.zeros(3, dtype=np.float32)
        policy = POLICY_FOR[self.name] if self.phase == 'kicking' else 'alpha_stand'
        self.elapsed += dt
        return policy, np.zeros(3, dtype=np.float32)
