from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3


class MyBot(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()
        self.kickoff_done = False

    def initialize_agent(self):
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())
        self.kickoff_done = False

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        self.boost_pad_tracker.update_boost_status(packet)

        # Continue active sequence if ongoing
        if self.active_sequence is not None and not self.active_sequence.done:
            controls = self.active_sequence.tick(packet)
            if controls is not None:
                return controls

        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        ball_location = Vec3(packet.game_ball.physics.location)
        distance_to_ball = car_location.dist(ball_location)
        time_since_kickoff = packet.game_info.seconds_elapsed

        # --- Kickoff logic ---
        if not self.kickoff_done and packet.game_info.is_kickoff_pause:
            # Center on ball, boost, and flip at right time
            kickoff_target = Vec3(0, ball_location.y, ball_location.z)
            steering = steer_toward_target(my_car, kickoff_target)
            controls = SimpleControllerState()
            controls.steer = steering
            controls.throttle = 1.0
            controls.boost = True

            # Flip when close and aligned
            if distance_to_ball < 550 and abs(steering) < 0.2 and self.active_sequence is None:
                self.kickoff_done = True
                return self.begin_front_flip(packet)
            return controls

        # --- Defensive/offensive positioning ---
        own_goal = Vec3(-4096, 0, 0) if self.team == 0 else Vec3(4096, 0, 0)
        enemy_goal = Vec3(4096, 0, 0) if self.team == 0 else Vec3(-4096, 0, 0)
        ball_x = ball_location.x
        car_x = my_car.physics.location.x

        ball_is_on_my_side = (self.team == 0 and ball_x < 0) or (self.team == 1 and ball_x > 0)
        ball_prediction = self.get_ball_prediction_struct()
        intercept_time = packet.game_info.seconds_elapsed + min(1.0, max(0.2, distance_to_ball / 2000))
        ball_in_future = find_slice_at_time(ball_prediction, intercept_time)
        predicted_ball = Vec3(ball_in_future.physics.location) if ball_in_future else ball_location

        # Target selection
        if ball_is_on_my_side:
            direction_to_goal = (own_goal - ball_location).normalized()
            shadow_distance = 400 if distance_to_ball > 800 else 200

            ball_velocity = Vec3(packet.game_ball.physics.velocity)
            if (ball_velocity.dot((own_goal - ball_location).normalized()) > 200 and abs(ball_location.x) < 1000):
                if self.team == 0:
                    left_corner = Vec3(-4096, -512, 0)
                    right_corner = Vec3(-4096, 512, 0)
                else:
                    left_corner = Vec3(4096, -512, 0)
                    right_corner = Vec3(4096, 512, 0)
                if ball_location.dist(left_corner) < ball_location.dist(right_corner):
                    corner_target = left_corner
                else:
                    corner_target = right_corner
                direction_to_corner = (corner_target - ball_location).normalized()
                target_location = ball_location + direction_to_corner * shadow_distance
            else:
                target_location = ball_location + direction_to_goal * shadow_distance
        else:
            direction_to_goal = (enemy_goal - predicted_ball).normalized()
            target_location = predicted_ball + direction_to_goal * 150

        # Smarter boost logic: only go for boost if safe and needed
        boost_target = None
        should_get_boost = my_car.boost < 20 and distance_to_ball > 1200 and not ball_is_on_my_side
        if should_get_boost:
            closest_pad = None
            closest_dist = float('inf')
            for pad in self.boost_pad_tracker.boost_pads:
                if pad.is_active and pad.is_full_boost:
                    pad_loc = Vec3(pad.location)
                    dist = car_location.dist(pad_loc)
                    # Only go for boost if not leaving goal open
                    if dist < closest_dist and (not ball_is_on_my_side or pad_loc.dist(own_goal) > 1000):
                        closest_pad = pad_loc
                        closest_dist = dist
            if closest_pad and closest_dist < 2000:
                boost_target = closest_pad

        if boost_target is not None:
            target_location = boost_target

        # Flip logic (non-kickoff): flip into ball if close and aligned
        steering_to_flip_target = steer_toward_target(my_car, predicted_ball)
        if (distance_to_ball < 140 and abs(steering_to_flip_target) < 0.25 and self.active_sequence is None):
            return self.begin_front_flip(packet)

        # Controls
        controls = SimpleControllerState()
        steering = steer_toward_target(my_car, target_location)
        controls.steer = steering

        # Throttle logic: full throttle if far, slow down if close
        aligned = abs(steering) < 0.25
        if aligned:
            if distance_to_ball > 1800:
                controls.throttle = 1.0
            elif distance_to_ball > 800:
                controls.throttle = 0.7
            elif distance_to_ball > 300:
                controls.throttle = 0.4
            else:
                controls.throttle = 0.1
        else:
            controls.throttle = 0.3

        # Boost if far from ball and not supersonic
        if distance_to_ball > 800 and car_velocity.length() < 2100 and aligned:
            controls.boost = True
        else:
            controls.boost = False

        # Debug visuals
        self.renderer.draw_line_3d(car_location, target_location, self.renderer.white())
        self.renderer.draw_string_3d(car_location, 1, 1, f'Speed: {car_velocity.length():.1f}', self.renderer.white())
        self.renderer.draw_rect_3d(target_location, 8, 8, True, self.renderer.cyan(), centered=True)

        return controls

    def begin_front_flip(self, packet):
        self.send_quick_chat(team_only=False, quick_chat=QuickChatSelection.Information_IGotIt)
        self.active_sequence = Sequence([
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.18, controls=SimpleControllerState(jump=True, pitch=-1)),
            ControlStep(duration=0.7, controls=SimpleControllerState()),
        ])
        return self.active_sequence.tick(packet)
