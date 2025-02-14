"""
TODO:
- Camera simulation: if the ball isn't in the FOV, the robot can't detect it, gives [x. 0] as state
	Also max dist to ball should not be INF
"""


extends RigidBody3D

var socket = StreamPeerTCP.new()
var last_request_time = 0.0
var response_timeout = 1.0 # seconds

var current_state = [0.0, 0.0]
var episode_reward = 0.0
var reward = 0.0

var motor_speeds = [0.0, 0.0]
var max_speed = 200
const speed_multiplier = 1

var ballLoc = []
var robotLoc = Vector3(0, 0, 0)

const reward_progress_scale = 0.1

const max_iterations = 1000
var iterations = 0
var episode_count = 0

var action = [1.0, 1.0]
var state = [0, 0]
var dist = INF
var connection_status = "Disconnected"
var response_time = 0.0
var average_reward = 0.0
var total_reward = 0.0

var camera = null


func _ready():
	var error = socket.connect_to_host("127.0.0.1", 65432)

	camera = get_node("Camera3D")

	if error != OK:
		print("Je bent niet bijzonder, Job, ik kan ook tellen!")


	# Get balls locations for reset func
	for ball in get_tree().get_nodes_in_group("balls"):
		ballLoc.append(ball.global_position)

	robotLoc = global_position




func _physics_process(_delta):
	socket.poll()

	if socket.get_status() == StreamPeerTCP.STATUS_CONNECTED:
		if connection_status != "Connected":
			connection_status = "Connected"
			print("Connected to server")
		

		state = get_state()
		calculate_reward()
		var done = check_done()

		# Send to Python server
		var data = {
			"state": state,
			"reward": reward,
			"done": done
		}
		socket.put_utf8_string(JSON.stringify(data) + "\n")

		last_request_time = Time.get_ticks_msec() / 1000.0

		episode_reward += reward
		total_reward += reward
		average_reward = total_reward / (episode_count + 1)

		if done:
			print("Episode reward: ", episode_reward)
			reset_environment()
			episode_reward = 0.0
			episode_count += 1
			return


		if socket.get_available_bytes() > 0:
			var response = socket.get_utf8_string(socket.get_available_bytes())
			var json = JSON.new()
			var error = json.parse(response)
			if error == OK:
				response_time = Time.get_ticks_msec() / 1000.0 - last_request_time
				action = json.data
				action = [1, -1]
				apply_action()
			else:
				print("Error parsing JSON")
				return
	else:
		apply_action()
		if connection_status != "Disconnected":
			connection_status = "Disconnected"
			response_time = 0.0
			print("Disconnected from server")

func get_state():
	var fov_limit: float = 45.0

	var closest_angle = false
	var closest_distance = INF
	var found = false

	var forward = -global_transform.basis.z
	var forward_flat = Vector2(forward.x, forward.z).normalized()
	var self_flat = Vector2(global_transform.origin.x, global_transform.origin.z)

	for ball in get_tree().get_nodes_in_group("balls"):
		var ball_flat = Vector2(ball.global_transform.origin.x, ball.global_transform.origin.z)
		var to_ball = ball_flat - self_flat
		var distance = to_ball.length()

		var to_ball_norm = to_ball.normalized()

		var dot_val = forward_flat.dot(to_ball_norm)
		dot_val = clamp(dot_val, -1, 1)

		var angle = acos(dot_val) / PI


		if angle <= fov_limit:
			if distance < closest_distance:
				closest_distance = distance
				closest_angle = angle
				found = true
	
	return [closest_angle, 1.0 if closest_angle else 0.0]
	

func calculate_reward():
	reward = 0.0
	dist = INF

	for ball in get_tree().get_nodes_in_group("balls"):
		var prev_dist = ball.get_meta("prev_distance")
		dist = global_position.distance_to(ball.global_position)

		if prev_dist:
			reward += (prev_dist - dist) * reward_progress_scale

		ball.set_meta("prev_distance", dist)
		


		if ball.get_meta('inside_corner') and not ball.get_meta('corner_reward_given'):
			reward += 10.0
			ball.set_meta("corner_reward_given", true)

		elif not ball.get_meta('inside_corner') and ball.get_meta('corner_reward_given'):
			reward -= 10.0
			ball.set_meta("corner_reward_given", false)

		
	reward -= 0.05 # Time penalty

	return reward


func check_done():
	if iterations >= max_iterations:
		iterations = 0
		return true
	iterations += 1

	var in_corner = true
	for ball in get_tree().get_nodes_in_group("balls"):
		if ball.get_meta("corner_reward_given") == false:
			in_corner = false
			break
	
	return in_corner
	

func calculate_speed():
	return (motor_speeds[0] + motor_speeds[1]) * speed_multiplier / 2

func calculate_angular_speed():
	return (motor_speeds[0] - motor_speeds[1]) * speed_multiplier / 2


func apply_action():
	# Update motor speeds based on the action received.
	motor_speeds[0] = action[0]
	motor_speeds[1] = action[1]

	var speed = calculate_speed()
	var angular_speed = calculate_angular_speed()

	# Compute forward direction from the current Y-rotation.
	# (In Godot, forward is usually -Z but the original code used (cos, sin) from rotation.y.)
	var direction = Vector3(cos(rotation.y), 0, sin(rotation.y)).normalized()

	# Get the current linear velocity so that we only override the X and Z components.
	linear_velocity.x = direction.x * speed
	linear_velocity.z = direction.z * speed
	linear_velocity.y = 0

	# Set angular velocity to rotate around the Y axis.
	angular_velocity = Vector3(0, angular_speed, 0)

func reset_environment():
	var count = 0

	for ball in get_tree().get_nodes_in_group("balls"):
		ball.set_meta("prev_distance", null)
		ball.set_meta("corner_reward_given", false)

		ball.global_position = ballLoc[count]

		ball.linear_velocity = Vector3(0, 0, 0)
		count += 1
	
	global_position = robotLoc

	rotation.y = 0
