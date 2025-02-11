extends CharacterBody3D

const SPEED = 3
const JUMP_VELOCITY = 4.5
const gravity_accel = -9.81 * 5

var socket = StreamPeerTCP.new()
var waiting_for_response = false
var last_request_time = 0.0
var response_timeout = 1.0 # seconds

var current_state = [0.0, 0.0]
var episode_reward = 0.0

var motor_speeds = [0.0, 0.0]
var max_speed = 200
const speed_multiplier = 1

var detection_radius = 200

var ballLoc = []
var robotLoc = Vector3(0, 0, 0)



func _ready():
	socket.set_no_delay(true)
	var error = socket.connect_to_host("127.0.0.1", 65432)

	if error != OK:
		print("Je bent niet bijzonder, Job, ik kan ook tellen!")


	# Get balls locations for reset func
	for ball in get_tree().get_nodes_in_group("balls"):
		ballLoc.append(ball.global_position)

	robotLoc = global_position



func _physics_process(delta: float):
	socket.poll()

	if waiting_for_response:
		if Time.get_ticks_msec() / 1000.0 - last_request_time > response_timeout:
			print("Response timeout")
			waiting_for_response = false

	if socket.get_status() == StreamPeerTCP.STATUS_CONNECTED:
		if not waiting_for_response:
			var state = get_state()
			var reward = calculate_reward()
			var done = check_done()

			# Send to Python server
			var data = {
				"state": state,
				"reward": reward,
				"done": done
			}
			socket.put_utf8_string(JSON.stringify(data) + "\n")

			waiting_for_response = true
			last_request_time = Time.get_ticks_msec() / 1000.0
			episode_reward += reward

			if done:
				print("Episode reward: ", episode_reward)
				reset_environment()
				episode_reward = 0.0
		else:
			if socket.get_available_bytes() > 0:
				var response = socket.get_utf8_string(socket.get_available_bytes())
				var json = JSON.new()
				var error = json.parse(response)
				if error == OK:
					var action = json.data
					apply_action(action, delta)
					waiting_for_response = false
				else:
					print("Error parsing JSON")
					return
	else:
		apply_action([1.0, 1.0], delta)

	

func get_state():
	# Same as previous implementation
	# Return [angle_to_nearest_ball, detection_flag]
	# angle_to_nearest_ball: angle between robot and nearest ball in radians
	# detection_flag: 1 if ball is detected, 0 otherwise

	var balls = get_tree().get_nodes_in_group("balls")
	var closest_ball = null
	var min_dist = INF

	for ball in balls:
		var dist = global_position.distance_to(ball.global_position)
		if dist < min_dist:
			min_dist = dist
			closest_ball = ball
	
	if closest_ball:
		var direction = closest_ball.global_position - global_position
		var angle = direction.angle_to(Vector3(1, 0, 0)) - rotation.y

		angle = fposmod(angle + PI, TAU) - PI

		return [angle / PI, 1.0 if min_dist < detection_radius else 0.0]
	else:
		return [0.0, 0]

func calculate_reward():
	var reward = 0.0

	var target_corner = Vector3(10, 0, 10) # Change to actual corner pos
	var corner_threshold = 20 # Change to actual threshold
	var progress_scale = 0.1

	for ball in get_tree().get_nodes_in_group("balls"):
		var dist = ball.global_position.distance_to(target_corner)

		if not ball.has_meta("prev_distance"):
			ball.set_meta("prev_distance", dist)

		var prev_dist = ball.get_meta("prev_distance")

		reward += (prev_dist - dist) * progress_scale

		ball.set_meta("prev_distance", dist)

		if dist < corner_threshold and not ball.has_meta("corner_reward_given"):
			reward += 10.0
			ball.set_meta("corner_reward_given", true)
		
	reward -= 0.05 # Time penalty

	return reward



func check_done():
	var in_corner = true
	for ball in get_tree().get_nodes_in_group("balls"):
		if ball.get_meta("corner_reward_given") == false:
			in_corner = false
			break
	
	#return in_corner
	return false
	

func calculate_speed():
	return (motor_speeds[0] + motor_speeds[1]) * speed_multiplier / 2

func calculate_angular_speed():
	return (motor_speeds[0] - motor_speeds[1]) * speed_multiplier / 2

func apply_action(action, delta):
	# This is for no time scaling
	# var delta_copy = delta 
	var delta_copy = 1

	motor_speeds[0] = action[0]
	motor_speeds[1] = action[1]

	if not is_on_floor():
		velocity.y += gravity_accel * delta_copy

	var speed = calculate_speed() * delta_copy
	var angular_speed = calculate_angular_speed() * delta_copy

	var direction = Vector3(cos(rotation.y), 0, sin(rotation.y)).normalized()

	velocity.x = direction.x * speed
	velocity.z = direction.z * speed
	rotation.y += angular_speed * delta_copy

	move_and_slide()

func reset_environment():
	var count = 0

	for ball in get_tree().get_nodes_in_group("balls"):
		ball.set_meta("prev_distance", null)
		ball.set_meta("corner_reward_given", false)

		ball.global_position = ballLoc[count]
		count += 1
	
	global_position = robotLoc

	rotation.y = 0
