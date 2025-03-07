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

const reward_progress_scale = 1

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
				#action = [1.0, -1.0]
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

# Returns the signed angle difference (in radians) between the node's forward
# (as determined by -global_transform.basis.z) and the direction toward the ball,
# but only if the ball lies within a 45° field-of-view.
# Otherwise, it returns null.
func vision(ball):
	# Calculate the vector from this node to the ball and ignore the Y component.
	var to_ball: Vector3 = ball.global_position - global_position
	to_ball.y = 0
	
	# If the ball is exactly at our position, we can’t compute an angle.
	if to_ball.length() == 0:
		return null
	to_ball = to_ball.normalized()
	
	# Determine the node's forward direction.
	# In Godot, a Node3D's forward is usually defined as -Z.
	var forward: Vector3 = -global_transform.basis.x
	forward.y = 0
	forward = forward.normalized()
	forward = Vector3(forward.x * -1, 0, forward.z * -1)
	
	# Compute the unsigned angle between forward and the direction to the ball.
	var angle_diff: float = forward.angle_to(to_ball)
	
	# Determine the sign of the angle using the cross product.
	# If the Y component is negative, the angle is negative.
	var cross_sign: float = forward.cross(to_ball).y
	if cross_sign < 0:
		angle_diff = -angle_diff
	
	# If the ball is within a 45° field-of-view, return the angle.
	if abs(angle_diff) < PI / 4:
		return angle_diff
	else:
		return null

# Computes the distance from this node to the ball in the XZ plane.
func distance_to_ball(ball):
	var diff: Vector3 = ball.global_position - global_position
	diff.y = 0
	return diff.length()

# Loops through all nodes in the "balls" group, finds the closest ball within vision,
# and returns an array: [normalized_angle, 1.0]. If no ball is found, returns [0.0, 0.0].
func get_state():
	var balls: Array = get_tree().get_nodes_in_group("balls")
	var chosen_angle: float = 0.0
	var closest_distance: float = INF
	
	for ball in balls:
		var angle_diff = vision(ball)
		# Use an explicit check against null so an angle_diff of 0.0 is valid.
		if angle_diff != null:
			var d: float = distance_to_ball(ball)
			if d < closest_distance:
				closest_distance = d
				chosen_angle = angle_diff
	
	# If no ball was found within our FOV, return [0.0, 0.0].
	if closest_distance == INF:
		return [0.0, 0.0]
		
	# Otherwise, return the normalized angle (divided by PI) and a signal of 1.0.
	return [chosen_angle / PI, 1.0]



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
	
	global_position = get_random_loc(robotLoc)

	# rotation.y = 0
	rotation.y = randf_range(-PI / 4, PI / 4)


func get_random_loc(original_loc):
	# get the current location, and add a random value to it +/- 0.75 for z and +/- 0.25 for x
	var x = original_loc.x + randf_range(-0.25, 0.25)
	var z = original_loc.z + randf_range(-0.75, 0.75)
	var y = original_loc.y

	return Vector3(x, y, z)
