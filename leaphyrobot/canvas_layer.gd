extends CanvasLayer

var episode_reward = 0.0
var current_state = [0.0, 0.0]
var motor_speeds = [0.0, 0.0]
var action = [0.0, 0.0]
var distance_to_nearest_ball = 0.0
var balls_collected = 0
var balls_in_corner = 0
var episode_time = 0.0
var connection_status = "Disconnected"
var response_time = 0.0
var reward_progress = 0.0
var angular_speed = 0.0
var linear_velocity = Vector3()
var robot_position = Vector3()
var ball_positions = []
var episode_count = 0
var average_reward = 0.0
var reward_for_current_action = 0.0
var total_reward = 0.0
var collision_count = 0
var inside_corner_flag = false
var corner_reward_given = false
var previous_distance_to_ball = 0.0
var current_distance_to_ball = 0.0
var done_flag = false

func _ready():
	# Create UI elements to display the stats
	var stats_panel = Panel.new()
	stats_panel.set_size(Vector2(300, 1080))
	add_child(stats_panel)

	var vbox = VBoxContainer.new()
	stats_panel.add_child(vbox)

	vbox.add_child(create_label("Episode Reward: ", episode_reward))
	vbox.add_child(create_label("Current State: ", str(current_state)))
	vbox.add_child(create_label("Motor Speeds: ", str(motor_speeds)))
	vbox.add_child(create_label("Action: ", str(action)))
	vbox.add_child(create_label("Distance to Nearest Ball: ", distance_to_nearest_ball))
	vbox.add_child(create_label("Balls Collected: ", balls_collected))
	vbox.add_child(create_label("Balls in Corner: ", balls_in_corner))
	vbox.add_child(create_label("Episode Time: ", episode_time))
	vbox.add_child(create_label("Connection Status: ", connection_status))
	vbox.add_child(create_label("Response Time: ", response_time))
	vbox.add_child(create_label("Reward Progress: ", reward_progress))
	vbox.add_child(create_label("Angular Speed: ", angular_speed))
	vbox.add_child(create_label("Linear Velocity: ", str(linear_velocity)))
	vbox.add_child(create_label("Robot Position: ", str(robot_position)))
	vbox.add_child(create_label("Ball Positions: ", str(ball_positions)))
	vbox.add_child(create_label("Episode Count: ", episode_count))
	vbox.add_child(create_label("Average Reward: ", average_reward))
	vbox.add_child(create_label("Reward for Current Action: ", reward_for_current_action))
	vbox.add_child(create_label("Total Reward: ", total_reward))
	vbox.add_child(create_label("Collision Count: ", collision_count))
	vbox.add_child(create_label("Inside Corner Flag: ", inside_corner_flag))
	vbox.add_child(create_label("Corner Reward Given: ", corner_reward_given))
	vbox.add_child(create_label("Previous Distance to Ball: ", previous_distance_to_ball))
	vbox.add_child(create_label("Current Distance to Ball: ", current_distance_to_ball))
	vbox.add_child(create_label("Done Flag: ", done_flag))

func create_label(text, value):
	var hbox = HBoxContainer.new()
	var label = Label.new()
	label.text = text
	hbox.add_child(label)

	var value_label = Label.new()
	value_label.text = str(value)
	hbox.add_child(value_label)

	return hbox

func _process(delta):
	# Update the stats values here
	pass
