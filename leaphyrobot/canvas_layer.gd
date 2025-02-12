extends CanvasLayer

var robot = null
var stats = [
	{ "label": "Episode Reward", "property": "episode_reward", "format": "%.3f" },
	{ "label": "Current State", "property": "state", "format": "%.3f, %.3f" },
	{ "label": "Action", "property": "action", "format": "%.3f, %.3f" },
	{ "label": "Distance to Nearest Ball", "property": "dist", "format": "%.3f" },
	{ "label": "Balls in Corner", "format": "%d", "custom": true },
	{ "label": "Episode Iterations", "property": "iterations", "format": "%d" },
	{ "label": "Connection Status", "property": "connection_status", "format": "%s" },
	{ "label": "Response Time", "property": "response_time", "format": "%.3f" },
	{ "label": "Angular Speed", "format": "%.3f", "custom": true },
	{ "label": "Linear Velocity", "format": "%.3f, %.3f", "custom": true },
	{ "label": "Robot Position", "format": "%.3f, %.3f", "custom": true },
	{ "label": "Episode Count", "property": "episode_count", "format": "%d" },
	{ "label": "Average Episode Reward", "property": "average_reward", "format": "%.3f" },
	{ "label": "Reward for Current Action", "property": "reward", "format": "%.3f" },
	{ "label": "Episode Reward", "property": "episode_reward", "format": "%.3f" },
]
var value_labels = []

func _ready():
	var stats_panel = Panel.new()
	stats_panel.size = Vector2(300, 1080)
	add_child(stats_panel)
	
	robot = get_node("../../Robot")
	
	var vbox = VBoxContainer.new()
	stats_panel.add_child(vbox)
	
	for entry in stats:
		var hbox = HBoxContainer.new()
		var label = Label.new()
		label.text = entry["label"] + ": "
		hbox.add_child(label)
		
		var value_label = Label.new()
		hbox.add_child(value_label)
		vbox.add_child(hbox)
		value_labels.append(value_label)

func _process(delta):
	for i in range(stats.size()):
		var entry = stats[i]
		var value_label = value_labels[i]
		var value = null
		
		if entry.get("custom", false):
			if entry["label"] == "Balls in Corner":
				value = get_balls_in_corner()
			if entry["label"] == "Angular Speed":
				value = robot.angular_velocity.y
			if entry["label"] == "Robot Position":
				value = [robot.global_position.x, robot.global_position.z]
			if entry["label"] == "Linear Velocity":
				value = [robot.linear_velocity.x, robot.linear_velocity.z]
		else:
			value = robot.get(entry["property"])
		
		var args = value if value is Array else [value]
		value_label.text = entry["format"] % args

func get_balls_in_corner():
	var count = 0
	for ball in get_tree().get_nodes_in_group("balls"):
		if ball.get_meta("inside_corner"):
			count += 1
	return count
