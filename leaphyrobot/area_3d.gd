extends Area3D

# Called when the node enters the scene tree for the first time.
func _ready():
	# Connect the signals to the corresponding functions
	body_entered.connect(_on_body_entered)
	body_exited.connect(_on_body_exited)

# Called when a body enters the area
func _on_body_entered(body):
	if body.is_in_group("balls"):
		body.set_meta("inside_corner", true)

# Called when a body exits the area
func _on_body_exited(body):
	if body.is_in_group("balls"):
		body.set_meta("inside_corner", false)
