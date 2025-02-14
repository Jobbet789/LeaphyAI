extends Node3D

var camera = null
var robot = null
var current = true

func _ready():
	camera = get_node("Camera3D")
	robot = get_node("Robot")

	camera.set_current(true)

func _on_button_pressed():
	camera.set_current(not current)
	current = not current
