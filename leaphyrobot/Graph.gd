extends Control

var max_points = 100
var min_value = 0.0
var max_value = 10.0
var line_color = Color(1, 0, 0)

var data = []

func add_point(value):
	data.append(value)

	if data.size() > max_points:
		data.pop_at(0)
	

func _draw():
	if data.size() < 2:
		return
	
	var width = size.x 
	var height = size.y
	var step  = width / (max_points - 1)

	for i in range(data.size() - 1):
		var x1 = i * step
		var x2 = (i + 1) * step

		var y1 = height - (data[i] - min_value) / (max_value - min_value) * height
		var y2 = height - (data[i + 1] - min_value) / (max_value - min_value) * height
		draw_line(Vector2(x1, y1), Vector2(x2, y2), line_color, 2)
