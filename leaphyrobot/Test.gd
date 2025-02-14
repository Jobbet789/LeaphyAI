extends Panel
var draw_ = false

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta):
	if not draw_:
		draw_line(Vector2(0, 0), Vector2(100, 100), Color(1, 0, 0), 2)
		draw_ = true
	
