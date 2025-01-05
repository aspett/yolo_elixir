defmodule YOLOXDraw do
  @font_size 18
  @stroke_width 3

  def draw_detected_objects(mat, detected_objects) do
    {:ok, image} = Image.from_evision(mat)

    detected_objects
    |> Enum.reduce(image, fn %{bbox: bbox}=od, image ->
      # left = bbox.cx
      # top = bbox.cy
      # [x0, y0, x1, y1]
      left = max(round(bbox.cx - bbox.w/2), 0)
      top = max(round(bbox.cy - bbox.h/2), 0)
      prob = round(od.prob * 100)
      color = class_color(od.class_idx)

      text_image =
        Image.Text.simple_text!("#{od.class} #{prob}%", text_fill_color: "black", font_size: @font_size)
        |> Image.Text.add_background_padding!(background_fill_color: color, padding: [5, 5])
        |> Image.Text.add_background!(background_fill_color: color)
        |> Image.split_alpha()
        |> elem(0)
      {_, text_hight, _} = Image.shape(text_image)

      image
      |> Image.Draw.rect!(left,top,bbox.w,bbox.h,[
        stroke_width: @stroke_width, color: color, fill: false
      ])
      |> Image.Draw.image!(text_image, left, max(top - text_hight - 2, 0))

    end)
  end

  @class_colors  [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#800000", "#008000", "#000080", "#FF00FF", "#800080", "#008080",
    "#C0C0C0", "#FFA500", "#A52A2A", "#8A2BE2", "#5F9EA0", "#7FFF00",
    "#D2691E", "#FF7F50", "#6495ED", "#DC143C", "#00FFFF", "#00008B",
    "#008B8B", "#B8860B", "#A9A9A9", "#006400", "#BDB76B", "#8B008B",
    "#556B2F", "#FF8C00", "#9932CC", "#8B0000", "#E9967A", "#8FBC8F",
    "#483D8B", "#2F4F4F", "#00CED1", "#9400D3", "#FF1493", "#00BFFF",
    "#696969", "#1E90FF", "#B22222", "#FFFAF0", "#228B22", "#FF00FF",
    "#DCDCDC", "#F8F8FF", "#FFD700", "#DAA520", "#808080", "#ADFF2F",
    "#F0FFF0", "#FF69B4", "#CD5C5C", "#4B0082", "#FFFFF0", "#F0E68C",
    "#E6E6FA", "#FFF0F5", "#7CFC00", "#FFFACD", "#ADD8E6", "#F08080",
    "#E0FFFF", "#FAFAD2", "#D3D3D3", "#90EE90", "#FFB6C1", "#FFA07A",
    "#20B2AA", "#87CEFA", "#778899", "#B0C4DE", "#FFFFE0", "#00FF7F",
    "#4682B4", "#D2B48C", "#008080", "#D8BFD8", "#FF6347", "#40E0D0",
    "#EE82EE", "#F5DEB3", "#FFFFFF", "#F5F5F5"
  ]
  |> Enum.with_index(&{&2, &1})
  |> Map.new()

  def class_color(class_idx) do
    Map.get(@class_colors, class_idx, "#FF0000")
  end
end
