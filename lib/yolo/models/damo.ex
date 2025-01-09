defmodule YOLO.Models.Damo do
  @moduledoc """
  DAMO-YOLO model implementation for preprocessing input images
  and postprocessing detections using non-maximum suppression (NMS).

  Supports DAMO-YOLO models found at [https://github.com/tinyvision/DAMO-YOLO?tab=readme-ov-file#general-models](github.com/tinyvision/DAMO-YOLO)

  Assumes the use of pre-exported ONNX models which have decoding in inference.
  """

  @behaviour YOLO.Model

  import Nx.Defn

  @impl true
  @spec preprocess(YOLO.Model.t(), term(), Keyword.t()) :: {Nx.Tensor.t(), ScalingConfig}
  def preprocess(model, image, options) do
    frame_scaler = Keyword.fetch!(options, :frame_scaler)
    {_, _channels, height, width} = model.shapes.input
    {image_nx, image_scaling} = YOLO.FrameScalers.fit(image, {height, width}, frame_scaler)

    image_nx =
      image_nx
      |> Nx.as_type({:f, 32})
      # {h, w, c} -> {c, h, w}
      |> Nx.transpose(axes: [2, 0, 1])
      # add another axis {3, 640, 640} -> {1, 3, 640, 640}
      |> Nx.new_axis(0)

    {image_nx, image_scaling}
  end

  @impl true
  def precalculate(_model_ref, _shapes, _options), do: nil

  @doc """
  Post-processes the model's raw output to produce a filtered list of detected objects.

  Options:
  * `nms_fun` - Optional custom NMS function. Must calculate detection scores as the product of the maximum class
    probability and the objectness score.
  * `prob_threshold` - Minimum probability threshold for detections
  * `iou_threshold` - IoU threshold for non-maximum suppression
  """
  @impl true
  def postprocess(%{precalculated: precalculated}, model_output, scaling_config, opts) do
    prob_threshold = Keyword.fetch!(opts, :prob_threshold)
    iou_threshold = Keyword.fetch!(opts, :iou_threshold)
    nms_fun = Keyword.get(opts, :nms_fun, &default_nms/3)

    model_output
    |> Nx.squeeze()
    |> convert_bboxes_to_cxcywh()
    |> nms_fun.(prob_threshold, iou_threshold)
    |> YOLO.FrameScalers.scale_bboxes_to_original(scaling_config)
  end

  defnp convert_bboxes_to_cxcywh(model_output) do
    bboxes = Nx.slice_along_axis(model_output, 0, 4, axis: 1)
    remainder = model_output[[.., 4..-1//1]]

    x_min = Nx.slice_along_axis(bboxes, 0, 1, axis: 1)
    y_min = Nx.slice_along_axis(bboxes, 1, 1, axis: 1)
    x_max = Nx.slice_along_axis(bboxes, 2, 1, axis: 1)
    y_max = Nx.slice_along_axis(bboxes, 3, 1, axis: 1)

    Nx.concatenate([
      (x_min + x_max) / 2,
      (y_min + y_max) / 2,
      x_max - x_min,
      y_max - y_min,
      remainder
    ], axis: 1)
  end

  @doc """
  Calculates the detection score for each prediction as the product of the maximum class
  probability and the objectness score.

  Adaptation of YOLO.NMS.filter_predictions/2, but calculates the correct score based on
  the product of the maximum class probability and the objectness score which differs from Ultralytics

  Removes prob_threshold filtering so that we can use Nx.Defn compilation for performance.
  """
  defn calculate_max_prob_score_per_prediction(predictions) do
    # {n, 4}
    bboxes = predictions[[.., 0..3]]

    # {n, 80}
    scores = predictions[[.., 4..-1//1]]

    # Per row, gets the max prob and the class with that prob
    {max_prob, max_prob_class} = Nx.top_k(scores, k: 1)

    # concatenating the columns [cx, cy, w, h, prob, class]
    Nx.concatenate([bboxes, max_prob, max_prob_class], axis: 1)
  end

  # Actually outputs {{1, n, 80}, {1, n, 4}} which we need to concatenate
  # into {{1, n, 84}}
  @impl true
  defn reshape_output({classes, bboxes}), do: Nx.concatenate([bboxes, classes], axis: 2)

  def default_nms(model_output_nx, prob_threshold, nms_threshold) do
    model_output_nx
    |> calculate_max_prob_score_per_prediction()
    |> Nx.to_list()
    |> Stream.filter(fn [_cx, _cy, _w, _h, prob, _class] -> prob >= prob_threshold end)
    |> Enum.sort_by(fn [_cx, _cy, _w, _h, prob, _class] -> prob end, :desc)
    |> YOLO.NMS.nms(nms_threshold)
  end
end
