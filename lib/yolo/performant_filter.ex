defmodule Yolo.PerformantFilter do
  use Rustler, otp_app: :yolo, crate: "yolo_performantfilter"

  # When your NIF is loaded, it will override this function.
  def idx_filter_greater(nx, threshold) do
    nx
    |> Nx.to_binary()
    |> filter_greater_binary(threshold)
    |> Nx.from_binary(:u64)
  end

  def filter_greater_binary(_binary, _threshold),
      do: :erlang.nif_error(:nif_not_loaded)
end
