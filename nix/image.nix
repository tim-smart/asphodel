# The OCI image: the binary, both models and the timezone database, with no
# shell. `asphodel serve` is the default command; every other subcommand
# runs with `kubectl exec <pod> -c asphodel -- asphodel <subcommand>`.
#
# `ASPHODEL_DATA_DIR` is left unset and the image has no `/data`, so a pod
# whose volume didn't mount fails at startup instead of serving an empty
# store from the container's own filesystem.
{
  dockerTools,
  tzdata,
  asphodel,
  models,
}:
dockerTools.buildLayeredImage {
  name = "asphodel";
  tag = asphodel.version;

  contents = [asphodel];

  extraCommands = ''
    mkdir -m 1777 tmp
  '';

  config = {
    Entrypoint = ["${asphodel}/bin/asphodel"];
    Cmd = ["serve"];
    # Numeric, since the image has no /etc/passwd. Give the data volume to
    # this group with the pod's `fsGroup`.
    User = "65532:65532";
    Env = [
      "PATH=/bin"
      "ASPHODEL_MODEL_DIR=${models}"
      # jiff reads IANA zones from here; the image has no /usr/share/zoneinfo.
      "TZDIR=${tzdata}/share/zoneinfo"
    ];
  };
}
