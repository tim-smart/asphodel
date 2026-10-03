//! Static guide contracts; these do not require a cluster or GNU time.
const GUIDE: &str = include_str!("../../../docs/hermes-data-evaluation.md");

#[test]
fn export_uses_the_hermes_deployment_namespace() {
    assert!(
        !GUIDE.contains("hermes-0"),
        "the cluster has no hermes-0 StatefulSet pod"
    );
    assert!(
        GUIDE.contains("deployment/") || GUIDE.contains("deploy/"),
        "select the Deployment rather than a fixed pod"
    );
    assert!(
        GUIDE.contains("-n hermes") || GUIDE.contains("--namespace hermes"),
        "the Deployment is in namespace hermes"
    );
}

#[test]
fn timing_instructions_cover_gnu_time_and_macos() {
    assert!(
        GUIDE.contains(r#"GNU_TIME="$(nix build --no-link --print-out-paths --inputs-from . nixpkgs#time)/bin/time""#)
            && GUIDE.contains(r#""$GNU_TIME" -v asphodel replay"#),
        "resolve GNU time from the locked nixpkgs input and use it with -v"
    );
    assert!(
        GUIDE.contains("/usr/bin/time -l"),
        "include the macOS time form"
    );
}
