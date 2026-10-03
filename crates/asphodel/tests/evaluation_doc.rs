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
        GUIDE.contains("/usr/bin/time -v"),
        "keep the GNU time instructions"
    );
    assert!(
        GUIDE.contains("/usr/bin/time -l"),
        "include the macOS time form"
    );
}
