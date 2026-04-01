import glob
import os

import nox

nox.options.default_venv_backend = "uv"


@nox.session(python=["3.9", "3.10", "3.11", "3.12", "3.13"])
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install(".[dev]")
    session.run("pytest")


@nox.session(python="3.11")
def build_test(session: nox.Session) -> None:
    """Build a release wheel, install it into a clean venv, and run tests."""
    wheel_dir = os.path.join(session.create_tmp(), "wheels")

    # Build the wheel using maturin (needs Rust toolchain)
    session.install("maturin")
    session.run(
        "maturin", "build", "--release",
        "--out", wheel_dir,
        external=True,
    )

    # Find the built wheel
    wheels = glob.glob(os.path.join(wheel_dir, "hyper_fingerprints-*.whl"))
    if not wheels:
        session.error("No wheel found after build")
    wheel = wheels[0]
    session.log(f"Built wheel: {wheel}")

    # Install the wheel + test dependencies into this session's venv
    session.install(wheel)
    session.install("pytest", "pytest-cov")

    # Run the full test suite against the installed wheel
    session.run("pytest", "tests/", "-v")
