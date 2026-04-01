#!/usr/bin/env bash
# Release a new version of hyper_fingerprints.
#
# Usage:
#   ./release.sh patch    # 0.1.0 -> 0.1.1
#   ./release.sh minor    # 0.1.0 -> 0.2.0
#   ./release.sh major    # 0.1.0 -> 1.0.0
#
# What this does:
#   1. Checks that the working directory is clean
#   2. Bumps the version (via bump-my-version)
#   3. Commits the version bump
#   4. Builds a wheel and runs the full test suite (via nox)
#   5. Creates a git tag (v0.2.0 etc.)
#   6. Pushes the commit and tag to origin
#   7. GitHub Actions CI then automatically builds all platform wheels
#      and publishes to PyPI

set -euo pipefail

PART="${1:-}"

if [[ -z "$PART" ]]; then
    echo "Usage: ./release.sh <patch|minor|major>"
    echo ""
    echo "Current version: $(cat hyper_fingerprints/VERSION)"
    exit 1
fi

if [[ "$PART" != "patch" && "$PART" != "minor" && "$PART" != "major" ]]; then
    echo "ERROR: argument must be 'patch', 'minor', or 'major' (got '$PART')"
    exit 1
fi

echo "============================================"
echo "  hyper_fingerprints release"
echo "============================================"
echo ""

# --- Step 1: Check clean working directory ---
echo "[1/6] Checking working directory..."
if [[ -n "$(git status --porcelain)" ]]; then
    echo "  ERROR: working directory is not clean. Commit or stash changes first."
    echo ""
    git status --short
    exit 1
fi
echo "  OK — working directory is clean."
echo ""

# --- Step 2: Bump version ---
echo "[2/6] Bumping version ($PART)..."
echo "  Current version: $(cat hyper_fingerprints/VERSION)"
bump-my-version bump "$PART"
NEW_VERSION="$(cat hyper_fingerprints/VERSION)"
TAG="v$NEW_VERSION"
echo "  New version:     $NEW_VERSION"
echo ""

# --- Step 3: Commit version bump ---
echo "[3/6] Committing version bump..."
git add -A
git commit -m "Bump version to $NEW_VERSION"
echo "  Committed."
echo ""

# --- Step 4: Build and test ---
echo "[4/6] Building wheel and running test suite (nox -s build_test)..."
echo "  This builds a release wheel, installs it in a clean venv,"
echo "  and runs all tests against it."
echo ""
if ! nox -s build_test; then
    echo ""
    echo "  ERROR: build_test failed! Aborting release."
    echo "  The version bump commit remains but no tag was created."
    echo "  Fix the issue and try again, or reset with:"
    echo "    git reset --soft HEAD~1"
    exit 1
fi
echo ""
echo "  All tests passed."
echo ""

# --- Step 5: Tag ---
echo "[5/6] Creating tag $TAG..."
if git rev-parse "$TAG" &>/dev/null; then
    echo "  ERROR: tag $TAG already exists!"
    exit 1
fi
git tag "$TAG"
echo "  Tagged."
echo ""

# --- Step 6: Push ---
echo "[6/6] Pushing to origin..."
git push origin master --tags
echo "  Pushed."
echo ""

echo "============================================"
echo "  Released $TAG"
echo "============================================"
echo ""
echo "  CI will now build wheels for all platforms"
echo "  and publish to PyPI."
echo ""
echo "  Watch progress:"
echo "  https://github.com/the16thpythonist/hyper-fingerprints/actions"
echo ""
